"""ModelRunner - open-weight inference with three confidence modes.

1. fixed_option (AfriMMLU, non-reasoning): next-token logits restricted to the
   option-letter tokens (leading-space variants), softmax over options. One pass.
2. mc1 (Uhura-TruthfulQA, non-reasoning): length-normalized log-prob of each
   candidate answer string, softmax across that question's candidate set.
3. self-consistency (Qwen3-4B-Thinking): sample N traces, confidence = modal-answer
   agreement; also parse a verbalized "0-100%".

Confidence-scoring paths (1 & 2) use a plain completion prompt ending in "Answer:" and
read the leading-space answer-token (" A"), matching HELM / lm-eval-harness conventions
and the validated GCP skill note. The chat template is applied only on the generation
(self-consistency) path.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.loader import MCQASample

log = logging.getLogger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
_ANSWER_RE = re.compile(r"answer\s*[:\-]?\s*([A-Za-z0-9]+)", re.IGNORECASE)
_CONF_RE = re.compile(r"confidence\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)?)\s*%?", re.IGNORECASE)


@dataclass
class Prediction:
    id: str
    language: str
    scoring: str
    mode: str  # "fixed_option" | "mc1" | "self_consistency"
    predicted_index: int  # -1 if unparsed
    answer_index: int
    correct: bool
    confidence: float  # max prob / modal-vote fraction
    entropy: float
    margin: float  # top1 - top2
    n_choices: int
    probs: list[float]
    domain: str | None = None
    verbalized_confidence: float | None = None  # in [0,1], self-consistency only
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("probs")  # probs saved separately in logits npz
        d.pop("metadata")
        return d


def _letters(n: int) -> list[str]:
    if n <= 26:
        return [chr(ord("A") + i) for i in range(n)]
    return [str(i) for i in range(n)]


def _options_block(choices: list[str], labels: list[str]) -> str:
    # "A: choice" matches the IrokoBench AfriMMLU prompt (Table 12, t1)
    return "\n".join(f"{lab}: {c}" for lab, c in zip(labels, choices))


def _entropy(probs: torch.Tensor) -> float:
    p = probs.clamp_min(1e-12)
    return float(-(p * p.log()).sum().item())


def _margin(probs: torch.Tensor) -> float:
    if probs.numel() < 2:
        return float(probs.max().item())
    top2 = torch.topk(probs, 2).values
    return float((top2[0] - top2[1]).item())


class ModelRunner:
    def __init__(self, model_cfg, token: str | None = None, seed: int = 1234):
        self.cfg = model_cfg
        self.reasoning = bool(model_cfg.reasoning)
        self.use_chat_template = bool(getattr(model_cfg, "use_chat_template", True))
        self.seed = seed

        dtype = _DTYPES[str(model_cfg.dtype)]
        log.info("Loading tokenizer %s", model_cfg.hf_id)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_cfg.hf_id,
            trust_remote_code=bool(getattr(model_cfg, "trust_remote_code", False)),
            token=token,
        )
        # left padding so the last-position logit is the real final token (skill #11)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        log.info("Loading model %s (%s, %s)", model_cfg.hf_id, dtype, model_cfg.attn_implementation)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_cfg.hf_id,
            dtype=dtype,  # transformers >=5 renamed torch_dtype -> dtype
            attn_implementation=str(getattr(model_cfg, "attn_implementation", "sdpa")),
            device_map=str(getattr(model_cfg, "device_map", "auto")),
            trust_remote_code=bool(getattr(model_cfg, "trust_remote_code", False)),
            token=token,
        )
        self.model.eval()
        self.device = next(self.model.parameters()).device
        self.max_length = int(model_cfg.max_length)

    # ----------------------------------------------------------------- #
    # prompt helpers
    # ----------------------------------------------------------------- #
    def _apply_chat(self, text: str) -> str:
        if self.use_chat_template and getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return text

    def get_answer_token_ids(self, labels: list[str]) -> list[int]:
        """Token id for each option letter as the model emits it after 'Answer:'.

        Prefer the leading-space variant (' A'); fall back to bare; error if multi-token.
        (skill #13)
        """
        ids = []
        for lab in labels:
            with_space = self.tokenizer.encode(" " + lab, add_special_tokens=False)
            if len(with_space) == 1:
                ids.append(with_space[0])
                continue
            bare = self.tokenizer.encode(lab, add_special_tokens=False)
            if len(bare) == 1:
                log.warning("Tokenizer splits ' %s'; falling back to bare '%s'", lab, lab)
                ids.append(bare[0])
            else:
                raise ValueError(f"Cannot represent ' {lab}' as a single token")
        return ids

    # ----------------------------------------------------------------- #
    # public entry
    # ----------------------------------------------------------------- #
    def run(
        self,
        samples: list[MCQASample],
        prompt_template: str,
        thinking_template: str,
        batch_size: int = 8,
    ) -> list[Prediction]:
        if self.reasoning:
            return self._run_self_consistency(samples, thinking_template)
        if samples[0].scoring == "fixed_option":
            return self._run_fixed_option(samples, prompt_template, batch_size)
        return self._run_mc1(samples, prompt_template, batch_size)

    # ----------------------------------------------------------------- #
    # mode 1: fixed-option logit readout
    # ----------------------------------------------------------------- #
    @torch.inference_mode()
    def _run_fixed_option(
        self, samples: list[MCQASample], template: str, batch_size: int
    ) -> list[Prediction]:
        preds: list[Prediction] = []
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            prompts = [
                template.format(
                    question=s.question,
                    options_block=_options_block(s.choices, s.choice_labels),
                    subject=(s.domain or "general knowledge"),  # AfriMMLU t1 uses {subject}
                )
                for s in batch
            ]
            enc = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)
            logits = self.model(**enc).logits[:, -1, :]  # (B, V)
            for i, s in enumerate(batch):
                answer_ids = self.get_answer_token_ids(s.choice_labels)
                opt_logits = logits[i, answer_ids]  # (K,)
                probs = F.softmax(opt_logits.float(), dim=-1)
                pred = int(probs.argmax().item())
                preds.append(
                    Prediction(
                        id=s.id,
                        language=s.language,
                        scoring=s.scoring,
                        mode="fixed_option",
                        predicted_index=pred,
                        answer_index=s.answer_index,
                        correct=(pred == s.answer_index),
                        confidence=float(probs.max().item()),
                        entropy=_entropy(probs),
                        margin=_margin(probs),
                        n_choices=len(s.choices),
                        probs=[float(x) for x in probs.tolist()],
                        domain=s.domain,
                    )
                )
        return preds

    # ----------------------------------------------------------------- #
    # mode 2: mc1 length-normalized candidate likelihood
    # ----------------------------------------------------------------- #
    @torch.inference_mode()
    def _run_mc1(
        self, samples: list[MCQASample], template: str, batch_size: int
    ) -> list[Prediction]:
        preds: list[Prediction] = []
        for s in samples:
            stem = template.format(question=s.question)
            norm_logprobs = [self._candidate_logprob(stem, c) for c in s.choices]
            t = torch.tensor(norm_logprobs, dtype=torch.float32)
            probs = F.softmax(t, dim=-1)
            pred = int(probs.argmax().item())
            preds.append(
                Prediction(
                    id=s.id,
                    language=s.language,
                    scoring=s.scoring,
                    mode="mc1",
                    predicted_index=pred,
                    answer_index=s.answer_index,
                    correct=(pred == s.answer_index),
                    confidence=float(probs.max().item()),
                    entropy=_entropy(probs),
                    margin=_margin(probs),
                    n_choices=len(s.choices),
                    probs=[float(x) for x in probs.tolist()],
                    domain=s.domain,
                )
            )
        return preds

    @torch.inference_mode()
    def _candidate_logprob(self, stem: str, candidate: str) -> float:
        """Length-normalized sum of token log-probs of `candidate` continuing `stem`."""
        prefix_ids = self.tokenizer(stem, add_special_tokens=True).input_ids
        full_text = stem + " " + candidate.strip()
        full_ids = self.tokenizer(full_text, add_special_tokens=True).input_ids

        # robust boundary: longest common prefix (tokenization isn't always prefix-stable)
        b = 0
        for a, c in zip(prefix_ids, full_ids):
            if a != c:
                break
            b += 1
        cand_ids = full_ids[b:]
        if len(cand_ids) == 0:  # degenerate; fall back to last token
            cand_ids = full_ids[-1:]
            b = len(full_ids) - 1

        if len(full_ids) > self.max_length:
            # left-truncate the prefix only; candidate tokens sit at the end
            k = len(full_ids) - self.max_length
            full_ids = full_ids[k:]
            b = max(0, b - k)

        input_ids = torch.tensor([full_ids], device=self.device)
        logits = self.model(input_ids=input_ids).logits[0]  # (T, V)
        logprobs = F.log_softmax(logits.float(), dim=-1)
        # token at position j is predicted by logits[j-1]; never read logits[-1]
        start = max(1, b)
        total = 0.0
        for j in range(start, len(full_ids)):
            total += float(logprobs[j - 1, full_ids[j]].item())
        return total / max(1, len(full_ids) - start)

    # ----------------------------------------------------------------- #
    # mode 3: self-consistency (reasoning model)
    # ----------------------------------------------------------------- #
    @torch.inference_mode()
    def _run_self_consistency(
        self, samples: list[MCQASample], thinking_template: str
    ) -> list[Prediction]:
        torch.manual_seed(self.seed)
        n_samples = int(self.cfg.n_samples)
        temperature = float(self.cfg.temperature)
        top_p = float(getattr(self.cfg, "top_p", 0.95))
        max_new = int(self.cfg.max_new_tokens)

        preds: list[Prediction] = []
        for s in samples:
            labels = s.choice_labels or _letters(len(s.choices))
            block = _options_block(s.choices, labels)
            prompt = thinking_template.format(question=s.question, options_block=block)
            text = self._apply_chat(prompt)
            enc = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=self.max_length
            ).to(self.device)
            gen = self.model.generate(
                **enc,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=n_samples,
                max_new_tokens=max_new,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            completions = self.tokenizer.batch_decode(
                gen[:, enc.input_ids.shape[1] :], skip_special_tokens=True
            )
            votes: list[int] = []
            conf_vals: list[float] = []
            for comp in completions:
                idx = self._parse_answer(comp, labels)
                if idx is not None:
                    votes.append(idx)
                vc = self._parse_confidence(comp)
                if vc is not None:
                    conf_vals.append(vc)

            counts = [0] * len(s.choices)
            for v in votes:
                if 0 <= v < len(s.choices):
                    counts[v] += 1
            total_votes = sum(counts)
            if total_votes == 0:
                preds.append(
                    Prediction(
                        id=s.id, language=s.language, scoring=s.scoring, mode="self_consistency",
                        predicted_index=-1, answer_index=s.answer_index, correct=False,
                        confidence=0.0, entropy=float("nan"), margin=0.0,
                        n_choices=len(s.choices), probs=[0.0] * len(s.choices), domain=s.domain,
                        verbalized_confidence=(
                            sum(conf_vals) / len(conf_vals) / 100.0 if conf_vals else None
                        ),
                        metadata={"unparsed": True, "n_samples": n_samples},
                    )
                )
                continue
            probs_t = torch.tensor([c / total_votes for c in counts], dtype=torch.float32)
            pred = int(probs_t.argmax().item())
            preds.append(
                Prediction(
                    id=s.id, language=s.language, scoring=s.scoring, mode="self_consistency",
                    predicted_index=pred, answer_index=s.answer_index,
                    correct=(pred == s.answer_index),
                    confidence=float(probs_t.max().item()),
                    entropy=_entropy(probs_t), margin=_margin(probs_t),
                    n_choices=len(s.choices), probs=[float(x) for x in probs_t.tolist()],
                    domain=s.domain,
                    verbalized_confidence=(sum(conf_vals) / len(conf_vals) / 100.0 if conf_vals else None),
                    metadata={"n_votes": total_votes, "n_samples": n_samples},
                )
            )
        return preds

    @staticmethod
    def _parse_answer(text: str, labels: list[str]) -> int | None:
        upper = [lab.upper() for lab in labels]
        # take the LAST "Answer: X" so we read the final decision, not one inside <think>
        matches = _ANSWER_RE.findall(text)
        for tok in reversed(matches):
            t = tok.strip().upper()
            if t in upper:
                return upper.index(t)
        return None

    @staticmethod
    def _parse_confidence(text: str) -> float | None:
        matches = _CONF_RE.findall(text)
        if not matches:
            return None
        try:
            val = float(matches[-1])
        except ValueError:
            return None
        return max(0.0, min(100.0, val))
