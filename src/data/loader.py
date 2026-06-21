"""Dataset loaders for Uhura-TruthfulQA (MC1) and AfriMMLU (fixed 4-way).

Both produce a uniform list of `MCQASample`. The loaders are deliberately defensive
about HF schema: column names are verified at load time and any missing field the
question depends on raises immediately (Step 0 - a silent field drop invalidates a run).

Original English-only TruthfulQA must NOT be used here - only `masakhane/uhura-truthfulqa`.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import random
from dataclasses import dataclass, field
from typing import Any

from datasets import get_dataset_config_names, load_dataset

log = logging.getLogger(__name__)


@dataclass
class MCQASample:
    """One multiple-choice item. Works for both scoring modes.

    For `fixed_option` (AfriMMLU): `choices` has `num_choices` entries and
    `choice_labels` is ["A","B","C","D"]. For `mc1` (Uhura-TruthfulQA): `choices`
    has a *variable* number of candidate answers and `choice_labels` is None.
    """

    id: str
    language: str
    question: str
    choices: list[str]
    answer_index: int
    scoring: str  # "fixed_option" | "mc1"
    choice_labels: list[str] | None = None
    domain: str | None = None
    context: str | None = None  # passage, if the benchmark has one (usually None here)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError(f"{self.id}: empty choices")
        if not (0 <= self.answer_index < len(self.choices)):
            raise ValueError(
                f"{self.id}: answer_index {self.answer_index} out of range "
                f"for {len(self.choices)} choices"
            )
        if self.scoring == "fixed_option" and self.choice_labels is None:
            raise ValueError(f"{self.id}: fixed_option requires choice_labels")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def available_configs(hf_id: str, token: str | None = None) -> list[str]:
    """List the HF *config* names (one per language) for a dataset."""
    return get_dataset_config_names(hf_id, token=token)


def _build_category_map(hf_id: str, gen_config: str, token: str | None = None) -> dict[str, str]:
    """question.strip() -> category, from a TruthfulQA `_generation` config (all splits).

    The Uhura `_multiple_choice` configs carry no category, but the parallel
    `_generation` config does; questions match exactly within a language.
    """
    from datasets import get_dataset_split_names

    mapping: dict[str, str] = {}
    try:
        splits = get_dataset_split_names(hf_id, gen_config, token=token)
    except Exception:  # noqa: BLE001
        splits = ["test", "train"]
    for sp in splits:
        try:
            ds = load_dataset(hf_id, gen_config, split=sp, token=token)
        except Exception:  # noqa: BLE001
            continue
        if "question" in ds.column_names and "category" in ds.column_names:
            for q, c in zip(ds["question"], ds["category"]):
                if q is not None and c is not None:
                    mapping[str(q).strip()] = str(c)
    return mapping


def _as_list(value: Any) -> list[str]:
    """Coerce a choices field that may be a list or a stringified list."""
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (list, tuple)):
                return [str(v) for v in parsed]
        except (ValueError, SyntaxError):
            pass
    raise TypeError(f"Cannot interpret choices field as a list: {value!r}")


def _first_present(row: dict, keys: list[str]) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _letters(n: int) -> list[str]:
    if n <= 26:
        return [chr(ord("A") + i) for i in range(n)]
    return [str(i) for i in range(n)]


def _answer_to_index(answer: Any, choices: list[str], labels: list[str] | None) -> int:
    """Map an answer cell (int index, letter, or full-text choice) to an index."""
    # integer index
    if isinstance(answer, int):
        return answer
    if isinstance(answer, str):
        a = answer.strip()
        # numeric string
        if a.isdigit():
            return int(a)
        # single letter label, e.g. "B"
        if labels is not None and a.upper() in labels:
            return labels.index(a.upper())
        if len(a) == 1 and a.upper() in [chr(ord("A") + i) for i in range(len(choices))]:
            return ord(a.upper()) - ord("A")
        # full-text match against a choice
        for i, c in enumerate(choices):
            if c.strip() == a:
                return i
    raise ValueError(f"Cannot resolve answer {answer!r} against choices {choices!r}")


# --------------------------------------------------------------------------- #
# row -> MCQASample
# --------------------------------------------------------------------------- #
def _row_to_mc1(
    row: dict, idx: int, lang: str, has_domain: bool, category_map: dict | None = None
) -> MCQASample:
    """TruthfulQA-MC1 row -> an `mc1` multiple-choice item.

    Non-reasoning models score this by the length-normalized log-prob of each candidate
    answer text (canonical TruthfulQA MC1) - this avoids the confidence saturation that
    letter-scoring caused on Uhura's many-option questions (which flatlined the abstention
    mitigation). We still deterministically SHUFFLE candidates (seeded by language+question)
    because the released `mc1_targets` list the correct answer first, and the reasoning
    (self-consistency) path presents them as a lettered list where position would bias votes.
    """
    question = _first_present(row, ["question", "prompt", "Question"])
    if question is None:
        raise KeyError(f"No question field in row keys {list(row)}")

    mc1 = _first_present(row, ["mc1_targets", "mc1"])
    if isinstance(mc1, str):  # Uhura ships mc1_targets as a stringified dict
        try:
            mc1 = ast.literal_eval(mc1)
        except (ValueError, SyntaxError):
            mc1 = None
    if isinstance(mc1, dict) and "choices" in mc1 and "labels" in mc1:
        choices = [str(c) for c in mc1["choices"]]
        labels = [int(x) for x in mc1["labels"]]  # normalize float/np/bool to int
        if 1 not in labels:
            raise ValueError(f"{lang}-{idx}: mc1_targets has no correct label")
        correct = labels.index(1)
    else:
        choices_raw = _first_present(row, ["choices", "options", "candidates"])
        if choices_raw is None:
            raise KeyError(
                f"{lang}-{idx}: no mc1_targets and no choices field; keys={list(row)}"
            )
        choices = _as_list(choices_raw)
        labels_raw = _first_present(row, ["labels", "label"])
        if isinstance(labels_raw, (list, tuple)) and 1 in [int(x) for x in labels_raw]:
            correct = [int(x) for x in labels_raw].index(1)
        else:
            ans = _first_present(row, ["answer", "correct", "label"])
            correct = _answer_to_index(ans, choices, None)

    # deterministic shuffle so the correct answer is not always at position 0/"A"
    seed = int.from_bytes(hashlib.sha256(f"{lang}|{question}".encode()).digest()[:8], "big")
    order = list(range(len(choices)))
    random.Random(seed).shuffle(order)
    shuffled = [choices[i] for i in order]
    answer_index = order.index(correct)

    context = _first_present(row, ["context", "passage"])
    rid = str(_first_present(row, ["id", "qid"]) or f"{lang}-{idx}")
    domain = category_map.get(str(question).strip()) if category_map else None
    return MCQASample(
        id=f"{lang}-{rid}",
        language=lang,
        question=str(question),
        choices=shuffled,
        answer_index=answer_index,
        scoring="mc1",  # answer-text length-normalized log-prob (non-reasoning path)
        choice_labels=None,  # self-consistency path letters them dynamically (A..N)
        domain=domain,  # TruthfulQA topic, joined from the <lang>_generation config
        context=str(context) if context is not None else None,
        metadata={"raw_id": rid, "source": "mc1_targets_shuffled", "n_candidates": len(shuffled)},
    )


def _row_to_fixed(
    row: dict, idx: int, lang: str, num_choices: int, choice_labels: list[str], has_domain: bool
) -> MCQASample:
    question = _first_present(row, ["question", "Question", "prompt"])
    if question is None:
        raise KeyError(f"No question field in row keys {list(row)}")

    choices_raw = _first_present(row, ["choices", "options"])
    if choices_raw is not None:
        choices = _as_list(choices_raw)
    else:
        # fall back to per-option columns: option_a.. or A/B/C/D
        choices = []
        for lab in choice_labels:
            val = _first_present(row, [f"option_{lab.lower()}", lab, lab.lower(), f"choice_{lab}"])
            if val is None:
                raise KeyError(
                    f"{lang}-{idx}: cannot find option '{lab}'; row keys={list(row)}"
                )
            choices.append(str(val))

    if len(choices) != num_choices:
        raise ValueError(
            f"{lang}-{idx}: expected {num_choices} choices, got {len(choices)}: {choices}"
        )

    ans = _first_present(row, ["answer", "label", "correct", "answer_key"])
    if ans is None:
        raise KeyError(f"{lang}-{idx}: no answer field; row keys={list(row)}")
    answer_index = _answer_to_index(ans, choices, choice_labels)

    domain = _first_present(row, ["subject", "category", "topic"]) if has_domain else None
    context = _first_present(row, ["context", "passage"])
    rid = str(_first_present(row, ["id", "qid"]) or f"{lang}-{idx}")
    return MCQASample(
        id=f"{lang}-{rid}",
        language=lang,
        question=str(question),
        choices=choices,
        answer_index=answer_index,
        scoring="fixed_option",
        choice_labels=list(choice_labels),
        domain=str(domain) if domain is not None else None,
        context=str(context) if context is not None else None,
        metadata={"raw_id": rid},
    )


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def load_samples(
    dataset_cfg,
    languages: list[str] | None = None,
    token: str | None = None,
    limit: int | None = None,
) -> list[MCQASample]:
    """Load `MCQASample`s for the requested languages.

    `dataset_cfg` is the Hydra dataset node (fields: hf_id, split, scoring,
    languages, num_choices?, choice_labels?, has_domain, hf_config_map?).

    Samples are tagged with the canonical language code (e.g. "eng"); if the HF
    config name differs (e.g. Uhura's "en_multiple_choice"), provide `hf_config_map`
    mapping canonical code -> HF config name.
    """
    hf_id = dataset_cfg.hf_id
    split = dataset_cfg.split
    scoring = dataset_cfg.scoring
    has_domain = bool(getattr(dataset_cfg, "has_domain", False))
    cfg_map = getattr(dataset_cfg, "hf_config_map", None)
    cfg_map = dict(cfg_map) if cfg_map is not None else None
    cat_cfg_map = getattr(dataset_cfg, "hf_category_config_map", None)
    cat_cfg_map = dict(cat_cfg_map) if cat_cfg_map is not None else None
    langs = list(languages) if languages else list(dataset_cfg.languages)

    samples: list[MCQASample] = []
    for lang in langs:
        hf_config = cfg_map[lang] if (cfg_map and lang in cfg_map) else lang
        # topic/category per question (Uhura: joined from the <lang>_generation config)
        category_map = None
        if cat_cfg_map and lang in cat_cfg_map:
            category_map = _build_category_map(hf_id, cat_cfg_map[lang], token)
        try:
            ds = load_dataset(hf_id, hf_config, split=split, token=token)
        except Exception as exc:  # noqa: BLE001 - surface available configs
            try:
                configs = available_configs(hf_id, token=token)
            except Exception:  # noqa: BLE001
                configs = ["<could not list configs>"]
            raise ValueError(
                f"Could not load config '{hf_config}' (lang='{lang}', split='{split}') "
                f"for '{hf_id}'.\nAvailable configs: {configs}\n"
                f"Fix the languages / hf_config_map in the dataset config to match these."
            ) from exc

        n_before = len(samples)
        for idx, row in enumerate(ds):
            if limit is not None and idx >= limit:
                break
            if scoring == "mc1":
                samples.append(_row_to_mc1(row, idx, lang, has_domain, category_map))
            elif scoring == "fixed_option":
                samples.append(
                    _row_to_fixed(
                        row,
                        idx,
                        lang,
                        int(dataset_cfg.num_choices),
                        list(dataset_cfg.choice_labels),
                        has_domain,
                    )
                )
            else:
                raise ValueError(f"Unknown scoring mode: {scoring}")
        log.info("Loaded %d samples for %s/%s", len(samples) - n_before, hf_id, lang)

    if not samples:
        raise ValueError(f"No samples loaded for {hf_id} langs={langs}")
    return samples


def print_one(samples: list[MCQASample], language: str | None = None) -> None:
    """Print one full sample verbatim - the Step 0 data-integrity check."""
    sample = next((s for s in samples if language is None or s.language == language), None)
    if sample is None:
        print(f"No sample for language={language}")
        return
    print("=" * 70)
    print(f"id={sample.id}  language={sample.language}  scoring={sample.scoring}")
    print(f"domain={sample.domain}  context={sample.context!r}")
    print(f"\nQUESTION:\n{sample.question}")
    print(f"\nCHOICES ({len(sample.choices)}):")
    for i, c in enumerate(sample.choices):
        mark = " <-- CORRECT" if i == sample.answer_index else ""
        label = sample.choice_labels[i] if sample.choice_labels else str(i)
        print(f"  [{label}] {c}{mark}")
    print("=" * 70)
