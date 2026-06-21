import pytest

from src.data.loader import (
    MCQASample,
    _answer_to_index,
    _as_list,
    _row_to_fixed,
    _row_to_mc1,
)


def test_as_list_handles_list_and_string():
    assert _as_list(["a", "b"]) == ["a", "b"]
    assert _as_list("['a', 'b', 'c']") == ["a", "b", "c"]
    with pytest.raises(TypeError):
        _as_list(42)


def test_answer_to_index_variants():
    choices = ["cat", "dog", "fish", "bird"]
    labels = ["A", "B", "C", "D"]
    assert _answer_to_index(1, choices, labels) == 1
    assert _answer_to_index("B", choices, labels) == 1
    assert _answer_to_index("2", choices, labels) == 2
    assert _answer_to_index("fish", choices, None) == 2
    with pytest.raises(ValueError):
        _answer_to_index("nonexistent", choices, labels)


def test_mcqasample_validates_answer_index():
    with pytest.raises(ValueError):
        MCQASample(
            id="x", language="eng", question="q", choices=["a", "b"],
            answer_index=5, scoring="mc1",
        )


def test_fixed_option_requires_labels():
    with pytest.raises(ValueError):
        MCQASample(
            id="x", language="eng", question="q", choices=["a", "b", "c", "d"],
            answer_index=0, scoring="fixed_option", choice_labels=None,
        )


def test_row_to_fixed_with_choices_list():
    row = {"question": "2+2?", "choices": ["3", "4", "5", "6"], "answer": "B", "subject": "math"}
    s = _row_to_fixed(row, 0, "eng", 4, ["A", "B", "C", "D"], has_domain=True)
    assert s.answer_index == 1
    assert s.choices[s.answer_index] == "4"
    assert s.domain == "math"
    assert s.scoring == "fixed_option"


def test_row_to_fixed_with_separate_option_columns():
    row = {
        "question": "q", "option_a": "w", "option_b": "x", "option_c": "y", "option_d": "z",
        "answer": 2,
    }
    s = _row_to_fixed(row, 1, "yor", 4, ["A", "B", "C", "D"], has_domain=False)
    assert s.choices == ["w", "x", "y", "z"]
    assert s.answer_index == 2


def test_row_to_mc1_with_mc1_targets():
    # mc1_targets -> scoring="mc1" (answer-text log-prob); shuffled but text-tracked; topic via category
    row = {
        "question": "Is the sky green?",
        "mc1_targets": {"choices": ["No", "Yes"], "labels": [1, 0]},
    }
    cat = {"Is the sky green?": "Misconceptions"}
    s = _row_to_mc1(row, 0, "eng", has_domain=True, category_map=cat)
    assert s.scoring == "mc1"
    assert s.choice_labels is None
    assert set(s.choices) == {"No", "Yes"}
    assert s.choices[s.answer_index] == "No"  # correct answer tracked through the shuffle
    assert s.domain == "Misconceptions"  # topic joined from category_map


def test_row_to_mc1_shuffle_is_deterministic():
    row = {"question": "q?", "mc1_targets": {"choices": ["x", "y", "z", "w"], "labels": [1, 0, 0, 0]}}
    a = _row_to_mc1(row, 0, "eng", has_domain=False)
    b = _row_to_mc1(row, 0, "eng", has_domain=False)
    assert a.choices == b.choices and a.answer_index == b.answer_index  # sha256-seeded
    assert a.choices[a.answer_index] == "x"


def test_row_to_mc1_with_choices_and_label():
    row = {"question": "q", "choices": ["a", "b", "c"], "labels": [0, 1, 0]}
    s = _row_to_mc1(row, 3, "swa", has_domain=False)
    assert s.scoring == "mc1"
    assert s.choices[s.answer_index] == "b"
