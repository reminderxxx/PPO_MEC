from scripts.run_public_alpr_pilot import normalize, distance, paired_summary
import pytest


def test_normalize_without_digit_guessing():
    assert normalize('ab-１２ 3') == 'AB123'
    assert normalize('O0') == 'O0'


def test_whole_answer_not_cherry_extracted():
    assert normalize('The plate is AB123') != 'AB123'


def test_edit_distance():
    assert distance('ABC', 'ABC') == 0
    assert distance('ABC', '') == 3
    assert distance('ABC', 'ADC') == 1
    assert distance('ABC', 'ABCD') == 1


def test_paired_counts_keep_losses():
    a = [dict(sample_id=str(i), split='locked_check', reference_characters=5, exact_match=v)
         for i, v in enumerate([True, True, False, False])]
    b = [dict(r, exact_match=v) for r, v in zip(a, [True, False, True, False])]
    counts = paired_summary(a, b)['locked_check']
    assert counts == dict(n=4, both_correct=1, adapter_only_correct=1, base_only_correct=1, both_wrong=1)


def test_paired_missing_or_duplicate_rejected():
    row = dict(sample_id='a', split='locked_check', reference_characters=5, exact_match=True)
    with pytest.raises(AssertionError):
        paired_summary([row], [])
    with pytest.raises(AssertionError):
        paired_summary([row, row], [row])
