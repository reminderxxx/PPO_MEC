from scripts.audit_public_alpr_pair import canonical_answer, edits


def test_independent_edit_distance():
    assert edits('', 'ABC') == 3
    assert edits('ABC', '') == 3
    assert edits('ABC', 'ABC') == 0
    assert edits('ABC', 'ADC') == 1
    assert edits('ABC', 'AC') == 1


def test_format_only_normalization_does_not_hide_letters():
    assert canonical_answer('ab - １２') == 'AB12'
    assert canonical_answer('O0') == 'O0'
    assert canonical_answer('plate: AB12') != 'AB12'
