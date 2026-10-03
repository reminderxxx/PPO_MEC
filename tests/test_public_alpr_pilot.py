from scripts.run_public_alpr_pilot import normalize, distance


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
