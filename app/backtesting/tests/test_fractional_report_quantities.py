from decimal import Decimal

import pytest

from app.backtesting.report_sections import submitted_quantity


def test_json_float_point_three_does_not_look_overfilled():
    assert Decimal('.3') > Decimal(.3)  # The original false-overfill trigger.
    assert submitted_quantity(.3, '.1') == Decimal('.3')


def test_float_multiplication_noise_still_counts_as_full_order():
    assert submitted_quantity(123.80000000000001, '.1') == Decimal('123.8')
    assert submitted_quantity('123.8', '.1') == Decimal('123.8')


@pytest.mark.parametrize('value', [.35, -1., float('nan')])
def test_invalid_and_off_lot_requests_are_rejected(value):
    with pytest.raises(ValueError):
        submitted_quantity(value, '.1')


def test_missing_legacy_lot_metadata_uses_decimal_text_not_binary_expansion():
    assert submitted_quantity(.3) == Decimal('.3')
