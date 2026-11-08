"""Public opposite-aggressor prints at a candidate price, never fill inference."""
import numpy as np


def query(trades, left, right, side, price_tick, tick):
    if side not in (1, -1) or left >= right or tick <= 0:
        raise ValueError('Nonempty window, side and positive tick required')
    times = trades['local_ts']
    if np.any(times[1:] < times[:-1]):
        raise ValueError('Public print clock must be ordered')
    a = np.searchsorted(times, left, side='left')
    b = np.searchsorted(times, right, side='left')
    sample = trades[a:b]
    buy = (sample['ev'] & (1 << 29)) != 0
    sell = (sample['ev'] & (1 << 28)) != 0
    if np.any((sample['ev'] & 255) != 2) or np.any(buy == sell):
        raise ValueError('Only uniquely sided public trade events are allowed')
    aggressor = np.where((sample['ev'] & (1 << 29)) != 0, 1, -1)
    values = sample['px']/tick
    rounded = np.rint(values)
    if (np.any(~np.isfinite(values)) or np.any(~np.isfinite(sample['qty'])) or
            np.any(np.abs(values-rounded) > 1e-6) or np.any(sample['qty'] <= 0)):
        raise ValueError('Off-tick or nonpositive certified trade')
    opposite = aggressor == -side
    reached = opposite & (side*(rounded-price_tick) <= 0)
    selected = sample[reached]
    return dict(public_prints=len(sample), opposite_prints=int(opposite.sum()),
                reaching_prints=int(reached.sum()), reaching_qty=float(selected['qty'].sum()),
                first_reaching_local_ns=int(selected['local_ts'][0]) if len(selected) else None)
