"""Causal, LOCAL-receipt minute trade bars; never exchange-time revised K-lines."""
import numpy as np

MINUTE = 60_000_000_000
NAMES = [
    'side_rsi14_centered', 'side_rsi7_centered', 'side_rsi14_change',
    'rsi14_reversal_extreme', 'side_return1_bp', 'side_return5_bp',
    'side_return15_bp', 'absolute_return1_bp', 'range1_bp',
    'side_body1_bp', 'side_close_minus_vwap1_bp', 'side_close_minus_sma20_bp',
    'realized_volatility20_bp', 'range_relative20', 'log_volume1',
    'volume_relative20', 'log_trade_count1', 'side_aggressor_imbalance1',
]


def local_trades(data, start_ns, end_ns):
    """Cache schema6 flags: LOCAL bit30, BUY bit29, TRADE type2.

    EXCH-only replay copies must never double-count LOCAL receipts.
    """
    mask = ((data['ev'] & 255) == 2) & ((data['ev'] & (1 << 30)) != 0)
    mask &= (data['local_ts'] >= start_ns) & (data['local_ts'] <= end_ns)
    a = data[mask]
    return (a['local_ts'].copy(), a['px'].copy(), a['qty'].copy(),
            (a['ev'] & (1 << 29)) != 0)


def bars(times, prices, quantities, buys, start_ns, end_ns):
    """All full certified minutes; zero-trade bars carry last *observed* trade.

    start/end describe one continuous certified segment, not a calendar file.
    No state crosses a gap. A trade received exactly at close belongs to next bar.
    """
    times = np.asarray(times, dtype=np.int64)
    prices, quantities = np.asarray(prices, float), np.asarray(quantities, float)
    buys = np.asarray(buys, bool)
    if not (len(times) == len(prices) == len(quantities) == len(buys)):
        raise ValueError('Trade array lengths differ')
    if np.any(np.diff(times) < 0) or np.any(times < start_ns) or np.any(times > end_ns):
        raise ValueError('Trades outside continuous LOCAL certified clock')
    if not (np.isfinite(prices).all() and np.isfinite(quantities).all()
            and np.all(prices > 0) and np.all(quantities > 0)):
        raise ValueError('Invalid trade')
    first = ((int(start_ns) + MINUTE - 1) // MINUTE) * MINUTE
    last = (int(end_ns) // MINUTE) * MINUTE
    opens = np.arange(first, last, MINUTE, dtype=np.int64)
    ends = opens + MINUTE
    out = {'open_ns': opens, 'close_ns': ends}
    for name in ['open', 'high', 'low', 'close', 'vwap']:
        out[name] = np.full(len(ends), np.nan)
    for name in ['volume', 'quote_volume', 'buy_volume', 'trade_count']:
        out[name] = np.zeros(len(ends))
    for i, (a, b) in enumerate(zip(opens, ends)):
        lo, hi = np.searchsorted(times, [a, b], side='left')
        if hi > lo:
            p, q = prices[lo:hi], quantities[lo:hi]
            out['open'][i], out['close'][i] = p[0], p[-1]
            out['high'][i], out['low'][i] = np.max(p), np.min(p)
            out['volume'][i] = np.sum(q)
            out['quote_volume'][i] = np.sum(p * q)
            out['buy_volume'][i] = np.sum(q[buys[lo:hi]])
            out['vwap'][i] = out['quote_volume'][i] / out['volume'][i]
            out['trade_count'][i] = hi - lo
        elif hi:
            # Also permits a known price from the discarded first partial minute.
            for name in ['open', 'high', 'low', 'close', 'vwap']:
                out[name][i] = prices[hi - 1]
    return out


def rsi(close, period):
    """Wilder RSI; seed from period changes, reset at every missing close."""
    out = np.full(len(close), np.nan)
    gains, losses = [], []
    avg_gain = avg_loss = None
    for i in range(1, len(close)):
        if not np.isfinite(close[i - 1:i + 1]).all():
            gains, losses, avg_gain, avg_loss = [], [], None, None
            continue
        change = close[i] - close[i - 1]
        gain, loss = max(change, 0.), max(-change, 0.)
        if avg_gain is None:
            gains.append(gain); losses.append(loss)
            if len(gains) < period:
                continue
            avg_gain, avg_loss = float(np.mean(gains)), float(np.mean(losses))
        else:
            avg_gain = ((period - 1) * avg_gain + gain) / period
            avg_loss = ((period - 1) * avg_loss + loss) / period
        out[i] = (50. if avg_gain == avg_loss == 0 else
                  100. if avg_loss == 0 else
                  100. - 100. / (1. + avg_gain / avg_loss))
    return out


def indicators(bar):
    """Buy-oriented features; caller changes signed features for sell queries."""
    c = bar['close']; n = len(c)
    X = np.full((n, len(NAMES)), np.nan)
    r14, r7 = rsi(c, 14), rsi(c, 7)
    for i in range(n):
        if not np.isfinite(c[i]):
            continue
        X[i, 0] = (r14[i] - 50.) / 50.
        X[i, 1] = (r7[i] - 50.) / 50.
        if i: X[i, 2] = (r14[i] - r14[i - 1]) / 50.
        if np.isfinite(r14[i]): X[i, 3] = float(r14[i] <= 30.)
        for k, horizon in [(4, 1), (5, 5), (6, 15)]:
            if i >= horizon and np.isfinite(c[i - horizon:i + 1]).all():
                X[i, k] = (c[i] / c[i - horizon] - 1.) * 10000.
        X[i, 7] = abs(X[i, 4])
        X[i, 8] = (bar['high'][i] - bar['low'][i]) / c[i] * 10000.
        X[i, 9] = (c[i] / bar['open'][i] - 1.) * 10000.
        X[i, 10] = (c[i] / bar['vwap'][i] - 1.) * 10000.
        X[i, 14] = np.log1p(bar['volume'][i])
        X[i, 16] = np.log1p(bar['trade_count'][i])
        X[i, 17] = (2 * bar['buy_volume'][i] / bar['volume'][i] - 1.
                    if bar['volume'][i] else 0.)
        if i >= 19 and np.isfinite(c[i - 19:i + 1]).all():
            z = c[i - 19:i + 1]
            X[i, 11] = (c[i] / np.mean(z) - 1.) * 10000.
            X[i, 12] = np.std(np.diff(np.log(z)), ddof=1) * 10000.
            ranges = (bar['high'][i - 19:i + 1] - bar['low'][i - 19:i + 1]) / z
            X[i, 13] = X[i, 8] / (np.mean(ranges) * 10000.) if np.mean(ranges) else 0.
            mean_volume = np.mean(bar['volume'][i - 19:i + 1])
            X[i, 15] = bar['volume'][i] / mean_volume if mean_volume else 0.
    return X


def align(bar, X, decisions, sides):
    """Most recent *closed* bar, explicit side orientation and cutoff evidence."""
    decisions, sides = np.asarray(decisions), np.asarray(sides)
    if len(decisions) != len(sides) or not np.isin(sides, [-1, 1]).all():
        raise ValueError('Invalid decision/side grid')
    ix = np.searchsorted(bar['close_ns'], decisions, side='right') - 1
    good = ix >= 0
    out = np.full((len(ix), len(NAMES)), np.nan)
    used = np.full(len(ix), -1, dtype=np.int64)
    raw_rsi = np.full(len(ix), np.nan)
    if good.any():
        out[good] = X[ix[good]]
        used[good] = bar['close_ns'][ix[good]]
        raw_rsi[good] = out[good, 0] * 50. + 50.
        signed = [0, 1, 2, 4, 5, 6, 9, 10, 11, 17]
        for col in signed: out[good, col] *= sides[good]
        out[good, 3] = np.where(np.isfinite(raw_rsi[good]),
            np.where(sides[good] == 1, raw_rsi[good] <= 30., raw_rsi[good] >= 70.), np.nan)
    assert np.all(used[good] <= decisions[good])
    return out, used, raw_rsi
