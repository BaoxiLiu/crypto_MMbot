"""Timers supplement event decisions: silence must not disable stale-data risk."""


def next_timer_ns(kernel, now):
    """Earliest future policy deadline, with a one-second health heartbeat.

    This is not a quote polling interval. Every market/private event may cause
    an earlier decision. Past deadlines are not rearmed in a busy loop.
    """
    config=kernel.config
    obs=kernel.strategy.obs
    deadlines=[now+1_000_000_000]
    deadline=kernel.strategy.next_deadline_ns()
    if deadline is not None:deadlines.append(deadline)
    if obs.book_ns is not None:
        deadlines.append(obs.book_ns+config.as_config.stale_ns+1)
    if kernel.mark_ns is not None:
        deadlines.append(kernel.mark_ns+config.mark_stale_ns+1)
    deadlines.extend(obs.flow_veto.values())
    if obs.trades:deadlines.append(obs.trades[0][0]+config.flow_window_ns+1)
    deadlines.extend(o.created_ns+config.min_quote_lifetime_ns for o in kernel.oms.active.values())
    if kernel.oms.new_times:deadlines.append(kernel.oms.new_times[0]+60_000_000_000)
    return min(t for t in deadlines if t>now)
