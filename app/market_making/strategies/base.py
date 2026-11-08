"""Shared causal observation and lifecycle state; no strategy pricing policy."""
from collections import Counter
from decimal import Decimal as D
from ..core import floor_step
from ..fusion.signals import Observations


class QuoteStrategy:
    """Minimal lifecycle: book/reset/desired plus explicitly optional hooks.

    desired(position, ns) returns side -> Desired using only visible state.
    Strategies own observations and policy state, never the execution ledger.
    """
    requires_quote_depth = False

    def __init__(self,rules,config,factors):
        self.rules,self.config,self.factors=rules,config,factors
        self.obs=Observations(config,rules.tick)
        self.reference_qty=None
        self.counts=Counter()
        self.last={}

    def reset_market(self):self.obs.reset()

    def desired(self, position, ns):
        raise NotImplementedError('Strategy must implement its quote policy')

    def next_deadline_ns(self):
        """Optional observation timer; execution does not inspect private fields."""
        return None

    def book(self,ns,bid,ask):
        self.obs.book(ns,bid,ask)
        if self.reference_qty is None:
            self.reference_qty=floor_step(self.config.as_config.order_notional/self.obs.mid,self.rules.step)
            if self.reference_qty<self.rules.minimum_qty:raise ValueError('reference size too small')

    def half_spread_floor_ticks(self,mid_ticks):
        return mid_ticks*max(self.config.as_config.half_spread_floor,float(self.config.as_config.maker_fee))

