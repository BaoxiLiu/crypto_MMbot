"""Controlled order intents for OMS/receipt tests, not a trading strategy.

Four ticks behind each touch make the synthetic tape's fill/no-fill events
explicit. No AS formula, volatility pricing, inventory policy or alpha model.
Actual AS/NH/HJB prices are covered by plugin and frozen trajectory tests.
"""
from decimal import Decimal as D
from app.market_making.core import floor_step
from app.market_making.strategies.base import QuoteStrategy
from app.market_making.contracts import Desired


class QuoteFixture(QuoteStrategy):
    def desired(self, position, ns):
        return self.desired_with_alpha(position, ns, 0.)

    def desired_with_alpha(self, position, ns, alpha_bp):
        state = self.obs.state(ns)
        if not state.ready:
            self.last = {'reason': 'not_ready'}
            return {}
        shift = state.mid * D(str(alpha_bp)) / D(10000)
        quotes = {}
        for side, touch in ((1, state.bid), (-1, state.ask)):
            price = self.rules.price(touch - side * 4 * self.rules.tick + shift, side)
            qty = floor_step(self.config.as_config.order_notional / price, self.rules.step)
            quotes[side] = Desired(side, price, qty)
        self.last = {'reason': 'controlled_test_intents'}
        return quotes
