"""One decision policy: preserve source R/F/I and add explicit optional model ports."""
from dataclasses import replace
import math
from .contracts import PriceForecast, SideAssessment


class DecisionPipeline:
    """Price forecast precedes quotes; order evaluation follows concrete quotes.

    None means a disabled/unavailable model and preserves the AS baseline. A
    configured evaluator's missing/expired side is denied for opening exposure;
    reducing inventory is preserved and explicitly marked reduce-only.
    """
    def __init__(self, price_model=None, order_evaluator=None):
        self.price_model = price_model
        self.order_evaluator = order_evaluator
        self.last = {}

    def forecast(self, context):
        self.last = {'ns': context.ns, 'stages': ['state', 'features', 'price_forecast']}
        signal = self.price_model.predict(context) if self.price_model else None
        if signal is None:
            self.last['forecast'] = 'disabled_or_unavailable'
            return 0.0
        if not isinstance(signal, PriceForecast):
            raise ValueError('PriceForecast contract required')
        if (not math.isfinite(signal.alpha_bp) or not signal.model_id
                or type(signal.as_of_ns) is not int or type(signal.valid_until_ns) is not int
                or not 0 <= signal.as_of_ns <= context.ns <= signal.valid_until_ns):
            self.last['forecast'] = 'invalid'
            raise ValueError('Future/expired/invalid price forecast')
        self.last['forecast'] = signal.model_id
        return signal.alpha_bp

    def choose(self, context, candidates):
        self.last['stages'].extend(['as_candidates', 'regime_flow_inventory', 'order_evaluation', 'decision'])
        if self.order_evaluator is None:
            self.last['assessment'] = 'disabled'
            return candidates
        assessments = self.order_evaluator.evaluate(context, tuple(candidates.values()))
        rows = () if assessments is None else tuple(assessments)
        approved, seen = set(), set()
        for row in rows:
            if not isinstance(row, SideAssessment) or row.side not in candidates or row.side in seen:
                raise ValueError('Invalid/duplicate side assessment')
            seen.add(row.side)
            if (type(row.allow_opening) is not bool or type(row.as_of_ns) is not int
                    or type(row.valid_until_ns) is not int):
                raise ValueError('Invalid assessment types')
            timely = 0 <= row.as_of_ns <= context.ns <= row.valid_until_ns
            if row.as_of_ns > context.ns:
                raise ValueError('Future order assessment')
            if timely and row.allow_opening:
                approved.add(row.side)
        result = {}
        for side, target in candidates.items():
            if side in approved:
                result[side] = target
            elif side * context.position < 0:
                result[side] = replace(target, qty=min(target.qty, abs(context.position)),
                                       reduce_only=True, reason=target.reason + '+model_reduce_only')
        self.last['assessment'] = 'applied'
        self.last['candidate_sides'], self.last['accepted_sides'] = len(candidates), len(result)
        return result
