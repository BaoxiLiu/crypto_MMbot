"""Read-only module boundaries; model code never receives the mutable OMS."""
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol
from app.market_making.fusion.signals import MarketState


@dataclass(frozen=True)
class OrderView:
    cid: str
    side: int
    price: Decimal
    remaining: Decimal
    phase: str
    reduce_only: bool


@dataclass(frozen=True)
class QueueView:
    cid: str
    valid: bool
    front: float | None
    as_of_ns: int
    reason: str


@dataclass(frozen=True)
class DecisionContext:
    ns: int
    market: MarketState
    position: Decimal
    working: tuple[OrderView, ...]
    queues: tuple[QueueView, ...] = ()


@dataclass(frozen=True)
class PriceForecast:
    alpha_bp: float
    as_of_ns: int
    valid_until_ns: int
    model_id: str


@dataclass(frozen=True)
class SideAssessment:
    side: int
    allow_opening: bool
    as_of_ns: int
    valid_until_ns: int
    reason: str


class PriceModel(Protocol):
    def predict(self, context: DecisionContext) -> PriceForecast | None: ...


class OrderEvaluator(Protocol):
    def evaluate(self, context: DecisionContext,
                 candidates: tuple) -> tuple[SideAssessment, ...] | None: ...
