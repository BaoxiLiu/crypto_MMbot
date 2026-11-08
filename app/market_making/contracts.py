"""Quote intent and public order phase; no OMS, strategy or I/O dependencies."""
from dataclasses import dataclass
from decimal import Decimal as D
from enum import Enum


class Phase(str,Enum):
    PENDING_NEW='PENDING_NEW'
    LIVE='LIVE'
    PARTIAL='PARTIALLY_FILLED'
    PENDING_CANCEL='PENDING_CANCEL'
    UNKNOWN='UNKNOWN'
    FILLED='FILLED'
    CANCELED='CANCELED'
    EXPIRED='EXPIRED'
    REJECTED='REJECTED'


@dataclass(frozen=True)
class Desired:
    side: int
    price: D
    qty: D
    reduce_only: bool=False
    reason: str='as'

