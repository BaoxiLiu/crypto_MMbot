"""Live/shadow adapter prototype using project-owned shared math. Imports never start network or trading."""
from app.market_making.fusion.config import FusionConfig as StrategyConfig, Factors as ProtectionSwitches
from app.market_making.core import Rules
from .engine import StrategyEngine
__all__=['StrategyConfig','ProtectionSwitches','Rules','StrategyEngine']
