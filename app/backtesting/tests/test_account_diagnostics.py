import json
from pathlib import Path
import unittest
from app.market_making.core import D






class ContinuityTests(unittest.TestCase):

    def test_ledger_restore_retains_cost_and_stop_state(self):
        from dataclasses import asdict
        from app.market_making.core import Ledger
        ledger=Ledger();ledger.fill(-1,D('4'),D('100'))
        ledger.day_key='2026-09-18';ledger.day_start_equity=D('601');ledger.day_stopped=True
        restored=Ledger.restore(json.loads(json.dumps(asdict(ledger),default=str)))
        self.assertEqual(restored,ledger)
        self.assertEqual(restored.equity(D('101')),D('596'))

