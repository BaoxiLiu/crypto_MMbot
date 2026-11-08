"""Independent Bellman checks for the paper's finite inventory-grid recursion."""
import numpy as np

from app.market_making.strategies.fb_hjb_math import solve_hjb, fit_objective


def scalar_reference(grid, actions, rates, costs, objective, steps, dt, discount, risk_scale):
    previous = np.zeros(len(grid))
    choices = np.zeros(len(grid), dtype=int)
    for _ in range(steps):
        current = np.zeros(len(grid))
        for j, q in enumerate(grid):
            scores = []
            for bid, ask in actions:
                pb = 0 if bid < 0 or j == len(grid)-1 else -np.expm1(-rates[0]*np.exp(-rates[2]*bid)*dt)
                pa = 0 if ask < 0 or j == 0 else -np.expm1(-rates[1]*np.exp(-rates[3]*ask)*dt)
                score = 0.
                for buy, sell in ((0,0), (1,0), (0,1), (1,1)):
                    probability = (pb if buy else 1-pb)*(pa if sell else 1-pa)
                    if probability == 0:
                        continue
                    next_j = j+buy-sell
                    next_q = grid[next_j]
                    features = np.array([buy*max(0,bid)+sell*max(0,ask),
                                         next_q, next_q**2*risk_scale,
                                         buy*costs[0]+sell*costs[1]])
                    score += probability*(features@objective+discount*previous[next_j])
                scores.append(score)
            choices[j] = np.argmax(scores)
            current[j] = scores[choices[j]]
        previous = current
    return choices, previous


def test_successor_values_equal_independent_multistep_scalar_bellman():
    rng = np.random.default_rng(20261106)
    grid = np.arange(-3., 4.)
    actions = np.array([(b,a) for b in (-1.,0.,1.,3.) for a in (-1.,0.,1.,3.)])
    for _ in range(10):
        rates = np.r_[rng.uniform(.01,5,2),rng.uniform(.01,2,2)]
        costs = rng.uniform(0,3,2)
        objective = np.array([1.,rng.normal(),-rng.uniform(.01,2),-rng.uniform(.01,2)])
        args = (grid,actions,rates,costs,objective,5,.5,.99,.7)
        choices, vector = solve_hjb(*args)
        expected_choices, expected_value = scalar_reference(*args)
        np.testing.assert_array_equal(choices, expected_choices)
        np.testing.assert_allclose(vector@objective, expected_value, atol=1e-10,rtol=1e-10)


def test_squared_inventory_retains_fill_variance():
    grid = np.arange(-1.,2.)
    actions = np.array([[0.,0.]])
    rates = np.array([1.,1.,1.,1.])
    _, values = solve_hjb(grid, actions, rates, np.zeros(2),
                           np.array([1.,0.,-1.,0.]),1,1.,1.,1.)
    probability = 1-np.exp(-1.)
    assert values[1,1] == 0.
    np.testing.assert_allclose(values[1,2],2*probability*(1-probability))
    assert values[1,2] > 0


def test_ridge_mixture_enforces_safe_objective_and_scaled_target():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(100,4))
    p = fit_objective(x,rng.normal(size=100),np.ones(100),
                      [1,0,-.01,-1],.01,.25,5*.7)
    assert p[0] == 1 and p[2] < 0 and p[3] <= 0
    assert abs(p[1]/(-2*p[2]*.7)) <= 5+1e-12


def test_target_projection_survives_volatility_drop_between_fit_updates():
    from dataclasses import replace
    from decimal import Decimal as D
    from app.market_making.as_math import ASConfig
    from app.market_making.core import Rules
    from app.market_making.fusion.config import Factors, FusionConfig
    from app.market_making.strategies.hjb_strategy import DEFAULTS
    from app.market_making.strategies.hjb_strategy import HJBStrategy

    settings = dict(DEFAULTS, trade_start_ns=1, trade_end_ns=100_000_000_000, capital_utilization=None)
    rules = Rules(D('.01'), D('.01'), D('5'), D('.01'), D('100000'), D('.5'), D('1.5'))
    config = replace(FusionConfig(), as_config=replace(
        ASConfig(), position_cap=D('2'), maker_fee=D(0),
        half_spread_floor=0, warmup_samples=2))
    strategy = HJBStrategy(rules, config, Factors(), settings)
    for i in range(30):
        strategy.book(1+i*100_000_000, D('100'), D('100.02'))
    # Emulate a fitted objective retained from a higher-volatility interval.
    strategy.objective = np.array([1., 2., -1e-8, -1.])
    strategy.next_fit = 100_000_000_000
    strategy.desired(D(0), 3_000_000_001)
    target = strategy.objective[1]/(-2*strategy.objective[2]*strategy.current_risk_scale)
    assert abs(target) <= settings['max_target_units']+1e-10
