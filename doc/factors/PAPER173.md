# 论文173项的计算入口

2026-11-01：已解除历史报告CSV依赖，名称/顺序固化在`app/factor_library/paper173_schema.json`，数学计算继续使用原NumPy参考实现。

```python
from app.factor_library.paper173 import FEATURE_NAMES, Observations, compute_named

# obs包含已经接收的成交与BBO历史，时间为LOCAL整数纳秒。
# bid/ask价格和数量阶梯必须是decision_ns时已可见的完整盘口。
values = compute_named(obs, decision_ns, side, bid_prices, bid_qty, ask_prices, ask_qty)
# values是按冻结顺序排列的173项字典；side=+1买单，-1卖单。
```

## 输入与边界

- `Observations`字段为trade_ns、trade_price、trade_qty、trade_buy、book_ns、book_bid、book_ask、epoch_ns。完整盘口另外传入，必须与查询时BBO匹配；适配层还需证明完整深度的as-of身份。
- 11类窗口统计×5种尺度（100ms/1s/5s/30s/300s）×3段历史=165项，另有8项盘口寿命、流动性和长波动统计。
- 近/远侧按拟挂买卖方向统一；buy/sell数量是基础币/合约数量，不是USDC成交额。
- 最长统计预热约83分30秒；500个10秒收益需要501个价格端点。历史不足保留NaN，不能填0。跨数据缺口重新预热。
- 固定50万报价币冲击指标需要足够深度，小币不足时是NaN；这不是程序失效。若改成其他金额必须另建定义版本。
- 名称顺序来自本地保存的论文附录，不使用论文回归系数作为我们的模型参数；数值约定见`doc/factors/PAPER173_CONTRACT.md`。

## 验证与使用范围

已有窗口端点、未来信息、方向、VWAP、深度、BBO寿命、长波动、缺口与陈旧输入测试；新增独立schema顺序校验和充分预热合成数据上的全173项输出/未来不变性测试。

```bash
python -m tools.factors.build_factor_catalog --check
python -m pytest tools/tests/test_paper173_computation.py tools/tests/test_paper173_library.py -q
python tools/build_factor_catalog.py --check
```

该入口可用于正确性验证与批量研究，不保证100微秒实时计算预算。173项可以单独做模型输入，也可与202项微观结构或18项分钟特征组合；选择顺序与模型训练清单绑定，不能静默重排。

目前尚未完成13币真实样本覆盖、统一事件流适配及策略增量回测；计算可用不等于有预测效果或能盈利。
