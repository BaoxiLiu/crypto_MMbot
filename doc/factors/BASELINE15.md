# 九项盘口特征与十五项训练基线

实现入口为tools/factors/compute_v3_baseline15.py；本页说明列名，不宣称这些特征已改善做市收益。b/a是已收到的最优买卖价，B/A是对应数量，m=(b+a)/2。所有历史窗口只使用决策时可见数据，缺失、重置与完整性要求以实现为准。

| 九项基础列 | 定义 |
|---|---|
| spread_bp | (a−b)/m×10000 |
| l1_imbalance | (B−A)/(B+A) |
| microprice_offset_bp | ((bA+aB)/(B+A)−m)/m×10000 |
| ofi | 过去1秒顶档价量事件OFI之和 |
| buy_qty / sell_qty | 过去1秒主动买/卖成交数量，两列 |
| signed_notional | 过去1秒主动买成交额减主动卖成交额 |
| mid_return | m(t)/m(t−1秒)−1，比例值，不乘10000 |
| book_age_ms | 决策时刻减最近深度接收时刻，毫秒 |

单次OFI为I(b新≥b旧)B新−I(b新≤b旧)B旧−I(a新≤a旧)A新+I(a新≥a旧)A旧；它不同于主动买卖成交量差。

十五项基线另含flow_imbalance_1s、btc_mid_ret_1s/5s、btc_ofi_1s/5s、btc_flow_imb_1s六列。主动流比例为(buy_qty−sell_qty)/(buy_qty+sell_qty)，有效窗口无成交时为0；数据缺失不能等同于无成交。BTC列使用同一因果时钟对齐。

当前v3工具在1秒网格重算：盘口取网格时刻之前最后可见事件，流量窗口为(t−1000ms,t]。旧缓存的固定5ms延迟不作为当前共同执行配置；标签由当前样本组装器生成，不能把未来标签当作输入。
