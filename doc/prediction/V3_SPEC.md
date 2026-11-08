# Kimi v3 预测线：因子与目标定义表（P0）

状态：v1.3，2026-10-11。分支 `prediction/kimi-v3`。v1.1 变更（P0.5 探针结论，见 data/prediction-v3/probe-p0.5/probe-report.md）：B 组撤单定义简化（量下降即撤单）；B6 基准改名义额 p90；新增 P1 管线校验门槛；补数据量级与对账锚点。v1.2 变更（P1.5 工程增补，**不改变任何因子/目标/门槛的语义**，见 §7）：认证段内跨小时窗口链式、E 组跨币 lockstep 接线、B6/C6 训练段冻结阈值表、段首行剔除规则声明。v1.3 变更（目标体系扩充）：按 doc/models.md 第四部分（五类目标体系，已冻结）新增第三类量加权路径收益与第四类时间加权路径收益（T∈{5s,30s}），T1–T5 保留原语义，第一类/第二类作为对照。本表冻结前任何实现不得开始；冻结后修改须注明版本。

## 0. 总则

1. **因果性**：所有因子与标签只用决策时刻 t 之前（含 t）的信息；任何"用未来数据"的实现必须被测试拦截。
2. **增量计算**：每个因子 O(1) 流式更新，维护有界状态；禁止全窗口重算。滚动回归等用充分统计量（Σxy/Σx² 队列）实现。
3. **行网格**：1 秒 UTC 网格；仅认证段内；跨缺口不插值；任何所需窗口/标签在段内不完整则该行剔除（与冻结链同口径）。
4. **归一化**：按币做稳健缩放（减训练窗中位数、除 IQR），统计量只从训练段估计、冻结后用于验证/测试。
5. **基线**：冻结 15 因子（旧 9 列 + v2 六列）原样重算，定义以 `app/backtesting/HISTORY.md` 与 `feature_store` 口径为准；本文件中不重复抄录，只列新增。
6. **比较公平性**：与冻结 ridge 的比较在原始模型层（未校准/未收缩评分）进行。

## 1. 预测目标

| ID | 定义 | 标签公式（mid 为 1s 网格中间价，bp=1e-4） | 优先级 |
|---|---|---|---|
| T1 | 未来 1s 均价 mid 收益 | (TWAP_mid[t+1s] − mid[t]) / mid[t] ×1e4，TWAP=未来 1s 网格均值；另出延迟对齐口径：窗口 [t+0.5s, t+1.5s] | 核心 |
| T2 | 未来 30s 已实现波动率 | std(Δlog mid, (t, t+30s]) ×1e4，30 个 1s 收益 | 核心 |
| T3 | 未来 5s 均价 mid 收益 | 同 T1 口径，5s 窗口 | 核心 |
| T6 | 第三类·量加权路径收益 R_vwap(t,T)，T∈{5s,30s} | P_vwap=Σ mid(t+k)·vol(t+k)/Σ vol(t+k)，vol=buy_n_1s+sell_n_1s；R_vwap=P_vwap/mid(t)−1（×1e4）；整窗零成交→NaN 剔除 | 核心-主攻候选 |
| T7 | 第三类·侧细分 R_askw / R_bidw(t,T)，T∈{5s,30s} | P_askw=Σ buy_vwap·buy_qty/Σ buy_qty（未来主动买盘平均成本）；P_bidw=Σ sell_vwap·sell_qty/Σ sell_qty（未来主动卖盘平均回收价）；对应侧整窗零成交→NaN；做市商被逆向选择后的真实退出基准 | 核心-主攻候选 |
| T8 | 第四类·时间加权路径收益 R_twap(t,T)，T∈{5s,30s} | P_twap=(1/T)Σ mid(t+k)；R_twap=P_twap/mid(t)−1（×1e4）；第三类的对照，用于分离"时间效应"与"成交量效应" | 对照 |
| T4 | 条件成交毒性（成交后 1s Markout） | 仅探索，T1–T3 完成后再议；已有阴性证据（vol-markout 诊断 2026-10-09） | 探索 |
| T5 | 未来 30s IOC 减仓冲击成本 | (VWAP_sell(t, t+30s] − mid[t]) / mid[t] ×1e4，VWAP 只含卖方发起成交；标签为 bp 成本，越大越贵 | 探索 |

所有标签窗口不完整（段尾/跨缺口）行剔除；第三类对应侧整窗零成交同样剔除（禁止补零）。T6–T8 定义出处：doc/models.md 第四部分第三类/第四类；与第一类差别：不只取端点，而是"沿途真实的量发生在什么价位"，无量抖动自动降权。T4/T5 实现与否不影响 T1–T3、T6–T8 的评估及格线。

## 2. 数据方案

| 期 | 区间 | 用途 | 切分 |
|---|---|---|---|
| P2a | 2026-09-19 ~ 2026-10-02（13 天认证窗） | 与冻结链同口径对账、管线验证 | 与冻结链相同 |
| P2b | 2026-08-05 ~ 2026-10-05（61 天，13 币） | 正式评估 | 训练 08-05~09-08 / 验证 09-08~09-18 / 测试 09-18~10-05（时间顺序；标签horizon≤30s，边界泄漏可忽略，不设 purge 间隔） |

币种：13 个全集（BTC SOL SUI UNI DOGE 1000SHIB WIF ETH 1000PEPE LTC CRV XRP 1000BONK，USDC 永续）。
来源：CryptoHFTData orderbook/trades/mark_price Parquet（认证清单内）；自采数据留二期对齐验证。

## 3. 新增因子定义

记法：t = 决策时刻；mid/best bid/ask 为盘口状态；qty 以基础币计；名义额 = qty×price。所有"每秒速率"类因子 = 窗口内总量 / 窗口实际秒数（缺秒不补）。空值规则统一：状态未就绪或窗口覆盖不足 → NaN（行剔除），禁止补零。

### A 组：深度形态（8）

| ID | 名称 | 定义 | 窗口 |
|---|---|---|---|
| A1 | depth_imb_3t | (BidQty@≤3tick − AskQty@≤3tick)/(sum) | 即时 |
| A2 | depth_imb_10t | 同上 @≤10tick | 即时 |
| A3 | depth_ratio_5bp | (±5bp 内买深度 − 卖深度)/(sum) | 即时 |
| A4 | depth_ratio_10bp | 同上 @±10bp | 即时 |
| A5 | depth_slope_imb | slope_s = (Qty@10t − Qty@3t)/7；因子 = (slope_bid − slope_ask)/(\|slope_bid\|+\|slope_ask\|+ε) | 即时 |
| A6 | depth_imb_delta | A1(t) − A1(t−5s) | 5s |
| A7 | book_update_rate | depth 事件数/秒 | 10s |
| A8 | spread_dev | spread_bp(t) − rolling_median_1h(spread_bp) | 1h |

### B 组：撤单流（6，v1.1 修订）

P0.5 探针结论（BTCUSDC 2026-09-19 00 时，归因窗 W=200ms）：成交归因覆盖 92.9% 成交总量（无数据缺口），但成交消耗仅占毛量下降 0.42%（撤单流约为成交的 220 倍，散布 8,000+ 档）。**v1.1 起主定义改为"量下降即撤单"**，先扣成交的版本降级为敏感性分析列（若启用须冻结 W=200ms 与方向映射）。

撤单检测：某价位 quantity 下降（或 0=删档），即记该价位该事件撤量 = |Δqty|。orderbook 为事件式增量（每小时 1 组全深度 snapshot + diff，quantity 为绝对量）。

| ID | 名称 | 定义 | 窗口 |
|---|---|---|---|
| B1/B2 | cancel_intensity_bid/ask | 买/卖侧撤单量（名义额）/秒 | 30s |
| B3 | cancel_add_ratio | 撤单名义额 / 新增名义额（量上升求和） | 30s |
| B4 | near_touch_cancel_share | ≤1 tick 内撤单名义额 / 总撤单名义额；触价距离取 **1s 网格状态**（与总则行网格一致） | 60s |
| B5 | cancel_imbalance | (B1 − B2)/(B1 + B2) | 30s |
| B6 | large_cancel_intensity | 单笔撤量名义额 > 滚动 p90（**名义额口径**，p90 从训练段估计冻结；另设最小撤量地板 = 该币 0.1×中位单笔撤量，防微幅抖动左偏基准）的撤单名义额/秒 | 60s |

**P1 管线校验门槛（随实现冻结）**：每小时文件成交捕获率（归因到 book 的成交名义额 / trades 总名义额）≥ 90%，且 trades 的 trade_id 在文件内连续；不满足则该小时文件拒绝进入特征构建并报错，不得静默跳过。

### C 组：主动成交与毒性（10）

| ID | 名称 | 定义 | 窗口 |
|---|---|---|---|
| C1–C3 | taker_imb_1s/5s/30s | (主动买名义额 − 主动卖名义额)/(sum) | 1s/5s/30s |
| C4 | trade_intensity | 成交笔数/秒 | 10s |
| C5 | trade_size_ratio | 单笔均量 / rolling_median_1h(单笔均量) | 即时+1h |
| C6 | large_trade_share | 大单（>滚动 p90）名义额 / 总名义额（p90 训练段冻结） | 30s |
| C7 | taker_persistence | 近窗内 taker_imb_1s>0 的秒数占比 | 30s |
| C8 | taker_accel | taker_imb_5s(t) − taker_imb_5s(t−5s) | 5s |
| C9 | flow_toxicity_vpin | Σ\|买−卖\|名义额 / Σ(买+卖) 名义额（VPIN 时间分桶近似） | 30s |
| C10 | kyle_lambda_60s | 滚动回归 slope：Δmid ~ signed_taker_notional（充分统计量 O(1) 更新） | 60s |

### D 组：时序与波动率（15）

| ID | 名称 | 定义 | 窗口 |
|---|---|---|---|
| D1–D3 | ret_5s/30s/60s | (mid(t)/mid(t−Δ)−1)×1e4 | 5s/30s/60s |
| D4/D5 | ofi_5s/30s | 触价档量变化构造的 OFI（冻结口径同族），Δ 窗口 | 5s/30s |
| D6–D8 | rv_30s/60s/300s | std(Δlog mid 1s 收益)×1e4 | 30s/60s/300s |
| D9 | rv_ratio | rv_30s / (rv_300s+ε) | — |
| D10/D11 | up_rv_60s / down_rv_60s | 正/负 1s 收益的平方和开根（有符号半方差） | 60s |
| D12 | semivar_ratio | (up_rv − down_rv)/(up_rv + down_rv + ε) | 60s |
| D13 | microprice_mom_10s | (microprice − mid)/spread 的 10s 变化 | 10s |
| D14 | range_position_60s | (mid − min_60s)/(max_60s − min_60s + ε) | 60s |
| D15 | jump_count_60s | \|1s 收益\| > 4×滚动中位(\|1s 收益\|) 的次数 | 60s |

### E 组：环境与跨币（11）

| ID | 名称 | 定义 | 窗口 |
|---|---|---|---|
| E1/E2 | btc_ret_5s/30s | BTC mid 收益（冻结链已含部分 BTC 特征，此处按本文件统一口径重算） | 5s/30s |
| E3 | btc_rv_60s | BTC rv_60s | 60s |
| E4 | btc_taker_imb_30s | BTC taker_imb_30s | 30s |
| E5–E7 | mkt_ret_5s/30s, mkt_rv_60s | 13 币等权 mid 收益 / 其 rv | 5s/30s/60s |
| E8 | funding_level | 当前资金费率（mark_price 流最近值） | 即时 |
| E9 | time_to_funding_min | 距下一结算点分钟数 | 即时 |
| E10/E11 | hour_sin/hour_cos | UTC 小时周期编码 | 即时 |

## 4. 评估及格线（P4 前冻结）

新因子 + LightGBM 相对"冻结 15 因子 + ridge"（同切分重训对照）：
1. 测试窗 rankIC 提升 ≥ 0.01，且块 bootstrap（≥1h 块）95% 区间不含零；
2. 七币以上方向一致；
3. T2（波动率目标）rankIC ≥ 0.15（管线自检线）；
4. 延迟对齐口径下 T1 增量不消失（消失则判不可用）；
5. 组级消融报告全部结果（含阴性），禁止事后单因子挑故事。

模型纪律：LightGBM 按币独立训练；超参在验证窗早停与选择，测试窗只碰一次；调参预算固定（网格 ≤ 27 组），不逐币定制。

## 5. 工程身份

- 新代码独立模块，不 import 进冻结 `feature_store`/`prediction_cache` 路径；冻结产物零接触。
- 特征缓存 NPZ 留 SHA256 登记指纹；产出目录 `data/prediction-v3/`。
- 本表冻结后方可进入 P1；因子/目标/及格线的任何变更记 v1.1+ 并注明理由。

## 6. 数据量级与对账锚点（P0.5 探针实测）

- 量级：13 币 × 13 天 ≈ 12.3 亿 orderbook diff 行 + 4,620 万 trades；×61 天 ≈ 57.7 亿 + 2.17 亿。BTC+ETH 约占 54%。
- 耗时估计（宿主机 python3.11，单进程向量化）：13 天单遍 2–6 小时，61 天 10–30 小时；多进程可再压 1/3–1/10。构建按"夜间批量 + 逐小时断点续跑"设计。
- 因子对账锚点（BTCUSDC 2026-09-19 00:59:56.859 UTC 等三个时刻，存 `data/prediction-v3/probe-p0.5/probe_results.json`，P1 单测必须复现）：A1=0.544228 / C1=−0.305241 / D6=0.746453（T=56.859s；另两组时刻见 JSON）。
- 已知数据事项：`order_count` 列全 NaN 不使用；trades 中 order_type="NA" 占 0.7%（P1 顺手确认含义，默认按普通成交处理需记录）；trades 与 book 同钟（trade_time−event_time ∈ [−22,0]ms）。

## 7. P1.5 工程增补（v1.2，2026-10-10）

1. **跨小时窗口链式（认证段内）**：滚动窗口因子（B/C 组 30s/60s、A8/C5 1h 中位数、D4/D5 OFI 前缀）在认证段内跨小时携带窗口尾部状态（每币每时 `<HH>.state.pkl`，机器 `export_state/import_state`）；**盘口仍按小时 snapshot 重建**，跨小时的只有因子窗口状态，无 book 状态携带。前一小时状态缺失（段首、日界、断档）时该小时按段首语义启动（窗口不足 → NaN）并在日志注明。mid 滞后类因子（D1–D3、D6–D15）因网格秒 0 的 mid 来自本小时 snapshot，本就不需要跨小时；C1 的 1s 窗口永不跨小时。
2. **E 组跨币接线**：`build_hour_multi` 按秒 lockstep 构建多币；E5–E7 的市场 mid = 当秒可用币种 mid 的等权均值（某币该秒无有效 mid 则剔除该币；全缺 → NaN）；BTC 的 1s 网格（mid、买/卖名义额）自馈并外馈给非 BTC 币（E1–E4）。单币构建（`build_hour`）不含市场聚合，E5–E7 = NaN。
3. **B6/C6 阈值冻结表**：`app/prediction/thresholds.json`，由 `tools/compute_v3_thresholds.py` 从训练段（2026-08-05~09-08）采样估计（当前 35 小时/币，按日轮转取小时覆盖不同时段）；构建时优先加载冻结表（meta 记 `source: frozen`），无该币条目时退回小时内 provisional 估计并记录。P2 正式评估前可用更密采样重算（工具支持 `--hours-per-day`），采样估计不改因子语义。
4. **段首/边界行剔除规则（保持现状，声明于此）**：builder 只产出 NaN 语义（窗口覆盖不足 → NaN，永不补零），**不做行剔除**；含段首窗口不足、以及零成交秒导致的 C1/C5 NaN 在内的所有 NaN 行，统一在 P2 数据集装配时按行剔除。`complete` 列（全因子非 NaN 标记，外部 feed 依赖除外）仅为装配提示。
5. **断点续跑**：已存在且 meta 状态 ok 的 `<HH>.parquet` 跳过；窗口状态文件缺失不影响 parquet 产物本身，只影响下一小时的窗口覆盖。
