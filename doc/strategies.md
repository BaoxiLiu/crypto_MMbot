# 策略逻辑与配置

本文件是该主题的统一维护入口。项目总览见 [README](../README.md)。


## 做市策略统一说明与参数目录

更新：2026-11-11。本文是策略逻辑、出处、参数和实现入口的统一说明。策略插件规范见[架构文档](development.md)；共同回测规范见[doc/backtesting.md](backtesting.md)。

### 1. 分类与命名

策略注册表接受四类：**NativeAS及早期改进AS、GLFT、NH预测报价、FB-AS HJB**。历史插件名称与HJB旧版本均已退役；无插件配置、非零mask及旧预测叠加配置均已拒绝。

同一数学逻辑的开关、阈值、预算比例作为参数；不因取值不同复制策略实现。`id`只是某次实验的配置名称，`type + version + params`才描述具体算法。实现合同的版本号与历史实验run_id中的v1/v2/v3分开。

| 家族/用途 | 规范type | 主要参数差异 | 来源 |
|---|---|---|---|
| 主线AS及早期改进 | `paper_native_as` v1 | `mode`、`min_half_spread_ticks` | AS理论、参考Hummingbot约定的本地实现、本项目早期paper工程 |
| GLFT | `glft` v1 | `gamma`、`xi`、`inventory_units`、`inventory_solution`、`reference_price`、`markout_feedback` | GLFT论文、HftBacktest示例的独立适配 |
| 主线NH | `nh_quote_value` v1 | 各币冻结的`models` | 用户提供的NH压缩包，移植预测和报价选择逻辑 |
| 主线HJB | `fb_as_hjb` v2 | `adaptive`、`capital_utilization`、求解/学习参数 | Zero-shot论文附录C，本地市场估计与预算扩展 |



主线代码结构：

```text
app/market_making/strategies/
  as_strategy.py       # ASStrategy：原始/改进AS，参数切换
  glft_strategy.py     # GLFTStrategy：闭式、谱解及报价覆盖参数
  nh_strategy.py       # NHStrategy：预测与报价选择
  hjb_strategy.py      # HJBStrategy：固定/自适应/预算，同一类
  base.py              # 共用观测和生命周期，不含报价算法
  registry.py          # 注册、校验、能力和按币绑定
  fb_hjb_math.py       # HJB数学辅助
  nh_quote_value/      # NH推理特征与报价评分辅助
```

主线直接导入 `as_strategy.py`、`nh_strategy.py`、`hjb_strategy.py`；已删除旧 AS、NH 和 HJB 的五个转发模块。HJB只接受v2参数，运行时只有HJBStrategy及v2身份；数学函数仍在 `fb_hjb_math.py`。

### 2. NativeAS与早期改进AS

**出处。** 理论源头为Marco Avellaneda与Sasha Stoikov的[High-frequency trading in a limit order book（2008）](https://math.nyu.edu/inmemoriam/avellaneda/HighFrequencyTrading.pdf)。本项目主线使用参考Hummingbot无限时域约定的本地Python AS计算器，不依赖Hummingbot编译模块或容器，入口是[as_quote.py](../app/market_making/as_quote.py)，并通过[as_strategy.py](../app/market_making/strategies/as_strategy.py)接入共同回测执行层。静态下限、S1、SC0、microprice与keeper来自本项目早期paper交易工程，不是原论文提供的模块。

**输入与输出。** 输入可见盘口、主动成交、估计的波动率/到达强度、当前仓位和有效历史反馈；输出买卖报价、数量及撤改单意图。AS按库存风险调整报价中心，并综合风险和流动性确定宽度。订单排队、成交、资金费、账户和硬风控交由共同OMS处理。

| 配置称呼 | `mode` | `min_half_spread_ticks` | 其他政策 |
|---|---|---|---|
| 原始NativeAS | `as_base` | 不设置 | 关闭microprice/keeper、策略库存覆盖、静态下限、S1和SC0；保留共同风险约束 |
| 早期完整改进AS | `paper_s1_sc0` | 不设置，原政策静态4 tick | 保留microprice、keeper、S1波动率下限和SC0反馈 |
| 改进AS：单侧2 tick下限 | `paper_s1_sc0` | `2` | 其余同完整改进版 |
| 改进AS：取消静态下限 | `paper_s1_sc0` | `0` | S1、SC0继续启用 |

同一个`ASStrategy`还支持独立布尔参数`volatility_adjustment`和`markout_adjustment`，分别控制S1与SC0。不设置时按mode恢复原行为：原始AS关闭、改进AS启用。`mode="as_base"`拒绝把这两个模块设为true；需要叠加模块时选择改进模式。microprice/keeper继续由mode预设控制。

“2 tick”是**单侧静态最低半价差**，不是2 bp；设0也不意味着实际报价宽度为0。S1、SC0或原生AS给出更大宽度时仍按政策生效。现有政策下限为`min(24, max(静态下限, S1下限, SC0目标))`，再约束原生AS报价；24 tick并不是对原生AS半价差的总封顶。

S1使用30分钟的1秒收益总体标准差，历史配置`k=2`，波动下限范围3—24 tick。SC0依据最近50次有效成交后1秒Markout的bp中位数，至少10个新样本、间隔5分钟，死区±0.25 bp；不利时目标乘1.5，有利时除1.25，单次最多变化4 tick，目标范围3—24 tick。当前政策由[s1_config.py](../app/market_making/s1_config.py)、[vol_floor.py](../app/market_making/vol_floor.py)、[spread_control.py](../app/market_making/spread_control.py)和[core.py](../app/market_making/core.py)实现。

统一profile为[paper_native_as_v1.json](../config/strategies/paper_native_as_v1.json)。其中`paper_config`保留既有政策初始化种子；适配器注入实验的币种、初始资金、单量、共同仓位限制与Maker费，实际执行延迟和费用仍由实验控制。它不是另一个执行账户。

**实现差异。** 历史归档没有独立bookTicker，回放使用因果L2 BBO及数量替代。不能宣称与早期paper逐事件等价；改变静态下限也不能单独证明波动率或Markout模块有效。

### 3. NH预测报价

**出处。** 用户提供的`Cryptocurrency_quant-marketmake_nh.zip`；源码包SHA-256为`4a97e3e3fa3bdcdbc1ef4a15be4bd8b7a7b426c693dcf4e859428012beab8270`，由[移植模块](../app/market_making/strategies/nh_quote_value/__init__.py)记录。该工程策略不是AS原论文的原始策略。本地移植未使用其原回测引擎或归档权重，而是以共同数据/队列重新构造标签及训练初始模型。

**逻辑与输入输出。** 100ms因果采样，23维盘口/成交流特征进入alpha模型，输出短期方向预测；28维候选报价特征分别预测成交概率和成交条件下的Markout边际。对候选报价按`成交概率 × 名义金额 × 预期成交比例 × 预测边际bp / 10000`评分，选择正值报价；无支持样本、减仓等情形保留适配器规定的回退路径。该分数不是已经扣齐全部费用和库存损益的账户净盈利预测。

alpha裁剪为±5 bp后与库存偏移共同调整中心；基础半价差取3 bp、2 tick、波动项的最大值。候选宽度包括1/2/3/4/6 tick和原始宽度；有流量过滤、2 tick改单保持阈值与5秒订单年龄处理。当前这些是实现中的常量，不伪装成已经开放的配置项。

**参数与文件。** `params.models[币种]`必须提供`alpha`、`value`、`trade_start_ns`和`trade_end_ns`。模型含真实均值、缩放、回归/分类权重、支持范围等；标签结束时间必须早于部署时刻。当前适配器冻结初始权重，没有自动吸收全因子库或评价期滚动重训。见[nh_strategy.py](../app/market_making/strategies/nh_strategy.py)、[features.py](../app/market_making/strategies/nh_quote_value/features.py)、[value_features.py](../app/market_making/strategies/nh_quote_value/value_features.py)、[training.py](../app/market_making/strategies/nh_quote_value/training.py)和[quote_policy.py](../app/market_making/strategies/nh_quote_value/quote_policy.py)。

### 4. FB-AS HJB家族

**出处。** Arip Asadulaev，[Zero-shot adaptation to order book dynamics（2026）](https://arxiv.org/abs/2605.21707)，本地实现主要依据附录C。论文将市场动力学与交易目标分离，通过HJB未来特征与目标向量的内积选报价。本项目的1秒反馈、市场估计、网格、窗口等是明确的本地实现假设，不能称为作者逐笔等价复现。具体参数与实现边界以下文及当前HJB策略代码为准。

**逻辑。** 用已可见主动成交穿透距离拟合两侧到达强度及距离衰减，用成熟反馈估计逆向选择成本；库存网格上的HJB计算四维未来特征：价差、库存、按方差缩放的库存平方、不利价格变化代理。目标向量对这些特征加权，选择买卖报价或停报某侧。自适应版只在1秒反馈成熟后更新残差Ridge回归并与先验混合；固定版不更新目标权重，但仍更新市场参数。

**统一入口。** 新配置使用`type="fb_as_hjb", version=2`，profile为[fb_as_hjb_v2.json](../config/strategies/fb_as_hjb_v2.json)。

| 研究配置 | `adaptive` | `capital_utilization` |
|---|---:|---|
| 固定权重HJB | `false` | `null` |
| 自适应HJB | `true` | `null` |
| 自适应HJB＋50%权益库存预算 | `true` | `"0.5"` |

`null`表示不添加策略侧权益预算，**仍有共同仓位/名义本金/账户约束**；数值必须严格在0与1之间。固定权重配合预算也是合法参数组合，不能因可配置就说已完成该组合回测。

| 参数 | 当前profile值 | 含义 |
|---|---|---|
| `step_seconds` / `horizon_steps` | 1 / 60 | 每秒求解、60步规划 |
| `market_window_seconds` / `reward_window_seconds` | 1800 / 1800 | 市场估计/成熟反馈窗口 |
| `half_life_seconds` | 300 | 回归样本权重半衰期 |
| `ridge` / `mixing` | 0.01 / 0.25 | 正则强度、与先验混合比例 |
| `minimum_labels` / `fit_interval_seconds` | 30 / 10 | 更新所需成熟样本及拟合间隔 |
| `prior` | `[1,0,-0.001,-1]` | 价差、库存方向、库存平方、不利变化权重 |
| `distances_ticks` | `[0,1,2,4,8,16,32]` | 离散报价动作；实现另有不报该侧动作 |
| `static_floor_ticks` | 0 | HJB动作静态距离下限，非改进AS的同名政策 |
| `max_target_units` | 10 | 隐含目标库存投影范围，**不是实际仓位硬上限** |
| `maximum_feedback_lag_ms` | 250 | 成熟反馈取价迟到上限 |
| `capital_utilization` | `null` | 可选当前权益库存预算比例 |

预算开启后按`min(共同数量上限, 共同名义额上限/mark, 当前权益×比例/mark)`规划，按lot向下取整；超预算时仅maker减仓，不跨过零。开仓意图另按mark与报价中更高价格检查额度。50%权益预算与撤量前后各50%的队列假设是两件事。在途成交和跳价仍可能穿越软预算，共同OMS压力停机保持生效；没有额外增加taker强平或固定3 bp止盈止损。

代码入口：[hjb_strategy.py](../app/market_making/strategies/hjb_strategy.py)中的单一`HJBStrategy`类，同时实现固定/自适应及可选预算；[fb_hjb_math.py](../app/market_making/strategies/fb_hjb_math.py)仅负责Bellman/回归数学。预算是同一策略的参数分支，不再使用独立预算策略子类或家族分发层。

**配置。** HJB仅接受`fb_as_hjb` v2；旧v1与`fb_hjb_capital_budget`已移除，读取旧类型会明确报错。项目配置已显式迁移，固定/自适应/预算均由同一类参数控制。已有报告不改写。旧研究报告已清理；当前实现不代表已获得持续做市正期望。

### 5. 测试与运行策略边界

旧轻量AS兼容实现及R/F/I保护层已删除。运行时仅使用三类主线插件；执行与协议测试使用固定报价夹具隔离撮合、账本和事件时序。该夹具没有AS定价公式，不是策略，也不能被运行时代码导入。各主线策略的数值行为通过插件与冻结事件轨迹单独验证。

### 6. 一份profile，多种参数配置

实验中的`profile`引用完整算法基线；`parameter_overrides`只修改这个profile的`params`，不会修改type、version、数据、费用或共同账户。字典递归合并，列表/标量整体替换，`null`保留为显式参数值；不支持删除键。未知或非法参数由策略校验器拒绝，完整展开值和profile哈希进入实验登记。内联`plugin`直接填写完整params，不再叠加`parameter_overrides`。

以下只是实验配置中的`strategies`片段，不是可以直接运行的完整实验：

```json
{
  "strategies": [
    {"id":"native_as","profile":"config/strategies/paper_native_as_v1.json",
     "parameter_overrides":{"mode":"as_base"}},
    {"id":"improved_as_2tick","profile":"config/strategies/paper_native_as_v1.json",
     "parameter_overrides":{"min_half_spread_ticks":2}},
    {"id":"improved_as_static0","profile":"config/strategies/paper_native_as_v1.json",
     "parameter_overrides":{"min_half_spread_ticks":0}},
    {"id":"hjb_fixed","profile":"config/strategies/fb_as_hjb_v2.json",
     "parameter_overrides":{"adaptive":false}},
    {"id":"hjb_adaptive","profile":"config/strategies/fb_as_hjb_v2.json"},
    {"id":"hjb_budget50","profile":"config/strategies/fb_as_hjb_v2.json",
     "parameter_overrides":{"capital_utilization":"0.5"}}
  ]
}
```

两个新profile的交易窗种子为UTC `[2026-08-01, 2026-08-11)`；改实验窗口时必须显式同步覆盖`trade_start_ns/trade_end_ns`。三日实验将`trade_end_ns`设为`1785801600000000000`（8月4日00:00 UTC），不能沿用十日期间的结束值而不说明。所有新回测必须包含BTCUSDC和ETHUSDC；quick/full仍执行13币及两Maker费率情景的共同规则。仅修改策略文档和参数组织，不触发历史回测重跑。

### 7. 后续维护规则

新增家族必须补充来源、输入/输出、核心逻辑、关键参数、单位、限制、实现入口和验证状态；已有家族的开关/阈值变化只新增参数配置，不复制实现、也不扩充“独立策略数”。真正更换定价公式、学习目标或模型输入合同，才考虑新类型或版本。目录里的代码可用、实验跑完、预测有效和稳定盈利是不同状态，必须分别记录。

### 8. 接入与验收

策略实现、注册和扩展测试合同见[开发文档](development.md)。本轮迁移前后八组原生行为轨迹完全一致，不更新黄金结果；软件验收与历史盈利验收分开。

可运行的[AS模块消融配置](../config/experiments/as_s1_ablation.json)对比原始AS与开启S1、关闭SC0反馈、静态单侧2 tick下限的改进AS。后者仍保留改进模式的microprice/keeper，不能解释为只改变S1的纯单变量消融。示例使用BTC/ETH、统一队列/账户和双maker费率；新增参数组合不新增策略家族，不修改Kernel、OMS和报告。

```sh
python -B -m app backtest validate-config --config config/experiments/as_s1_ablation.json
python -B -m app backtest run --config config/experiments/as_s1_ablation.json
```

完整运行请求UTC 8月1日至3日；命令增加`--max-segments 1 --max-files 1`只验证有限输入和输出流程，不能作为三日收益。当前已发布的局部示例参见[回测文档](backtesting.md)。

## 策略插件与实验配置

策略逻辑、出处与参数变体统一见 [doc/strategies.md](strategies.md)。本文只维护配置和接入协议。

统一入口由策略插件、数据/执行/风险profile、公共回测和报告组成。新实验使用config/experiments中的现行格式；不维护旧运行复现入口。

### 各部分的位置

| 部分 | 位置 | 负责什么 |
|---|---|---|
| 数据profile | `config/profiles/data/` | window、symbols、datasets、rules；不再携带策略参数路径 |
| 执行profile | `config/profiles/execution/` | engine、延迟、费用、队列与部分成交 |
| 风险profile | `config/profiles/risk/` | 共同账户资金、名义额上限、亏损停机设置 |
| 策略profile | `config/strategies/` | type、version、params；选择注册过的实现 |
| AS参数基线 | 实验的`strategy_parameters`映射 | 每个币显式、完整的AS/Fusion参数；与行情数据选择分开 |
| 公共执行 | fusion Kernel/OMS + hftbacktest | 延迟、下撤改单、排队、部分成交、记账与硬风控 |
| 结果 | final_only工作区、单文件HTML + src/metrics | 每币每策略独立账户，标准图表、指标和输入/代码身份 |

策略case可用`parameter_overrides`显式修改所引用profile的params：字典递归合并，列表/标量整体替换，null是显式值；不修改type/version或共同配置，不支持删除键。未知参数由策略校验器拒绝，原profile哈希和最终展开参数同时登记。只允许与profile一起使用；内联plugin直接填写params。

profile中的未知字段和跨层设置会拒绝；同字段同时内联和引用且值不同会拒绝。不会以加载顺序悄悄覆盖。profile路径相对仓库根，或显式绝对路径；不支持嵌套profile链。

### 注册策略类型

| type | version | 行为 |
|---|---:|---|
| `paper_native_as` | 1 | `mode`选择原始NativeAS/完整早期改进AS；静态tick下限为参数 |
| `nh_quote_value` | 1 | 各币冻结alpha及报价价值模型 |
| `fb_as_hjb` | 2 | v2以adaptive和capital_utilization统一固定/自适应/预算配置 |

`strategies`与`available_strategies()`只列出`paper_native_as`、`nh_quote_value`、`fb_as_hjb`。旧插件名称和HJB v1明确报错，没有隐藏插件列表。

旧`model_profile`自动展开已移除。预测模型由具体插件的参数引用真实manifest，输入清单、哈希与可用时间必须匹配；因子库、模型和策略分别维护版本。旧无插件预测和非零mask路径已删除；缺少显式插件时拒绝运行。

每币每策略创建独立实例/控制器/账户/输出目录，不共享可变状态。策略只返回报价意图，不能绕过OMS、修改成交账本或接触未来标签。PnL与Markout等统计仍使用已有统一报告与`src/metrics`口径。

### HJB统一参数入口与归因观测

新HJB配置使用`fb_as_hjb` v2和`config/strategies/fb_as_hjb_v2.json`。`adaptive=false/true`切换目标权重更新；`capital_utilization=null/"0.5"`关闭或启用50%当前权益预算。仅接受v2；旧v1和预算别名不再转换。固定/自适应/预算统一由`hjb_strategy.HJBStrategy`实现。案例的profile参数覆盖及三种配置示例见统一策略文档。

可选`observations.pnl_attribution=true`在标准全订单观测基础上记录成交回报可见mid优势、库存价格变化、缺口变动、末次mark基差与资金费/手续费，并闭合到账本净PnL。它不改变策略；新观测曾通过与旧v3结果的精确对账。首次风控停机和后续权益变动单列。最终HTML保留有界归因与来源哈希，临时逐笔文件按final_only清理。

### 三个主线代码入口

`as_strategy.ASStrategy`、`nh_strategy.NHStrategy`、`hjb_strategy.HJBStrategy`分别承载主线策略；`base.QuoteStrategy`只保存通用观测/生命周期。特征、数学、OMS和记账继续独立。旧compat目录和导入转发均已删除。prepare登记各插件实际源码及公共执行源码；不为旧插件名保留隐藏入口。

AS仍用mode选择原始/改进、min_half_spread_ticks选择静态下限，并新增可选volatility_adjustment/markout_adjustment独立覆盖。未设置覆盖时保持旧mode的行为；原始AS拒绝启用改进模块。HJB沿用v2的adaptive/capital_utilization。各case仍有独立实例、可变状态、账户及完整展开参数。


## GLFT

用户明确要求复现后新增`glft` v1家族；实现位于`app/market_making/strategies/glft_strategy.py`，profile为`config/strategies/glft_v1.json`。它复用AS生命周期与因果成交强度估计，独立替换定价器；没有另一套执行器。原生AS和既有家族的冻结轨迹保持不变。

参考[GLFT原论文](https://arxiv.org/abs/1105.3115)、[Guéant广义模型](https://arxiv.org/abs/1605.01862)及[HftBacktest GitHub示例](https://github.com/nkaz001/hftbacktest/blob/66b2e7537f0005b1bf691373017160566814ee4a/examples/GLFT%20Market%20Making%20Model%20and%20Grid%20Trading.ipynb)。独立实现Delta=1的渐近闭式近似，不声明复现有限期限矩阵精确解，也不引入示例网格、方向预测或返佣。

距离单位为tick；A是每秒到达强度代理，k由现有AS拟合的每价格单位衰减乘tick转换；sigma为1秒价格变化的总体标准差（tick/sqrt秒）。c1=log(1+xi/k)/xi，c2=sqrt(gamma/(2Ak)*(1+xi/k)^(k/xi+1))；半价差=c1+sigma*c2/2，库存偏移=position/inventory_unit*sigma*c2。买卖价限定为被动报价，并经共同订单策略、OMS和交易规则检查。

`gamma=xi=0.05`为参考实现提供的研究起点，没有收益调参或最优性保证；库存单位固定为BTC 0.001、ETH 0.013基础币。AS参数gamma自动初始标定、库存按共同数量上限归一化，GLFT则按明确固定交易单位归一化；两者不能把同名gamma数值直接等同。两者复用同一10分钟/至少120秒、每30秒尝试拟合的到达样本；该拟合使用每秒最远成交穿透与0.5至11.5tick距离，属于到达率近似，不是真实逐订单成交概率，偏离GitHub的采样/拟合范围。GLFT波动用标准差，既有AS仍用价格增量RMS；这些适配差异须随结果披露。

比较配置`config/experiments/as_glft_30day.json`使用BTC/ETH、UTC [2026-08-01,2026-08-31)，7月30日起预热，两家族均关闭微价格/keeper/S1/SC0和额外库存控制。共同600USDC初始资金、1000USDC敞口上限；BTC目标单笔100USDC（两模型一致，避免旧69USDC目标在价格上升时不足最小一手），ETH目标25USDC、maker 0/taker 4bp、50/50撤单队列与既有延迟假设。GLFT不提供盈利用保证。共同保证金压力停机、最小下单量和期末残仓估值仍会影响结果，回放完整不等于全程做市。

### BTC/ETH 报价机制对照

`config/experiments/glft_quote_variants.json`对照原始闭式GLFT及三个单项变化。交易窗为UTC [2026-09-20, 2026-09-27)，9月19日起预热。保留8月比较的资金、下单目标、库存限制、执行与费用配置。回放前固定参数，按币比较期末净PnL和最大回撤，结合1秒成交额加权Markout、成交额、库存暴露及活跃天数判断收益来源。

| 参数 | 报价变化 | 研究依据 |
|---|---|---|
| `inventory_solution=spectral` | 用有限库存网格的最小特征值及正特征向量计算两个报价距离，允许库存惩罚随仓位非线性变化 | [GLFT原论文 Theorem 2](https://arxiv.org/pdf/1105.3115) |
| `reference_price=weighted_mid` | 用当前一档反向数量加权价替换报价中心的中价，风险估值和Markout沿用共同口径 | [Gould–Bonart队列不平衡](https://arxiv.org/abs/1512.03492)、[Cartea–Donnelly–Jaimungal盘口信号](https://ora.ox.ac.uk/objects/uuid%3A006addde-3a03-4d75-89c1-04b59026e1c0)、[HftBacktest盘口不平衡示例](https://hftbacktest.readthedocs.io/en/latest/tutorials/Market%20Making%20with%20Alpha%20-%20Order%20Book%20Imbalance.html) |
| `markout_feedback` | 按侧使用已成熟的近期1秒Markout估计逆向选择，向外移动该侧的加仓报价 | [Barzykin等的逆向选择研究](https://arxiv.org/abs/2508.20225)、[r/quant的Markout讨论](https://www.reddit.com/r/quant/comments/1oxprfm)、[mm-lab开源对照](https://github.com/batuhanboztepe0/mm-lab) |

谱解采用Δ=1、γ=ξ的平稳模型。令 $Q=\lceil\text{position cap}/\text{inventory unit}\rceil$，矩阵对角为 $k\gamma\sigma^2 q^2/2$，相邻非对角为 $-A(1+\gamma/k)^{-(1+k/\gamma)}$。若正特征向量为 $v$，买卖距分别为 $c_1+\log(v_q/v_{q+1})/k$ 和 $c_1+\log(v_q/v_{q-1})/k$。部分成交以 $\log v$ 线性插值，边界外沿用最后一格斜率计算指示价，OMS仍按实际数量和名义上限裁剪。输入 $A,k,\sigma$ 沿用因果估计，每秒更新。默认闭式报价的缓存也按现有采样秒和拟合时间更新。

加权中价只用当时可见的一档盘口，$p=(aQ_b+bQ_a)/(Q_b+Q_a)$。它是简单的盘口代理。mm-lab在自己的七个币对中未发现其预登记的Markout改善，该结果的市场与成交设定与此实验有别。[mm-lab H5](https://github.com/batuhanboztepe0/mm-lab)

Markout为本地规则。每侧最多保留五个不同订单，每个订单使用最近一条已成熟的部分成交样本。至少三个样本，半衰期20秒，90秒过期，迟到上限250毫秒。该侧按时间加权的均值低于−0.25bp时，超出部分最多加宽1bp，并按最新样本的年龄继续衰减。长仓只加宽买价，短仓只加宽卖价，空仓分别评估两侧。未启用这些参数时使用闭式、中价、无Markout反馈。
