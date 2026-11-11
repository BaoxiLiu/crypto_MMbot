# 数据管理

本文件是该主题的统一维护入口。项目总览见 [README](../README.md)。


## 数据位置与验证边界

本页说明存储职责，覆盖范围和可回测性以实际manifest及认证核验为准，不延用历史下载报告的“全量无缺失”结论。

| 数据 | 位置 | 边界 |
|---|---|---|
| 供应商原始行情 | `$QUANT_DATA_ROOT/crypto/raw/hft/binance_futures/` | 保留；下载齐全不代表可直接回放 |
| 自采原始行情 | `$QUANT_DATA_ROOT/crypto/raw/self_collected/binance_futures/` | 保留收到时间与事件内容；local-sui-20260928含从旧回放目录迁出的16份原始文件，逐文件哈希一致，未经当前回放认证 |
| 官方逐笔/资金费/K线 | `$QUANT_DATA_ROOT/crypto/raw/official_binance/` | 保留；funding下包含从旧基线包迁出的BTC/DOGE/1000PEPE原始响应、结算数据与manifest，共9文件，哈希一致 |
| 来源对照中的原始样本 | `$QUANT_DATA_ROOT/crypto/raw/provider_comparison_20261001/` | 含原始自采与供应商样本，不能整目录当作回测结果删除；.partial不得作为完整输入 |
| 认证回放缓存 | `$QUANT_DATA_ROOT/crypto/certified/` | 使用前校验config/datasets登记、窗口及实际manifest身份 |
| 当前NH模型 | 项目data/models/nh-deploy-20260801/ | 13币alpha/value和来源记录，部署日期与特征契约必须匹配 |
| 运行工作区 | 项目data/experiments/.work/ | final_only发布成功后清理；不得删除仍在运行的工作区 |

输入配置在config/profiles/data；正式实验配置决定币种和日期。旧data/runtime、team_baseline、replay_baseline及app/data、app/logs已清理，不再作为运行入口。

迁移先核对原始文件与依赖，再逐文件验证哈希；删除派生产物不能删除共享行情或当前模型。扩展数据覆盖须重新核验，不根据文件数量直接宣称认证完成。

原始数据已迁出项目；17份输入配置指向外部官方行情。既有认证manifest中的旧路径是生成时来源记录，保持原字节和哈希；重新转换按现行输入配置生成新身份，不篡改旧认证。

## 从原始行情到预测和回测

配置和目录见[数据入口](data.md)、[资产位置](data.md)。原始行情不与本次运行的工作区混放。

1. providers读取供应商盘口、官方逐笔和资金费；datasets核对时钟、快照、序列与缺口，形成认证事件缓存。下载完整不等于撮合可靠。
2. config/datasets登记缓存身份；实验prepare核对币种、UTC窗口、来源配置和manifest哈希，worker执行前再次检查登记身份。
3. 因子库计算决策时可见特征；预测样本的未来标签只用于训练与评价。NH训练从认证事件读取部署前样本，保存模型、维度、部署时刻及数据身份。
4. fusion_experiment按币种、策略和费率生成独立账户；event_backtest将行情送入公共Kernel/OMS及显式策略工厂，原生HFT负责模拟撮合。
5. src/metrics统一计算指标；报告模块生成单文件HTML，验证、发布和登记完成后清理本次工作区。

交易所事件时间T、发布时间E、应用可见时间、决策时间和订单到达时间分别处理。以实验执行配置和数据认证口径为准，不沿用旧研究的固定5ms假设。跨缺口和末端库存按有效配置处理，不补造未知成交，不默认认为末端已经平仓。

自采文件保留接收时间、session和序号；有文件及封存哈希并不等于已通过当前连续回放认证。独立币账户求和不代表共享保证金组合。

最终报告保留有效配置、输入/代码身份和有界曲线；逐笔中间文件默认清理。更多口径见[结果解读](backtesting.md)。

## 币安数据在项目中的含义

本文描述本地采集与回放的字段契约，不声明实时接口可用性、服务器部署状态或网络速度。读取实现归app/market_data与app/backtesting/providers；原始行情统一保存在项目外。

| 数据 | 字段或内容 | 用途与边界 |
|---|---|---|
| trade | t、p、q、m、T、E | 成交编号、价格、数量、买方是否maker、事务/事件时间；成交编号不是订单编号 |
| aggTrade | 聚合成交及关联编号 | 聚合成交流；不能冒充完整逐笔队列 |
| depth | U/u/pu、b/a | 更新编号及两侧价位最新总量；数量减少不等于撤单 |
| bookTicker | 最优买卖价及数量 | 独立BBO观察；编号不必逐一递增 |
| REST快照 | lastUpdateId、bids/asks | 有限档位的初始化状态，须与增量编号衔接 |
| 资金费结算 | 结算时刻与费率 | 永续持仓现金流；mark_price观察不替代实际结算记录 |

m=false表示买方主动，m=true表示卖方主动。不使用未核验的X等标签推断方向。价格、数量与币种交易规则共同决定tick/step对齐；缺失规则不可自行猜测。

交易所T/E按源格式读取为毫秒，本地received_ns为应用接收纳秒时间戳。将旧received_ms乘以一百万只改变单位，不增加精度。不同机器时钟未校准时，不把时间差直接视为真实网络延迟。

重建盘口先初始化完整快照，再按更新编号应用增量；跨缺口重置。对价位总量采用替换语义，零量删除。L2没有逐订单排队信息，因此前后各50%撤单是回测假设，不是真实队列观测。成交判定、部分成交、挂撤单延迟归公共执行层，策略不能自行更改。

采集保留原始异常及quality记录；转换仅按显式认证规则排除已核验的占位记录，不静默修补成交。资金费、窗口末端库存和缺口处理遵循有效配置，不假定自动平仓。

操作见[采集器指南](data.md)、[供应商接入](data.md)、[资金费回放](backtesting.md)。当前币种、日期、费用与敞口以[实验规则](backtesting.md)为准。

## 供应商行情接入

本文说明当前本地读取契约；实现见 `app/backtesting/providers/cryptohftdata.py` 和 `datasets.py`。历史样本的格式不能自动推广到其他交易所或新版本数据。

### 文件与字段

原始文件位于 `$QUANT_DATA_ROOT/crypto/raw/hft/binance_futures/YYYY-MM-DD/HH/`。当前读取器逐UTC小时要求 `<SYMBOL>_orderbook.parquet`、`<SYMBOL>_trades.parquet`、`<SYMBOL>_mark_price.parquet`，缺文件直接失败。窗口从context_start（没有时用start）到end_exclusive，不把预热计作交易窗口。

| 字段 | 本地格式含义 |
|---|---|
| received_time | 供应商接收时间，纳秒；不代表纳秒测量精度 |
| event_time / transaction_time | 事件发布时间/事务时间，毫秒；事务时间可缺失 |
| symbol / event_type | 币种及snapshot/update等事件类型 |
| side / price / quantity | 买卖侧、价位和该价位最新总量；价格数量读取时转数值 |
| first_update_id / final_update_id / prev_final_update_id | 深度编号与连续性信息 |
| last_update_id | 快照对应编号，不能当作逐笔成交编号 |

逐笔成交另含trade_time、trade_id、is_buyer_maker。读取器核对币种及接收时间，不把未知方向或缺失时钟补成有效观察。

### 盘口、时间与成交

同价位数量更新是替换，零量删除；数量减少本身不能区分成交和撤单。一条消息可展开为多行，快照必须按完整组初始化一次，不能每行清空盘口。缺少有效快照或序列断链时按配置等待初始化/划分独立区间，不伪造连续性。

publication时钟模式分别处理事务时间T与发布时间E；本地可见事件时间至少不早于事务时间，再加配置的feed_latency_ms。官方补充成交缺少公开发布时间时可参与交易所侧撮合，不凭空制造策略可见事件。具体实现和认证信息以转换器及manifest为准。

官方逐笔根目录位于 `$QUANT_DATA_ROOT/crypto/raw/official_binance/trades/`。占位成交仅按显式策略与官方核验排除，不能任意删掉异常行。由L2派生的BBO不等于独立bookTicker全事件。

### 下载与认证

下载工具为 `tools/data/download_cryptohftdata.py`；通过 `--help` 查看参数。start、end、root为必填参数；运行时显式指定symbols和项目外root，不提供历史日期或项目内下载目录默认值。max-files只做有限下载，不代表整个请求窗口完成。本次文档整理没有发起下载。

转换缓存记录来源及代码哈希；注册配置位于config/datasets，使用前核对实际manifest身份。

`tools/data/prepare_event_dataset.py --rules config/market/规则快照.json` 可在登记前扫描完整事件价格/数量步长。默认遇到偏离步长即拒绝；显式 `--exclude-orphan-depth-deletions` 仅允许排除数量为零的深度删除，且其四舍五入目标在整个预热及交易窗口中从未出现正数量盘口。有效盘口、成交和数量异常仍拒绝；不舍入价格。排除产生新缓存身份，保留父缓存和原始行情，manifest及输入说明记录规则、工具哈希与逐文件审计。
已迁移原始文件的旧manifest保留生成时路径，不改写旧认证哈希。重新转换应使用当前输入路径并产生新身份。

当前数据位置见[资产清单](data.md)，统一回测流程见[数据链路](data.md)。下载齐全、盘口重建和模拟成交可信是三个不同的验收结论。

## 独立行情采集器

采集器只读取公开行情，不读取交易密钥或提交订单。实现为 `app/market_data/collector.py` 和 `storage.py`，本机Python运行；本文不声明任何服务器正在运行。

### 配置与启动

```bash
python -m pip install -r requirements.txt
python -B -m app.market_data.collector check-config --config /absolute/path/collector.json
python -B -m app.market_data.collector collect --config /absolute/path/collector.json
```

从 `app/market_data/config.example.json` 创建配置，显式设置symbols及项目外的output绝对路径，例如QUANT_DATA_ROOT下的crypto/raw/self_collected。示例的七币采集列表不是回测币种规则；回测遵循[统一实验配置](backtesting.md)。check-config只校验配置，collect才会连接网络。

默认channels为book_ticker、depth、agg_trade，depth_speed为100ms；也支持trade与0ms选项。这些是代码支持的配置，不代表当前服务端可用性或延迟保证。启动时核对合约状态。默认空闲超时60秒、深度快照1000档、队列20000条、最低剩余空间5GB；完整参数见示例和load_config。

### 数据与故障边界

先缓存深度增量，再获取快照并衔接U/u/pu；断链后记录quality事件并重新初始化。快照只覆盖返回的有限价位，不能恢复逐订单队列。

消息保留原始data以及run_id、session_id、seq、received_ms/ns。按币种、UTC日/小时与通道保存压缩JSONL；跨通道顺序应结合会话和序号，不能按文件名拼接。接收时间是应用层时间戳，不等于网卡时间戳。

写入队列有界，溢出或磁盘不足时停止，不静默丢数据。每个输出目录只允许一个采集进程；正在写入的文件使用.partial，正常封存后产生含数量、时间范围和SHA256的manifest。崩溃残留不得改名冒充完整输入。health.json提供连接、深度衔接和停止原因，不能代替连续性认证。

### 转存

```bash
python -B -m app.market_data.collector archive --source /absolute/raw/source --destination /absolute/raw/archive
```

目标须在源目录之外，只处理已封存且有manifest的文件并核验哈希。只有显式增加 `--prune --retention-days 7` 才清理已验证的过期源文件；未完成文件和唯一原始副本不能误删。不自动创建定时任务。

采集文件封存不等于通过回测认证，后续流程见[数据链路](data.md)。离线测试覆盖模拟连接、快照衔接、队列与存储行为，不证明真实网络长期稳定性。
