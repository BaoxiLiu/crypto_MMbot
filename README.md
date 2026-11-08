# Crypto MMbot

本机Python加密资产做市研究框架。包含AS（原始/改进）、NH预测报价、HJB（固定/自适应/库存预算）三个策略家族，同类差异通过参数控制，共用执行、账户、指标和报告流程。

## 快速开始

使用Python 3.11，安装根requirements.txt中的依赖，以及已验证的本机makerfix2引擎。解释器配置在paths.local.json，行情根目录通过QUANT_DATA_ROOT指定；具体安装限制见[开发说明](doc/development.md)。

```sh
python -B -m app backtest strategies
python -B -m app backtest validate-config --config config/experiments/default.json
python -B -m app backtest run --config config/experiments/default.json
python -B -m app check --native --tools
```

默认实验为13币、UTC 8月1日至3日、原始/改进AS、双maker费率。首次运行前核验数据和模型；局部验收不证明完整窗口收益，NH历史参数使用限制见模型说明。在线下单必须另外明确授权。

已有家族的新配置参考[BTC/ETH参数示例](config/experiments/as_s1_ablation.json)：只改参数，复用数据、执行和标准报告。新增数学家族先遵守[策略接入合同](doc/development.md)，无需修改公共引擎。架构边界、循环依赖、模块预算和产物规则由`app check`强制检查，规则见[AGENTS](AGENTS.md)。

## 项目结构

| 目录 | 职责 |
|---|---|
| app | 行情、因子、预测、策略、执行、回测与报告 |
| src/metrics | 公共指标计算 |
| config | 数据、策略、账户及执行配置 |
| tools | 下载、训练、诊断与开发检查 |
| data | 当前模型与运行登记；临时工作区发布成功后清理 |
| doc | 内部技术说明、因子定义与最终HTML报告 |
| notebooks | 行情浏览及探索笔记 |

原始行情与认证缓存保存在项目外。回测默认最终只发布单文件HTML及轻量登记，表格和图表按币种联动，并分别比较maker 0和−0.3bp情景。

## 技术文档

- [开发与架构](doc/development.md)：模块边界、环境、检查和开发规则。
- [策略](doc/strategies.md)：AS/NH/HJB的逻辑、出处与参数。
- [回测](doc/backtesting.md)：统一配置、指标、报告和清理契约。
- [数据](doc/data.md)：数据位置、格式、采集与认证。
- [因子](doc/factors.md)：分类、计算与详细定义。
- [模型](doc/models.md)：训练、使用条件及评测协议。

根AGENTS.md供开发助手读取。新说明归入对应主题，不在代码目录分散新增README或历史过程文档。
