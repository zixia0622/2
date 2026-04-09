补贴建模交接说明
================

文档目的
--------
这份文档用于交接当前 `subsidy_modeling/` 目录下的建模代码，重点面向后续继续推进
“大样本 + 内生预算优化 DFCL 训练”的接手同学。

如果只想用一句话理解当前进度，可以概括为：
小样本 baseline 已经完整跑通，sklearn 版升级链路已经跑通，带有内生 dual 变量（`lambda`）
和软/硬统一求解接口的 DFCL 风格原型，也已经在中小样本上跑通。


截至目前已完成的工作
--------------------
1. 竞赛原始数据可以从 RAR 包中自动抽取、读取、清洗，并转换成可建模的数据集。
2. 已经建立了统一的 `用户-BU-日期`（`user-BU-day`）决策面板作为核心样本粒度。
3. 已定义多种补贴 treatment，并以 `T0_no_coupon` 作为基准动作。
4. 已基于访问、交易、领券、画像数据构建历史行为特征。
5. 已构造未来 7 天结果变量，包括收入和补贴成本。
6. 目前支持三种建模模式：
   - `empirical`：按 BU 聚合的经验型回退 baseline
   - `sklearn`：多分类倾向模型 + treatment-specific ridge 结果模型
   - `dfcl`：将预算压力嵌入训练过程的决策导向原型
7. 优化器已经从“只有 hard 离散求解”升级成“软/硬双接口统一”：
   - soft solver：用于训练期，构造可微的预算感知策略
   - hard solver：用于推断期，生成最终离散发券决策
8. 已经在一个中小样本 DFCL 实验中观察到正且非零的 dual 变量（`lambda`），说明预算约束
   已不再只是训练后的后处理概念，而是真正进入了训练过程。


今天具体做了什么
----------------
今天的核心推进，是把代码从“先预测、后优化”的 baseline，推进到了一个更接近 DFCL 思路的
决策导向建模版本。

具体包括：
1. 保留了原有 baseline 主链路，确保旧版本仍可运行。
2. 稳定了 sklearn 版 DR 估计链路：
   - 改进了特征类型处理
   - 将 datetime 列从编码输入中剔除
   - 提升了预处理稳定性
3. 强化了预算优化部分的业务合理性约束。
4. 实现了 DFCL 风格训练路径：
   - 模型直接学习各 treatment 下的收入和成本预测
   - 训练时使用 `value - lambda * cost` 形式的预算感知 gain
   - 用 soft policy 近似可微动作选择
   - 在训练中同步更新 dual 变量（`lambda`）
5. 统一了训练期和推断期的求解逻辑，让两者共享同一套 action tensor / gain 机制。
6. 升级了运行脚本，使其同时输出 hard 和 soft 两套 allocation summary。

可以用一句话概括今天的成果：
- baseline 可运行
- sklearn 升级版可运行
- DFCL 风格的小样本 / 中样本原型可运行
- 下一步重点是大样本正式训练与调参


当前成熟度判断
--------------
需要准确理解当前版本的成熟度，不要夸大，也不要低估。

已经完成的部分：
- 小样本端到端 baseline 已完成
- 小样本 sklearn uplift + optimization 已完成
- DFCL 风格原型已经实现并可运行
- 在 tighter budget 设置下，`lambda` 已经可以被压出来
- soft / hard solver 已经打通并保持一致逻辑

尚未完全完成的部分：
- 大样本或全样本 DFCL 训练还没有最终跑完
- 面向大规模内生预算优化的超参数调优还没有完成
- 报告级别的全量正式结论还没有冻结
- 当前 DFCL 实现是一个工程上可运行的原型，不是完全照论文复刻的神经网络端到端 solver

准确描述当前状态的说法应当是：
这个仓库已经具备交接给下一位同学继续做大样本 DFCL 实验的条件，而不是还停留在从零开始阶段。


目录结构总览
------------
`subsidy_modeling/` 目录下当前主要文件如下：

- `config.py`
  全局配置文件。定义路径、时间窗、默认预算和 DFCL 超参数。

- `data_io.py`
  负责从 RAR 包中抽取四张原始 CSV，并读取成 pandas DataFrame。

- `preprocess.py`
  负责清洗原始表，并生成若干画像衍生特征。

- `features.py`
  最重要的数据工程模块。负责构建 `user-BU-day` 面板、treatment、历史特征和未来 7 天结果变量。

- `pipeline.py`
  串联完整的数据构建流程，从原始表一路生成最终建模数据集。

- `modeling.py`
  核心建模模块，包含 empirical、sklearn、dfcl 三种模式。

- `optimization.py`
  核心预算分配模块，包含统一的 soft / hard 求解接口。

- `results.py`
  负责生成 treatment summary、budget action mix、BU summary 等汇总结果。

- `run_modeling.py`
  命令行主入口。接手人平时最常用的运行脚本。

- `utils.py`
  一些辅助函数，例如创建目录、RAR 抽取。

- `data/`
  本地抽取后的原始数据和 processed 数据集。

- `outputs/`
  运行脚本产生的建模输出结果。


逐文件说明
----------

1）`config.py`
--------------
这个文件存放所有可复用的配置项。

主要包括几类内容：
- 路径配置：
  - 项目根目录
  - 原始数据目录
  - 处理后数据目录
  - 输出目录
- 原始文件名：
  - profile
  - visit
  - coupon
  - order
- 建模默认参数：
  - 历史特征时间窗：7 天、30 天
  - 结果窗口：7 天
  - treatment 最小样本量
  - 默认预算网格
- DFCL 超参数：
  - 训练轮数
  - 学习率
  - L2 正则
  - 决策损失权重
  - softmax 温度
  - cost 平滑温度
  - dual 变量更新步长

为什么重要：
接手人后续做大样本 DFCL 调优时，最先要看的通常就是这个文件。
最可能需要调整的参数包括：
- `BUDGET_GRID`
- `DFCL_EPOCHS`
- `DFCL_LR`
- `DFCL_DECISION_WEIGHT`
- `DFCL_TAU`
- `DFCL_DUAL_LR`


2）`data_io.py`
---------------
这个文件负责把 RAR 数据包变成可用的 pandas 表。

具体做的事：
- 定义每张 CSV 对应哪个逻辑表
- 如果本地不存在，则从 RAR 包中抽取出来
- 使用 pandas 读取为 DataFrame

重要特点：
- 支持 `sample_nrows`，非常适合做 smoke test 和小样本调试
- 也正因为有这个参数，前期可以快速验证代码，而不必每次都加载全量数据

交接提醒：
如果后续做大样本，I/O 可能成为瓶颈，这个模块可能需要继续优化，例如：
- 分块读取
- 转 parquet
- 做本地缓存


3）`preprocess.py`
------------------
这个文件负责对原始表做标准化清洗。

主要功能：
- 把日期列转成 `datetime`
- 把金额列转成数值型
- 去重
- 统一 BU、优惠券状态等字段类型
- 把画像里的等级字段转成数值特征

为什么重要：
它保证下游特征工程和建模代码不会因为脏数据类型而报错。后面的 sklearn 和 DFCL 部分，
都依赖这里先把数据类型处理正确。


4）`features.py`
----------------
这是整个数据构建流程里最核心的文件。

全套建模样本围绕一个中心粒度：
`user + BU + decision_date`

这个文件的核心逻辑是：
1. 用交易日期、领券日期、访问 fallback 日期构造 `user-BU-day` 主面板。
2. 确定当天的主券动作。
3. 把原始券信息离散成 treatment 标签。
4. 对每一行样本向前看 7/30 天，构造：
   - 访问历史特征
   - 交易历史特征
   - 领券历史特征
5. 合并用户画像特征。
6. 对每一行样本向后看 7 天，构造结果变量：
   - `Y_rev_7d`
   - `Y_cost_7d`
   - `Y_visit_7d`
   - `Y_order_7d`

当前 treatment 示例包括：
- `T0_no_coupon`
- `T1_low_coupon`
- `T2_mid_coupon`
- `T3_high_coupon`
- 稀疏或未知 bucket

非常重要的工程提醒：
这个文件里有很多基于 group 和逐行时间窗扫描的显式循环。它在小样本和中样本下可读性好、
逻辑清晰，但如果后续做大样本，全量运行速度可能会成为瓶颈，这个文件很可能是第一优先级
优化对象。


5）`pipeline.py`
----------------
这个文件是数据构建总调度器。

它的职责可以理解为：
- 读取原始表
- 清洗数据
- 增加画像衍生特征
- 构建主面板
- 分配 treatment
- 增加历史特征
- 构造结果变量
- 做缺失值收尾处理

为什么重要：
它是运行脚本里真正用来生成建模数据集的主函数。
如果后续有人想替换某部分特征逻辑，通常也还会保留这个文件作为总流程 orchestration 层。


6）`modeling.py`
----------------
这是核心建模模块。

目前支持三种模式。

A. `empirical`
~~~~~~~~~~~~~~
这是经验型回退 baseline。
它不训练通用机器学习模型，而是基于 BU 粒度的经验均值加 shrinkage 来估计倾向和结果。

适用场景：
- 快速调试
- sklearn 不可用时的回退版本
- 做 sanity check

B. `sklearn`
~~~~~~~~~~~~
这是升级后的标准机器学习 baseline。

它使用：
- 多分类 logistic regression 拟合 treatment propensity
- 针对每个 treatment 的 ridge regression 拟合结果

之后用 doubly robust 方法生成：
- 收入 DR 估计
- 成本 DR 估计

再进一步计算相对不发券的增量量：
- `delta_rev_*`
- `delta_cost_*`
- `delta_roi_*`
- `dual_score_*`

适用场景：
- 强于经验型 baseline
- 更容易解释
- 可以作为 DFCL 前的标准对照组

C. `dfcl`
~~~~~~~~~
这是今天新增和强化的重点模式。

概念上它做了以下事情：
1. 仍然使用 sklearn 的多分类 propensity 作为偏差校正部件。
2. 从 panel 中构造 dense features。
3. 训练 treatment 级别的 revenue / cost 预测头。
4. 在训练中构造 `gain = value - lambda * cost`。
5. 用 soft action policy 近似可微动作选择。
6. 在训练中同步更新 dual 变量 `lambda`，让预算压力真正进入训练过程。

为什么重要：
这是本仓库第一次实现“预算优化不再只是训练后处理，而是训练目标本身就感知预算约束”的版本。

重要边界：
当前实现是一个工程上可运行的 DFCL 风格原型，不是严格按论文做的隐式微分 / 大型神经网络 solver。
但它已经足够作为下一位同学继续做大样本实验的交接基础。


7）`optimization.py`
--------------------
这个文件今天被显著升级。

以前：
- 只有 hard 离散分配逻辑
- 主要用于训练后，根据 `delta_rev / delta_cost` 做预算分配

现在：
- soft 和 hard solver 共用同一套动作表示
- 训练期和推断期使用一致的 gain 定义

主要函数概念：
- `prepare_action_tensors()`
  把每个用户-每个 treatment 的增量收入和增量成本，转成统一动作张量
- `gain_matrix()`
  按 `value - lambda * cost` 计算动作得分
- `soft_action_probabilities()`
  为训练阶段提供可微软策略
- `choose_actions_for_lambda()`
  支持 hard / soft 两种动作选择方式
- `solve_budget_allocation()`
  在给定预算下搜索合适的 `lambda`
- `summarize_allocations()`
  生成适合汇报和对比的预算摘要表

为什么这部分重要：
它现在已经变成连接“训练时的决策感知目标”和“推断时的预算分配”的桥梁。
这也是今天最重要的升级之一。


8）`results.py`
---------------
这个文件负责把原始打分表整理成可读的结果表。

当前支持的输出包括：
- treatment summary
- budget action mix
- BU summary

并且现在支持 hard 和 soft 两种 budget action summary。
也就是说，接手人可以同时查看：
- 训练期 soft policy 的概率视角
- 推断期 hard policy 的离散决策视角


9）`run_modeling.py`
--------------------
这是主运行脚本。

关键参数：
- `--sample-nrows`
  控制每张原始表读取多少行，用于小样本实验
- `--output-prefix`
  控制输出文件名前缀
- `--model-mode`
  选择 `auto`、`dfcl`、`sklearn`、`empirical`
- `--budget-grid`
  自定义预算列表，例如 `5,10,20,40`

当前该脚本会输出：
- processed dataset
- DR scoring 文件
- hard allocation summary
- soft allocation summary
- treatment summary
- hard action mix
- soft action mix
- BU summary

对于交接来说，这就是最主要的运行入口。
多数实验都可以先从这个文件开始，不需要一上来就改别的模块。


10）`utils.py`
--------------
这个文件放了一些辅助工具函数。
虽然不属于建模核心逻辑，但它负责：
- 创建目录
- 从 RAR 包中抽取文件


当前模型到底在学什么
----------------------
这一节专门用非常直白的话写给接手人。

输入是什么
~~~~~~~~~~
每一条训练样本都是一个 `user-BU-day` 行，里面包含：
- 用户画像信息
- 最近访问行为
- 最近交易行为
- 最近领券行为
- 当前业务线 BU
- 当前 treatment 标签

训练目标是什么
~~~~~~~~~~~~~~~
代码要学习的是：
对于每种可能的 treatment，给这个用户以后，未来 7 天会带来：
- 多少收入
- 多少补贴成本

在 `dfcl` 模式下，训练并不止于此。
它还会进一步要求这些预测结果，能更好地服务最终预算约束下的发券决策。

输出是什么
~~~~~~~~~~
对每个样本和每个 treatment，模型最终会产生：
- 预测收入
- 预测成本
- DR 修正后的收入
- DR 修正后的成本
- 相对不发券的增量收入
- 相对不发券的增量成本

这些量随后会被交给预算优化器，决定最终发券动作。


原始 CSV 是怎么匹配起来的
--------------------------
这一部分对交接尤其重要。

系统并不是把所有 CSV 直接做一次大宽表 join，而是采用“先构造决策面板，再逐步挂特征和结果”的方式。

第 1 步：
先用以下日期构造中心主面板 `user-BU-decision_date`：
- 交易日期
- 领券日期
- 访问 fallback 日期

第 2 步：
根据以下 key，把当天主券动作挂到面板上：
- `User_id`
- `BU`
- `Decision_date`

第 3 步：
按时间窗回看历史，构建 rolling 特征：
- 访问特征：按 `User_id`
- 交易特征：按 `User_id + BU`
- 领券特征：按 `User_id + BU`

第 4 步：
按 `User_id` 合并画像表

第 5 步：
从 `Decision_date` 起向后看 7 天，构造未来结果变量

所以整套系统不是围绕静态用户表，而是围绕一个“决策时点面板”来组织的。


今天已经产出的代表性结果
------------------------
今天开发过程中，已经产出过一些代表性输出，例如：
- sklearn smoke / quick test 输出
- DFCL smoke 输出
- 使用 tighter budget 的 `dfcl_probe_*` 输出
- 使用更接近正式实验规模的 `dfcl_formalish_*` 输出

今天 DFCL 最重要的结果是：
在收紧预算网格后，dual 变量已经不再全部为零。
这意味着“内生预算约束”已经真正开始起作用。

代表性的 DFCL dual 变量结果示例：
- budget 5  -> lambda 大约 0.092
- budget 10 -> lambda 大约 0.081
- budget 20 -> lambda 大约 0.058
- budget 40 -> lambda 大约 0.013

这组结果的经济学解释是：
- 预算越紧，影子价格越高
- 预算越宽，影子价格越低

这个方向是符合预期的。


推荐给接手人的运行命令
----------------------
1. 快速 smoke test：
   subsidy_modeling/.venv/bin/python -m subsidy_modeling.run_modeling --sample-nrows 2000 --output-prefix smoke --model-mode empirical

2. sklearn baseline 测试：
   subsidy_modeling/.venv/bin/python -m subsidy_modeling.run_modeling --sample-nrows 5000 --output-prefix sklearn_test --model-mode sklearn

3. DFCL 紧预算探测实验：
   subsidy_modeling/.venv/bin/python -m subsidy_modeling.run_modeling --sample-nrows 3000 --output-prefix dfcl_probe --model-mode dfcl --budget-grid 1,2,4,8

4. DFCL 中等规模 formalish 实验：
   subsidy_modeling/.venv/bin/python -m subsidy_modeling.run_modeling --sample-nrows 8000 --output-prefix dfcl_formalish --model-mode dfcl --budget-grid 5,10,20,40

5. 后续建议尝试的大样本交接实验：
   subsidy_modeling/.venv/bin/python -m subsidy_modeling.run_modeling --sample-nrows 20000 --output-prefix dfcl_20k --model-mode dfcl --budget-grid 10,20,40,80

注意：
预算网格应结合当前成本尺度来选。
如果所有 lambda 都还是 0，通常说明预算过宽，没有真正压住预测支出。


输出文件应该怎么看
------------------
对于一个给定前缀，主要输出包括：

- `*_model_dataset.csv`
  最终工程化后的建模数据集

- `*_dr_scoring.csv`
  每条样本的 treatment 级预测与 DR 打分表

- `*_allocation_summary.csv`
  hard 离散分配结果摘要

- `*_allocation_summary_soft.csv`
  soft 可微分配结果摘要

- `*_treatment_summary.csv`
  treatment 级整体质量检查表

- `*_budget_action_mix.csv`
  hard 动作分配结构

- `*_budget_action_mix_soft.csv`
  soft 概率质量分配结构

- `*_bu_summary.csv`
  BU 粒度描述性摘要

建议新接手的人按以下顺序看结果：
1. `*_allocation_summary.csv`
2. `*_allocation_summary_soft.csv`
3. `*_budget_action_mix.csv`
4. `*_treatment_summary.csv`
5. `*_dr_scoring.csv`


当前已知限制与风险
------------------
1. `features.py` 中有较多循环式逻辑。
   可读性好，但大样本下可能较慢。

2. 当前 DFCL 实现是 NumPy 原型。
   适合做实验验证，但如果后续想做：
   - 更大特征空间
   - 更强表达能力
   - GPU 训练
   - 更接近论文的神经网络结构
   则可能需要重构。

3. treatment summary 的全局均值仍可能整体偏负。
   这并不一定意味着策略无效，常见原因是：
   全样本平均不值得发券，但某些局部高价值人群仍值得定向投放。

4. 预算尺度对 lambda 行为影响非常大。
   如果预算太大，约束就不会绑定，lambda 可能退化接近 0。

5. 当前还没有自动化的模型选择、交叉验证或大规模实验管理机制。
   这些都可以作为未来增强项。


推荐给下一位接手人的后续工作
----------------------------
优先级 1：做更大样本 DFCL 训练
- 跑 20k 甚至全样本 DFCL 实验
- 监控运行时间和内存
- 检查 lambda 是否持续保持经济意义

优先级 2：做超参数调优
- 调预算网格
- 调 `DFCL_DECISION_WEIGHT`
- 调 `DFCL_TAU`
- 调 `DFCL_DUAL_LR`
- 调 epoch 和 learning rate

优先级 3：做性能优化
- 加速 panel 特征构建
- 减少对历史表的重复扫描
- 考虑缓存或预聚合

优先级 4：做建模增强
- 如果需要，可以把 DFCL 的线性 head 换成更强模型
- 尝试更强的 outcome function
- 提升 cost 预测的校准性

优先级 5：做评估与汇报
- 系统性比较 empirical / sklearn / dfcl
- 检查不同 budget grid 下的稳定性
- 总结不同预算下哪类券最常被选中


最简交接摘要
------------
如果接手人只看这一段，也应该能快速理解当前状态。

- 仓库已经支持从原始压缩包到预算分配结果的端到端运行。
- 小样本 baseline 已完成。
- sklearn 版 DR uplift 已完成。
- DFCL 风格决策导向原型已经实现并跑通。
- 训练和推断现在共享统一的 soft / hard 预算求解接口。
- 在 tighter budget 设置下，已经观察到非零 lambda。
- 下一步重点是做更大样本 DFCL 训练和调参，而不是从零开始搭建整条链路。


环境说明
--------
建议环境：
- Python 虚拟环境位于 `subsidy_modeling/.venv`
- 至少安装以下依赖：
  - pandas
  - numpy
  - scipy
  - scikit-learn

典型安装命令：
  subsidy_modeling/.venv/bin/python -m pip install pandas numpy scipy scikit-learn


文档结束
--------
如果明天要继续推进，建议从 `run_modeling.py` 开始，先看最近产出的 output 前缀，
再进入 `modeling.py` 和 `optimization.py` 做大样本 DFCL 调参与训练延展。
