# Stream 严格统计当前v2合同（2026-10-29）

v2是独立审查后的另立软件任务，不是v1重试或预算改写。冻结私有计划 `stream_statistics_v2_software/plan.json` 的SHA为 `eec04aafcab7adf161e5705ba1c6b057f1749006c13175064c29175b8dd929c8`。本次1800秒、64MiB、恰好8个新物化fixture目录，最多两轮输出；复验只能只读复用同8份输入，不能重新创建目录。父任务64实例和预算FAIL永久保留，本次累计上限72。只修改统计器、schema兼容fixture builder、新8项验收文件和本说明；科学交互、真实数据统计、下载均为0。

## 当前v2接口和物理语义

plan/manifest/native row/canonical row/bundle/result全部使用 `*_v2` schema，旧v1明确拒绝。manifest必须声明 `interval_semantics=closed_native_sample_ticks`；`raw_start_tick`和`raw_end_tick`均为inclusive原生采样tick，允许单采样点区间。源clock仍是整数tick、有理数duration和origin，统一到经过外部资格绑定的capture/domain；不能靠任意segment label增加独立样本。

闭区间必须先严格不重叠：端点相等也拒绝（包括quarantine，gap=0亦然）。在已严格分离后，精确时间距离必须不小于冻结minimum gap；正gap的精确相等允许，使用Fraction而非小数截断。状态传输bytes和link frames端点必须是非负原生整数，不能传入秒或分数frame-equivalent；若需连续时长，须另立命名指标、单位及资格。

完整方法×seed×外生条件矩阵、固定final/checkpoint身份、所有原件SHA和case绑定保持强制；分文件路径不参与科学pair。确定规则每条件一条物理记录只逻辑共享到seed配对。full metric×baseline family、unavailable整项保留、Holm p=1占位校正、空/退化效应null与原因、无缺键fallback、create-only和来源篡改拒绝继续保留。

## 当前v2估计量与不确定性边界

估计量是块等权、块内窗口等权、冻结seed/regime权重。训练seed是跨所有物理窗口共享的checkpoint轴：每个bootstrap replicate仅抽一次带配对的seed索引向量，所有被抽到的block/window复用该向量；regime按冻结权重在窗口内重采样。独立块和其窗口仍按层级抽样。该实现名为 `global_crossed_seed__block_window_nested_regime_v2`，不是通用多维协方差或覆盖率保证。

外层paired sign test仍只计独立block；BCa的jackknife轴也是独立block，无法因新增seed行增加外层n。恰好一个独立block时percentile和BCa的时间总体95%区间都标 `unavailable`，上下界null；不能把其内部bootstrap范围当时间总体CI。少于12块保留low-power。全局seed支持、权重和端点校验通过不认证真实独立性、holdout或科学显著性。

BCa在此仅指“outer-block leave-one-out jackknife加速项 + global-seed bootstrap分布”的指定数值算法。未验收joint crossed bootstrap覆盖率、真实95%覆盖率或全family同时置信区间；Holm校正只用于事前全族的sign-test p值，不把逐项区间变为同时family CI。

## 本次合成资格

一次冻结V20–V27八个目录，包含每family事前列出的边界/内存view；输入注册表记录52个原件的hash。pytest仅读取已存在来源，输出以前缀写回同目录；测试本身不创建fixture目录。V20核完整矩阵/单条规则及原有严格门，V21/V22核inclusive边界与exact正gap，V23/V24核整数端点，V25逐replicate核同checkpoint全局seed的精确抽样支持，V26用非均匀权重和3:1窗口数手算估计量并核one-block禁用外层CI，V27拒绝缺失/不支持语义和全部v1 schema。

本次只有首轮目标验收：`python -m pytest tests/test_stream_statistics_contract_closure.py -q -p no:cacheprovider`，8 passed；没有运行原31项suite，没有第二轮或再次物化。软件通过和资源账单分别在私有v2资格中记录，不能抹去下方v1失败。

v2永久限制 `evidence_kind=synthetic_software_fixture`、`paper_goal_complete=false`、`scientific_claim_ready=false`；同一raw SHA映射多个capture/clock仍fail-closed，真实CSV多run需独立native-run选择器资格。本任务没有认证实际capture、预测/成本/外部对照或论文完成。

## 当前CLI

```text
python scripts/analyze_stream_statistics.py produce --plan V2_PLAN --expected_plan_sha SHA --output FRESH_BUNDLE
python scripts/analyze_stream_statistics.py analyze --plan V2_PLAN --expected_plan_sha SHA --bundle V2_BUNDLE --expected_bundle_sha SHA --output FRESH_RESULT
```

---

以下为历史v1合同与原失败结案，保留其原文作provenance；不能按其中旧语义消费v2输入。

# Stream 严格统计软件合同（2026-10-29）

本轮来自独立审查后的 `strict_stream_statistics_software` 任务。只补统计生产/消费接口和合成资格，不重新分析旧科学结果，不训练、评价、拟合、下载或开启 sealed formal/holdout。目标稿件 CSCWD2027；TMC 是内部证据规范；`paper_goal_complete=false`。真实开始时间和 source commit 在私有 `stream_statistics_software/prospective_plan.json`，硬上限32个合成fixture、7200秒、128MiB全部新增软件原件。

## 冻结统计单位、估计量与比较族

独立抽样单元是可信物理资格manifest声明的 `independent_block_id`，时间窗口是块内层，training seed 与 workload regime 是窗口内交叉层。估计量依次按冻结seed权重、regime权重、块内窗口等权、独立块等权平均。外层 paired sign test 和 leave-one-block-out BCa 只用独立块，不能把 seed、workflow、文件或别名增加为样本量。多窗口同block时保留依赖。实际capture身份和独立性仍依赖外部资格原件，不由统计器创造。

比较族必须事前完整枚举 `candidate × all baselines × all metrics`；不得按结果省略指标或baseline。全部family作Holm校正；明确unavailable端点使对应整项统计unavailable，保留在family中按p=1占位校正，其已不可用p/CI仍为null。无静默complete-case筛选。

固定10,000次带配对的层级bootstrap，独立块重采样，块内窗口重采样；窗口内对seed和regime分别按冻结权重抽样，保留其交叉结构。给出percentile和BCa95%区间、块层paired Cohen dz、块层sign test和ties。退化percentile显式degenerate，BCa零jackknife方差/极端bias和效应零方差给null与原因，不能伪造可用区间或零效应。少于12块显式low-power，软件资格不能晋级科学claim。

## 生产与消费 schema

`strict_stream_statistics_plan_v1` 绑定可信manifest、所有native输入JSONL的路径/SHA、指标native字段/方向/单位、完整方法/seed/条件矩阵、固定final/checkpoint provenance、权重、10,000次bootstrap RNG与全部比较族。`strict_stream_statistics_manifest_v1` 声明来源registry、真实capture/clock domain、有理数tick duration和origin、每个物理区间/外层block、quarantine、精确case外生输入及learned/rule身份。别名先归一capture/domain，再以Fraction计算区间；不以任意segment label隔开相同物理源，不截断小数秒。

`produce` 只接受 `strict_stream_native_row_v1` 原生行，按明确 `case_binding_sha256` 连接已冻结manifest；原件SHA和行ID独立保留，文件名不参与pair key。必须原生端点状态、单位与方向及checkpoint/rule身份完整，不猜字段、不缺键fallback。输出create-only `strict_stream_statistics_bundle_v1`，含所有canonical rows、plan/manifest/native文件SHA和软件SHA。

`analyze` 重新核绑定、完整笛卡尔矩阵、身份、物理重叠和quarantine、确定规则每case恰好一条物理引用，输出create-only `strict_stream_statistics_result_v1`。确定规则可逻辑共享到learned seeds，物理评估记录只计一次；learned行不能重复、遗漏或错配外生输入。producer与consumer的schema一起验收；旧统计脚本、环境和baseline不变。

当前资格只允许 `evidence_kind=synthetic_software_fixture`，结果永久标 `software_only`，不声称真实显著性/泛化或认证任何formal/hidden。真实producer资格、物理manifest真实性和独立窗口足够性须另立任务，不把此软件通过当已完成。

## 验收

保留S0正例；独立覆盖S1跨文件、S2外层缺键、S3规则单引用、S4方向单位、S5退化状态、S6多别名、S7精确分数tick。再验收矩阵缺失/重复、原件篡改、schema/CP不匹配、不可用端点family完整性、冻结权重、预算、create-only与来源绑定。失败原始输出保留。fixture总数不超过32，所有fixture和原始pytest输出只写私有任务目录。

v1实际结案：两轮各31项功能测试通过，但第一轮与最终源码SHA不同；最终源码只有第二轮的功能验收。
32个场景物化两轮共64个实例，超过原32实例上限，预算记FAIL，不能称全门通过。原件与偏差报告保留，
不再生成旧fixture或重写原资格。独立只读审查还指出末端inclusive/exclusive语义、原生整数计数和
同checkpoint跨窗seed相关性边界。后续修正另立 `stream_statistics_v2_software/plan.json`，
先明确v2语义和最多8个新物化目录，再匹配验收；不将v1功能通过升级为真实统计资格。

## CLI

```text
python scripts/analyze_stream_statistics.py produce --plan PLAN --expected_plan_sha SHA --output FRESH_BUNDLE
python scripts/analyze_stream_statistics.py analyze --plan PLAN --expected_plan_sha SHA --bundle BUNDLE --expected_bundle_sha SHA --output FRESH_RESULT
```

只有root最后验收、提交和推送，本子任务不改README/PROGRESS/BUGS，不自行提交，不执行真实数据统计。
