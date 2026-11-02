# Helmet24真实任务诊断结果（2026-11-02）

- reviewed_at: 2026-11-02；实际运行与独立审查时间以私有记录为准。
- literature_cutoff: 2026-11-02；本轮复用已核验文献，不新增检索。
- target_venue: CSCWD2027稿件目标；TMC只作内部证据规范。
- policy_version: tmc_review_policy_v3_20260621。
- artifact_run_id: paper_helmet_prospective_20261102_v1/evaluation。
- scientific_git_commit: `3b5cfb794c7a4efaf678a1e006e7a0c43e6a4113`。
- evidence_level: 原件/协议/计算完整性已审计的有限任务诊断；不构成整篇论文E2或formal/holdout资格。
- verdict: 完整执行通过，当前固定接口不支持Helmet adapter质量提升；`paper_goal_complete=false`。

## 实际做了什么

24例选择SHA `388f1cc6939a9e9ae66d507d4cc4f7fc8c309b52477c3b127042ee8568030743`，两类各12、24个不同图像/文件hash，排除旧六图。选择及双映射规则在读取旧六例修复结果前冻结；与预训练或adapter训练的重叠、主体及采集独立性仍未知。

首获取14请求后HTTP200 body超时，6完整、1部分、17未取，所有原件和失败审计封存。[单次恢复](paper_helmet_prospective_recovery_20261102.md)只读引用6完整原图，完整获取其余18；35新请求、13,490,982B body、0新读取上界、503.954452375执行秒。两次合计49请求、1018.663999292执行秒。原额外1,408,825B实体reserve可能与读取上界重叠，保守合计19,610,143B，不是精确wire流量或含准备/等待的全流程历时。

24个原件、GT出处、EXIF方向和native框均核验。独立初轮接受23例，q14小图因遮挡未决；另立单原图上下文检查，确认其为波点布包裹的头部，接受为遮挡例。原80×64裁剪仍有歧义，无法视觉断言布下隐藏状态；保留publisher NoHelmet标签、原裁剪和初轮false报告，模型只读取原裁剪。没有替换样本、重标、提供原图上下文或删除困难例。

模型plan SHA `60d238cae2b366e1d4bcb070f6dcf7eec1e9496d91fdcd2d8e245002f6f8687f`。同冻结base、Helmet权重和旧离线CPU float32/threads1/batch1运行环境；24例×两种相反A/B标签映射×两臂，共96显式评分、96实际native forward、0生成/训练/下载/环境帧。两臂输入tensor、520个实际加载adapter tensor、操作原件及全部结果由自动单次consumer核验。完整CLI为1643.284571292秒，未超5100秒上限，无残留进程组。

## 全24结果，全部保留

| 固定模型 | 总正确 | 有盔正确 | 无盔正确 | 弃答 | 两映射类别冲突 |
| --- | --- | --- | --- | --- | --- |
| Base | 12/24（50%） | 11/12 | 1/12 | 0 | 24/24 |
| Base + Helmet adapter | 12/24（50%） | 12/12 | 0/12 | 0 | 20/24 |

成对结果：两者都正确11、仅base正确1、仅adapter正确1、都错误11。Adapter全24预测HELMET；在本均衡样本上等于常数有盔预测的描述性准确率，未显示准确率增益。它修正了一个base错误，同时破坏了一个base正确例。q14两臂都预测HELMET，按原publisher NoHelmet标签计错，未删例重算主结果。

分数为 `.5 + .5*(pA_map0-pA_map1)`，大于/小于`.5`分别预测HELMET/NO_HELMET，恰等于时弃答并计错。全部float32词表logit保留，由独立float64计算；分数不是校准概率。A+B全词表mass范围：base `0.8822717984514733–0.9997533084713741`，adapter `0.996591384909893–0.9999992473594274`。这与明显映射敏感现象共同报告，不能据此认定某一个因素是唯一原因。

没有独立采集/预训练分离证明，不给确认性p值、CI或泛化结论；不与已消费旧六例合并。旧自由生成严格0/6、修复六例3/6和本轮12/24属于不同输入/接口范围，不能写成格式修复导致“0→50%质量提升”。真实激活adapter、字节/状态可恢复，也不能代替任务质量或系统净收益。

## 可复核交付与验证

主目录私有包 `artifacts/analysis/paper_helmet_prospective_20261102_v1/evaluation/`：

- `audit.json` SHA `86d890fe3ac0ff00d5a702b8a7c266deb5c1cea983542604376e1f62f72f317e`，自动单次审计，未重复运行。
- `choice_mapping_scores.csv`（96行）、`choice_case_scores.csv`（48行）、`prospective_choice_results.json`，原始logit/逐操作日志/加载参数保留。
- `prospective24_confusion.png/.svg`、`prospective24_scores.png/.svg`：已实际查看，所有案例、完整0–1分数轴，无平滑或删例。
- `resources.csv`记录两臂调用数及worker start至complete区间；不含start记录之前的源/初始schema检查，不是推理延迟benchmark。
- `delivery_verification.json`、`input_exposure_and_closeout.json`、`protected_main_delivery.json`及独立交付审查；7个主目录用户源码hash未变，R20/老师稿位未编辑。

独立交付审查接受负结果、无交付blocker（`independent_delivery_review.json` SHA `cf2ff6fa58d085bf3548196dc0b70b3bef18b1ae1336440a26679a8538e5124d`）；核现成审计、全部摘要CSV和图的口径，没有重跑原审计/模型。仅两例预测发生变化：p02仅base正确，p07仅adapter正确。

实际验证入口：`recover_helmet_prospective.py --plan … --plan-sha256 …`及自动新`audit_helmet_prospective_recovery.py`；`run_helmet_prospective_choice.py --plan … --plan_sha256 … --grant …`及自动单次`audit_helmet_prospective_choice.py`。两条真实链路与审计均通过；恢复软件冻结前单轮4微例通过，AST/旧4helper hash/`git diff --check`通过，不追加重跑旧测试或科学。

软件资格没有伪装为同一最终版本完整全链路：新24评分原4微例后有已留档AST静态增量；恢复新producer/consumer精确冻结版本4微例通过，但未做完整24合成CLI。实际这一次真实24执行提供了随后完整链路事实，失败历史与不同资格版本均保留。

## 剩余问题与后续任务边界

本任务补到的是可审计的**质量负证据**与输入/执行链路修复，不能作为“adapter提高真实任务质量”的正证据。下一项有效改进须先建立任务匹配的训练/prompt provenance与独立资格测试合同，再决定模型替换或训练；不是继续在这24例上试提示、挑分数或重复当前权重。

formal/holdout/support、强外部公平对照、模块必要性、真实RSU/无线/provider费用与质量校准净收益仍缺；本轮无新主方法候选、科学训练或环境改变，不晋级canonical、不改论文主表，不称论文完成。
