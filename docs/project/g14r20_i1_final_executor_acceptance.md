# G14R20-I1 evaluation-only 终版实物验收

- reviewed_at: 2026-09-11（Asia/Shanghai）
- literature_cutoff: null（本轮不评价文献或科学结论）
- target_venue: IEEE TMC artifact boundary only；不作 paper-ready 判断
- artifact_run_id: G14R20-I1 / typed_model_cache_evaluation_only_20260911_g14r20_i1_pending
- policy_version: tmc_review_policy_v3_20260621
- status: READY_FOR_EVALUATION_ONLY_AUTHORIZATION
- executor_commit: 9d9f2aeea565ced8d576df3f6a778b4217b58de9
- executor_git_tree: b3258c64b731043608de83c743128f89a0d1eb13
- scientific_commit: a6d1fd822d7d0cb93f7aeadb6b621f0279d95a4d
- original_implementation: fcf36f986261587a1daa70805569cfbf25920332
- original_independent_evidence_head: 42d42466a9e90042834653e9db26005a9f4b0deb
- formal_execution_authorized: false
- formal_execution_started: false
- holdout_opened: false

## 代码、环境和权限

持久、clean detached executor：
`/Users/howen/Projects/PPO_MEC/artifacts/execution_checkouts/g14r20_i1_executor_release`。
HEAD/tree 必须等于上列代码身份。后续证据文档提交位于独立的
`codex/g14r20-i1-acceptance` 分支，不作为 executor 身份。原 I checkout 和原申请包保持原件。

未来 evaluation 使用 `/Users/howen/Projects/PPO_MEC/.venv/bin/python`，实际 Python 3.9.6；
Torch 2.8.0、NumPy 2.0.2、pandas 2.3.3、PyYAML 6.0.3、pytest 8.4.2。
已运行真实冻结 Protocol 2.9 环境 resolver，核验完整依赖指纹、源码清洁度及实际
`src/scripts` import 来源。系统 Python 的编译结果不构成本次验收依据。

`requirements_continuation.txt` 声明的 cryptography 46.0.5 及其依赖只安装在
`artifacts/execution_checkouts/g14r20_i1_acceptance_env` 中。完整回归使用原项目 Python，
通过显式、仅用于测试的 `PYTHONPATH` 加载该隔离目录；科学子进程用真实
`child_environment` 将 `PYTHONPATH` 重置为 clean checkout，保留冻结依赖指纹。
未来 evaluation 命令矩阵不使用隔离解释器，也不包含测试工具 overlay。
没有安装、升级或改写共享科学依赖。

本轮曾尝试直接采用隔离解释器；虽然科学库版本相同，新增 distribution 仍改变完整依赖指纹，
因此该候选被拒绝。失败验收和未签发候选包保留，不被改写为成功。
验收工具完整版本另见 `acceptance_tool_dependencies.json`。

SingleWriter 依赖 `ps` 读取进程身份。沙箱拒绝导致首个 writer 在输出 `locked` 前退出；
原错误缺失的 stderr 不能事后补造，本轮另行捕获了可复现的完整 traceback。
主机权限下真实并发 writer 和 crash/recovery fixture 通过，保留胜者 stdout 与败者
`BlockingIOError`。没有吞掉权限异常、关闭检查或清理真实锁。新 checkout 的 Git LFS
clean 检查也需要本地元数据权限；主机权限下完成核验，没有改动原始数据。

## 最小修复与真实消费者

1. 共享环境来源推断改为禁止父仓库实际 `src/scripts` 包，避免把嵌套 clean checkout
   一并当作 dirty source；实际 import 仍必须位于指定 clean root，父仓库与 shadow import 仍拒绝。
2. evaluation 来源消费者除严格验证原训练身份外，必须验证持久 evaluation contract 及精确 context。
   修改前实际复现了缺失 context、错误新 run/commit 和 stale context hash 被接受；现均拒绝。
   gate 消费者从显式 input root 定位该 evaluation context。
3. 来源审计的 `run_root` 使用 JSON 可序列化字符串。真实 statistics 子进程曾在读取合成 committed
   rows 后因 `PosixPath` 写入 JSON 失败；修复覆盖共享审计的 benchmark/statistics/gate 消费者。
4. 申请包锁目录直接由实际共享 `writer_lock_path` 生成；live validator 拒绝重算 hash 后仍指向
   错误目录的合同，确保申请包与 SingleWriter 实际使用的 `.continuation_locks` 一致。
5. 历史 continuation 测试以 session-scoped 独立 clone 固定科学代码 `a6d1fd8`，保留原解释器和
   import 检查；不恢复已失效临时 v16 目录，不复制真实 run。native 子进程 argv/cwd/stdout/stderr
   保存在 JUnit properties。

新增真实链条：三个合成 controller cells 经 `FormalCellLedger.commit_cell`、producer 清单重定位、
transaction inventory 和最终目录读回，随后真实 statistics 入口及其 nested 子进程消费三个 CSV。
真实 integrity/gate 入口针对缺失 cache/support 证据生成 false gate；completion 拒绝 false/missing gate。
fixture 显式标注 non-formal，未生成任何真实授权。身份、发布、gate 函数没有被 mock。

真实 150 模型按原来源和三容量 companion 只读核验 hash；已有 checkpoint-envelope 测试另以合成
checkpoint 实际执行保存/读回/严格 loader/benchmark main 边界。合成权重、统计行和测试输出不是
真实训练、选模或论文证据。原 selection/freeze/provenance、科学参数、旧失败 run、七个用户文件、
holdout 状态保持不变。

## 本地产物和新申请包

证据根目录：
`/Users/howen/Projects/PPO_MEC/artifacts/analysis/g14r20_i1_final_acceptance_20260911`。
该目录受 Git ignore 规则管理，未虚构 artifact commit；模型、原始数据、旧 run 不上传。
Git 只提交本次实现/测试/文档，文档记录本地证据 SHA-256。

| 交付 | 文件 |
|---|---|
| 原 145 项异常逐项根因、影响和最终匹配 | `regression_failure_triage.json`、`.csv`、`.md` |
| 最终代码、环境、命令及 JUnit | `final_executor_acceptance.json` |
| Python、依赖、import 来源与代码文件 hash | `executor_environment_release.json` |
| 冻结科学环境真实核验 | `release_frozen_environment_contract_validation.json` |
| 唯一终版未签发申请包 | `authorization_request_release.json` |
| 申请包与环境完整性绑定 | `authorization_request_integrity_manifest.json` |
| 固定 checkout 和未来命令 | `future_commands_release.json` |
| 候选包作废/当前状态索引 | `request_status_index.json` |
| 独立于旧摘要的验收结论 | `independent_acceptance_conclusion.md` |

终版 request semantic SHA-256：
`838f43b4335fbc0ce99572c1b1ba64132dc8ab5dbfaba5c6182e4dca21d24af8`。
execution contract SHA-256：
`4ed90972ee0d636aa97b1bda40bc48798964e2ed09c8b18820c4d12111fc269e`。
model-source reference SHA-256 保持：
`0ba289caa663d38dd35f61feafa509a3f8dc2eb4269b3220f18d383be81c3ea3`。

真实 prepare/validate 已核验 150 模型、三容量、八阶段、24 commands、三个 nested benchmark commands
与最终 executor/Python 的绑定。重复 prepare 被 create-only 拒绝，文件字节未变。
原 I 包和本轮错误锁目录候选被 release 的实际 validator 拒绝。
旧 288 MB partial cell 和 576 MB staging 继续排除；holdout capability=false。
待执行目录不存在，没有签发 grant、创建正式 run root/ledger 或运行正式 execute。

## 验证和结论边界

最终全仓 1659 passed，必经目标 54 passed，独立 single-writer 2 passed，均为 0 failure/error/skip。
原 145 项异常已按相同 node ID 逐项匹配至最终通过记录；没有批量 skip 或未闭环项。
本轮不评价正式性能、baseline 优势、paper-ready、canonical 或文献 novelty，科学评分 N/S。
这里的独立复核指重新运行、检查原始证据，不沿用旧 I readiness 摘要；不冒充第二位人员/agent
签署独立授权审查，也不能替代未来独立 review/grant。


## 最终验证命令与证据锚点

完整命令、cwd、环境、权限、返回码及 stdout/stderr 由 `release_full.command.json`、
`release_active_targets.command.json`、`release_single_writer.command.json` 保留。

- `/Users/howen/Projects/PPO_MEC/.venv/bin/python -B -m pytest tests`：1659 passed，1028.76 秒。
- 同解释器运行 evaluation-only 五组测试和两个单写者 node：54 passed。
- 同提交独立运行两个单写者 node：2 passed。
- 真实 `prepare_typed_model_cache_evaluation_only.py --action prepare/validate`：通过；
  150 models、8 phases、24 commands、3 nested commands；重复 prepare 按 create-only 拒绝。
- 冻结环境 resolver、完整性清单和保护核验：通过；150 checkpoint hash 与七个用户文件未变，
  旧 run 109620 个文件的 size/mtime_ns 清单无变化。此元数据清单不冒称对每个旧 payload 全量重算 hash。

全仓测试使用显式测试工具 overlay；未来科学执行不使用该 overlay，详见环境说明与 RUNBOOK。
只读审计产物被 Git 忽略，以下锚点记录其实际本地字节：

| 文件 | SHA-256 |
|---|---|
| `final_executor_acceptance.json` | `64eecd485b99e7f66a809d393b01f5cc3b16614dfe747c6285111a40706f1384` |
| `regression_failure_triage.json` | `8e00b960d30e2356d1cc79649d2f3053b6aee65b697f43518b3a1c48c6b0d290` |
| `release_full.junit.xml` | `b4de9395a8c4601ce055e18e0ee376d3c2e581c7b82851bfa788beae43e2927b` |
| `release_active_targets.junit.xml` | `5c7e7c8a039c2458ed994a429cfac5c8bd7e6f46bc00fe414b2814ad1fde0573` |
| `release_single_writer.junit.xml` | `f411eab038e940b277b2b64696bae982c64a2857bcb8790bcf941413019c9fb4` |
| `authorization_request_release.json` | `6383ea305d72f0eb14263bcea8c7d3730e1cbb28c8b74e7d00ca80b8f237d232` |
| `authorization_request_integrity_manifest.json` | `b30eab930d26e86dac79131f3bfe98db094667e02d320359e5544c161e444dcf` |
| `executor_environment_release.json` | `c87726b5a0479b47eba0234056b9895b594d22d7613b6001f09393474b2c4f88` |
| `release_frozen_environment_contract_validation.json` | `f1296bc90a6177be240b7cac58c63c16298a01bb4881845ad7a6ed036c458118` |
| `protected_assets_verification.json` | `9aabd512c93829f15c873e78658489acbbb766c289da7fd992166e83af738c65` |
| `independent_acceptance_conclusion.md` | `14d97ae364b7e0871ddd2dd26bd4f3dbc315da1a92a331aa97e83b42fedad4f9` |
