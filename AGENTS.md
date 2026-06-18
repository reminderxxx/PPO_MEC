# AGENTS.md

用途：PPO_MEC 的长期 AI 协作约束。这里写稳定规则，不写单次实验结论。

## 项目基线

- 项目名称：`PPO_MEC`
- 项目定位：面向 AI-driven VEC 的研究原型，主线是跨 RSU 连续 DAG workflow 执行、adapter cache 协同、handoff 状态迁移和多时间尺度控制。
- 默认沟通语言：中文。代码标识符、命令、路径、字段名和报错保持原文。
- 仓库根目录是命令默认执行位置。
- 主要代码目录：`src/`
- 主要脚本目录：`scripts/`
- 主要配置目录：`configs/`
- 正式产物根目录：`artifacts/`
- 真实数据根目录：`data/`
- 长期项目文档入口：`docs/project/`

## 阅读顺序

非简单任务先读：

1. `AGENTS.md`
2. `README.md`
3. `docs/project/CONTEXT.md`
4. `docs/project/PROGRESS.md`
5. `docs/project/BUGS.md`
6. `docs/project/README.md`
7. 与任务相关的 `scripts/`、`configs/`、`src/` 模块

数据准备、真实 dry-run 和 benchmark 操作优先看 `docs/project/RUNBOOK.md`。
历史 artifact 结论优先看 `docs/project/ARTIFACT_RECORDS.md`。

通用模板内容已整理进 `docs/project/`。项目内的有效规则以根目录 `AGENTS.md` 和 `docs/project/` 为准。

## 修改规则

- 默认最小改动。不要重构无关模块，不移动已归档数据、checkpoint 或历史实验产物。
- 每次完成代码、配置、脚本或长期文档更新后，必须在匹配验证通过后提交到 Git，并执行 `git push` 同步到 GitHub 远程；若 push 因认证、网络或远程状态失败，最终回复必须说明失败命令、错误原因、本地 commit 状态和用户需要执行的后续命令。
- 提交和 push 前必须检查 `git status`，只纳入本次任务相关文件；不得把无关未跟踪文件、真实数据、checkpoint、缓存或历史实验产物混入提交。
- 新增脚本、配置和文档文件默认使用小写 `snake_case`；已有入口为兼容可以保留原名。
- `src/agents/` 只按算法分文件：主方法和对比方法都直接放在根目录算法文件中，`registry.py` 直接导入算法文件并映射 agent 名称。
- PPO / MAPPO 必须分别由 `ppo_agent.py` / `mappo_agent.py` 承载，不再通过 `ppo_family.py`、`baselines/` 或 `marl/` package 包装组织。
- 未稳定 contract 的 TD3 / SAC / MADDPG 不进入 live registry；QMIX 已按 controller-level value-decomposition contract 接入。后续若要接入 full vehicle/RSU-level QMIX 或连续控制算法，必须先冻结匹配的 observation/action contract，并补齐训练、评估和 benchmark 消费端。
- 涉及 schema、manifest、checkpoint、接口字段、输出目录约定变化时，必须同步检查生产端和消费端。
- 涉及目录结构、主入口、正式产物路径或运行流程变化时，同步更新 `README.md`、`docs/project/DIRECTORY_STRUCTURE.md` 和 `docs/project/RUNBOOK.md`。
- 涉及模块职责或依赖方向变化时，同步更新 `docs/project/CODE_MODULE_MAP.md`。
- 涉及长期设计取舍时，更新 `docs/project/DECISION_LOG.md`。
- 不把未验证猜测写成事实；未验证项必须显式标注。

## 研究主线约束

- 当前正式主线优先级：`NGSIM + Alibaba`。
- `LuST` 保留 provider 和导出脚本，但当前不阻塞正式主线。
- `highD` 保留为后补数据源和 provider 骨架。
- `smoke_run` 只用于联调，不用于论文结论。
- 正式主表优先使用 `scripts/benchmark_main_results.py` 的多窗口、多 seed 输出。
- 对算法效果的表述必须能回溯到 `artifacts/` 下的明确产物。
- 文献检索或浏览网页时，若发现与本项目方向相关且尚未记录在 `docs/project/literature_reference_table.md` 的论文，必须同步追加到该表；按表内维护规范补齐“可提供参考点”和“PPO_MEC 优化点 / 差异点”，不确定的 venue、年份、DOI 或结论必须标注待核验。

## 顶刊 AI 审查规则

- 涉及论文结论、实验晋级、canonical 更新、paper-ready 判断或文献水平评价时，必须先读并执行 `docs/project/top_journal_review_policy.md`。
- 审查顺序固定为 artifact 完整性、protocol/provenance、统计与 baseline 公平性、机制兑现、claim 边界、文献 novelty；不得只依据 README、PROGRESS 或 gate 的 `paper_claim_ready=true` 下结论。
- 缺少正式 checkpoint、manifest、command log、formal/holdout/support 原始结果时，结论必须为 `Unverifiable` 或显式 `UNVERIFIED`，不得用文档摘要补位。
- 审查输出必须包含 `reviewed_at`、`literature_cutoff`、`target_venue`、`artifact_run_id`、`policy_version`、Git commit 和 evidence level。
- AI 审查默认只读；同一审查轮不得为消除 blocker 自动修改算法、调参或筛选结果。需要修复时先输出独立审查报告，再另立实现任务。
- 不向外部网站或公开 AI 服务上传未公开 manuscript、artifact、checkpoint、真实数据或审稿材料；网页检索只使用公开关键词和公开论文。
- 规范、评分、blocker 或审计结论有实质变化时更新实际日期，并同步维护 `docs/project/PROGRESS.md`；长期风险同步维护 `docs/project/BUGS.md`。

## 验证规则

每次代码改动后至少做一个与改动范围匹配的验证：

1. 语法或 import 检查
2. 最小 smoke
3. 目标链路局部验证
4. 全链路 benchmark 或训练验证，必要时执行

常用最小验证：

```bash
python scripts/smoke_test.py
python -m pytest tests/test_env_contract.py
```

真实数据链路的最小验证优先使用：

```bash
python scripts/run_ngsim_sample.py --max_rows 500
python scripts/run_alibaba_sample.py --limit_jobs 3 --min_tasks 5 --max_tasks 20
python scripts/run_real_sample_dryrun.py --mobility_source ngsim --workflow_source alibaba --max_mobility_rows 1500 --max_workflows 3 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_selector max_handoff_candidate --window_length 24 --max_steps 12
```

最终回复必须说明执行过的验证命令、结果和未覆盖风险。

## 禁行方向

- 不在缺少 first-order 问题定位时直接大改架构。
- 不把 smoke 结果当正式论文结论。
- 不单边修改输出字段、路径或 manifest 而不检查消费者。
- 不把历史产物路径继续写成 live 路径。
- 不自动下载或覆盖原始数据；数据准备只做显式检查或按用户要求执行。


