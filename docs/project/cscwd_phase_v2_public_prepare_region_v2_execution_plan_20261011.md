# CSCWD phase-v2 公开 prepare 区域诊断：兼容修复后独立执行冻结

- `frozen_at`: 2026-10-11，v2 科学 `env.step` 前。
- 新 `run_id`: `cscwd_phase_v2_public_prepare_regions_20261011_v2`；全新 create-only root `artifacts/analysis/cscwd_phase_v2_public_prepare_regions_20261011_v2/`。
- 原失败 `v1` 的 `failure_receipt.json` SHA-256 为 `55b826571c4b2474bf1f52ab228892d37855a850425ce20249b942100127e1ad`，`selection_inventory.json` SHA-256 为 `875163d7022ce209746c1e7dd954891d12da2f92f640cac7e815a0604a2d54cb`；不覆盖、不补记原 root。
- 科学规则仍以 `cscwd_phase_v2_public_prepare_region_plan_20261011.md` 为准，SHA-256 `472808fc90a3af16f8fe2c4f76acb69a9921a70f50d02184baa1eb40ec9c1def`。新机器 manifest 为 `configs/experiment/cscwd_phase_v2_public_prepare_regions_v2_manifest.json`，绑定 Python 3.9 修复 commit `54968953b2bd67551b0a934e60019289732d3034`；执行时启动回执再记录实际干净 Git commit 与 manifest SHA。

## 唯一变更与启动门

原 v1 因 Python 3.9.6 对 `zip(strict=True)` 不兼容，在首个环境步前停止。v2 仅将该配对改为显式长度检查与普通 `zip`，增加新 run 身份、原件/manifest 身份校验和实际路径的 pre-step 配对校验；结果盲选起点、24 格、11 个 US101 起点（3 已暴露 `[4]` 回归、8 同轨迹新覆盖）、27 个去重动作分支、动作3/4/原策略臂、统一公开 continuation、模型/状态/输入/backhaul/partial 成本、区域判据与 **12 起点/36 分支/288 `env.step`** 硬限均不变。训练、模型前向、formal/holdout、下载和在线策略/奖励修改均为零。

先在 Python 3.9.6 实际解释器中完成语法、局部测试、只读输入身份与真实行为 CSV 的 24 格/11 起点重建；对每个入选起点调用执行器的 `prefix_history_pairs`，确认 `[4]` 等非空前缀能到达原错误位置，清单与 v1 原件逐项一致。这一 preflight 不调用 `env.step`，不得把它称为效果结果。需检查无科学进程、目标 root 不存在、代码/方案/manifest 已提交且工作树干净。原方案只要求冻结干净 commit；GitHub `push` 按仓库规则尝试并记录远端状态，网络失败不改写本地执行身份或声称已同步。

上述门全部通过后，仅从新 run ID 单次执行；任何身份、预算、合法性、脚本或输出错误即保留失败回执并停止，不自动重试、换 root、增 seed、改 prefix 或扩矩阵。原三 `[4]` 回归与八个新状态分列；全部新状态仍为同一 US101 相关开发轨迹，不是新的独立数据。按旧判据完整报告正区域、无益区域、服务损伤、混合/截断和逐项成本，并审查在线仅可见公开字段；没有正例必须如实记录，不按结果改参数。论文贡献、优秀基线排名与 paper-ready 在缺 formal/holdout/support 前仍 `Unverifiable`。
