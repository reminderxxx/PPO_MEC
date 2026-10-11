# CSCWD phase-v2 公开 prepare 区域 v3：待授权执行方案

`status`: **AUTHORIZED_PENDING_SINGLE_EXECUTION**（2026-10-11 用户另行授权；执行状态以新 root 回执为准）。科学选择、比较臂、公开 continuation、指标和成本仍按原 `cscwd_phase_v2_public_prepare_region_plan_20261011.md`；工程身份与持久化细则见 `cscwd_phase_v2_public_prepare_region_v3_engineering_design_20261011.md`。机器授权为 `configs/experiment/cscwd_phase_v2_public_prepare_regions_v3_authorization.json`，绑定执行器基础 commit `2705fa49a92ff79b83f7a09fc750221fdfe2ebaf`、旧方案与两失败回执。

固定候选全集仍为 24 格，按原公开字段预期 11 个 US101 入选格、13 个原理由排除格。v3 不删除重复公开观测：所有 11 格都进入身份表，同一 `public_observation_sha256` 形成部分可观测组；仅 `full_simulation_sha256` 完全相同的格被标为代表的 alias，保留原格与原动作标签，由代表执行该 full-state 组所有所需首动作的并集。**尚未知**代表 full state 数、同公开观测但不同 full state 的组、alias 数、实际比较分支数、正/负区域效果；不得将 v2 的 77 步当作这些量的测定。

执行前冻结并提交仓库内授权 manifest，写入 v3 run ID、原方案/两失败回执/选择 SHA、288 步含 prefix 回放硬限、state identity 版本和明确的单次授权；脚本只接受干净提交、精确 `artifacts/analysis/cscwd_phase_v2_public_prepare_regions_20261011_v3/` create-only root。先核对公开 CSV 选择，再在真实环境回放 11 格合法 prefix，并保存每格身份/前缀事件；这最多 8 个 `env.step`，**计入**同一 288 步预算。全部身份登记和碰撞分类完成前不得执行任何动作3/4/原策略比较分支。

之后只对不同 full state 代表执行原定首动作并集及最多 7 步公开 continuation，最多 36 分支、总计最多 288 个真实环境步。每完成一条分支即原子保存输入身份、事件、指标和 SHA；失败回执必须列可读回分支、未完成分支和 `PARTIAL_UNVERIFIABLE`，不自动重试或把部分数据当正式结论。完整执行后从落盘文件复算逐格和按独立 full state 的区域表，三旧回归格与八新覆盖格分列，并报告同公开组与 alias 关系、逐项字节/耗时/重算及服务结果。全部仍属于同一 US101 开发轨迹，不支持独立确认、优秀基线排名或论文贡献。不得读取 formal/holdout、训练或修改生产策略。
