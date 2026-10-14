# 显式候选站与模型/状态工程补齐（2026-10-14）

- 状态：`ENGINEERING_TRAIN_REPLAY_AUDITED`；局部合同与完整 clean train-window 回放已通过。
- `reviewed_at`: 2026-10-14；`literature_cutoff`: 2026-10-14；`target_venue`: IEEE TMC（内部边界）；`policy_version`: `tmc_review_policy_v3_20260621`。
- `artifact_run_id`: `20261014T101637Z_engineering`；运行 commit `1a03fe55b4561bc0ce7a6a78b6d4f13d6d11a378`，`git_dirty=false`；工程证据已审计，论文完整证据仍 `E0_UNAVAILABLE`，无 formal/holdout/support。
- 工程：`/Users/howen/.codex/worktrees/eight-vehicle-simulation/PPO_MEC`，分支 `codex/eight-vehicle-simulation`；Git common dir 为主目录 `/Users/howen/Projects/PPO_MEC/.git`。主目录已有未提交算法修改，本轮没有修改主目录文件或合并这些变更。
- 范围：工程实现与反事实验证，不是 SA/MAPPO 性能实验；formal/holdout 未开启。

## 参考与取舍

一手参考包括 [TMC 2025 的 Dual Dependency-Aware Collaborative Service Caching and Task Offloading](https://ieeexplore.ieee.org/document/11014496/)，其摘要覆盖依赖感知服务需求、分层缓存与 PPO 协同卸载；[VEC-Sim 作者全文](https://arxiv.org/html/2410.06934v1)用于检查通信、计算和服务缓存的仿真分层；[MAPPO 官方项目](https://github.com/marlbenchmark/on-policy)用于动作可用性、actor/critic 和复现预算边界；[S-LoRA 官方项目](https://github.com/S-LoRA/S-LoRA)用于 base/adapter/执行状态资源分层。S-LoRA 仓库已归档，只作为方法参考。本轮未复制外部代码、安装推理依赖、下载数据/模型或宣称复现其性能；参考补充已登记到 `literature_reference_table.md`。

v3 优先解决上一轮的可控性缺口：车辆在覆盖内可有多个 RSU，而旧五动作无法选择执行站；预取只面向当前 adapter，执行与结果送达也未独立计账。新接口具有独立版本，避免用旧五动作网络直接执行不同含义的动作。

## 已实现的共同工程合同

`MultiRequestTargetControlEnv` 使用 `JointRequestControl`：

- `execution_target`：当前覆盖内 RSU、本车或等待；车端执行要求声明兼容 base 与 adapter。
- `cache_target/cache_node_id/cache_kind`：为已知 DAG 的任意节点显式准备 base-only 或 adapter+base 依赖，可预置未来节点但不能越过父依赖执行。
- `evict_object_ids`：验证有序驱逐，禁止 orphan、pinned、重复或不存在的对象；完整联合驱逐在任何缓存执行前验证。没有足够空间时的其余 admission 仍用公共依赖安全 LRU，尚不是已训练的缓存策略。
- `prepare_target`：版本化前缀与执行进度预复制；旧副本失效、完成帧收费。已算完的待返回结果也能经付费状态转发到新站再下传。
- `priority`：0..3 的明确资源仲裁优先级；同优先级采用确定性轮转。该头同时影响同帧请求处理顺序，并非已实现多个更新周期的分层 RL。

每个节点在 RSU 执行前上传声明的输入字节；迁移后目标站需要自己的输入就绪记录。计算完成后输出进入待传输状态，送达车辆才计节点/工作流完成。输入、结果、模型和状态争用同一 RSU 数据链路槽，逐帧收据记录费用、字节、模型事件与每站槽使用量。移出原站覆盖后，可以在新覆盖站恢复当前版本与待返回输出，继续回传；没有可达站时等待，不计成功。

`verify_joint_target_control.py` 从现有 v2 开发 manifest 校验原始数据 SHA-256、重建相同 train/dev 时间窗，输出两种相同物理条件的工程规则、初始/完成 manifest、结果和逐帧事件。balanced 是当前请求协调规则，未使用未来轨迹，不能称为学习算法或论文基线。source identity 变化会在建立输出前拒绝；未完成运行的 manifest 不标 complete。

缓存 PPO 新模型使用真实 typed 字节占用比，包含 base/adapter 驻留和在途预留。checkpoint 增加 `cache_feature_contract`；历史缺字段的 checkpoint 加载时恢复旧槽位特征以保持历史语义。第一方缓存 PPO 元数据不再标成外部论文独立复现。预测器没有 cache_demand 时仍明确记录缺失来源，不伪造需求预测。

## 验证与尚未覆盖的风险

局部命令见 `RUNBOOK.md`：环境、入口、cache、v2、launcher、算法池和 checkpoint 的联合目标回归 **316 passed**，另有 `py_compile` 与 `git diff --check`。反事实测试验证目标站分配改变完成时间/奖励，优先级改变请求进度，base/未来 adapter 与手动驱逐影响驻留，pre-copy 改变到站状态费用，输入/输出完成帧收费和跨站输出转发。v2 学习策略对 v3 明确拒绝。验收后另补 `output_source_rsu_id` 事件字段，明确等待控制下自动回传及跨站回传的实际来源；该字段不改变物理行为，也不改写已产生的 raw run。

完整 20 个训练窗口的 clean run 已完成：40 条结果、27,866 条事件、4,511 个帧收据通过 `independent_engineering_audit.json` 的 10 类检查，包括从原始轨迹重建覆盖、未舍入奖励守恒、节点/请求计数、I/O/state/model 总量、每站槽位和缓存容量、配对成本配置完全相同。全量原件位于 `artifacts/joint_target_control_engineering/20261014T101637Z_engineering/`，审计程序位于 `artifacts/joint_target_control_engineering_audit_20261014/audit.py`。

| 规则 | 平均奖励 | 完整请求 / 8 | 完成节点 / 70 | 平均步数 | 模型链路帧 | 状态链路帧 |
|---|---:|---:|---:|---:|---:|---:|
| 关联站反应式 | 54.533118 | 6.20 | 64.45 | 118.70 | 22.35 | 7.90 |
| 可达站固定分配反应式 | 60.293977 | 7.65 | 69.35 | 106.85 | 24.75 | 4.30 |

20 个配对训练窗口有 19 个奖励提高、1 个下降；完成请求数有 12 个提高、8 个相同。负向窗口 offset 2280 的奖励差为 −0.208805，完成数相同，原件保留。平均奖励差 +5.760859；提高伴随更多模型链路开销，未声明全指标优势。两规则提前状态准备均为 0，不能据此归因到预测/pre-copy。它们仅证明新动作合同能改变结果和释放控制空间，不是 SA/MAPPO 学习优势，且 v3 新增 I/O 成本后不能与旧 v2 reward 直接比较。

两个真实输入训练窗口的开发见证位于 `artifacts/joint_target_control_engineering/20261014T100001Z_engineering/`，运行时工作树包含开发修改，不能作为冻结证据。关联规则均值 54.249684、5.5/8，可达分配规则均值 60.007326、7.5/8；其含新增 I/O 费用的 v3 reward 与旧 v2 不可直接比较。首次入口因旧 manifest 的 nested `costs.replay_costs` 解析失败，已修复并加入生产/消费端测试，失败目录保留。

主要剩余项是参数化 SA/MAPPO/缓存 PPO actor heads、joint mask/log-prob 和 checkpoint 适配；学习型预测/surrogate；控制面消息与数据面资源竞争；设备实测参数、车辆缓存容量、有限状态存储、动态无线信道、并行 DAG 与外生请求到达。当前输入/输出和状态包仍为 synthetic profile byte 代理，destination-link 模型不是实测无线/双向 backhaul 网络。没有 formal/holdout 或主方法最高奖励证据。
