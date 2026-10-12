# 四 RSU、八车竞争：环境工程与论文完整性审查

- `reviewed_at`: 2026-10-12
- `literature_cutoff`: 2026-10-12
- `target_venue`: IEEE Transactions on Mobile Computing (TMC)
- `artifact_run_id`: `vec_competition_v1_20261012_4fd8549`（工程 replay）；正式 canonical run **不存在**
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `4fd8549`（已推送分支 `codex/vec-final-environment`）
- `evidence_level`: 工程 replay 可复查；按正式论文关键 artifact 要求为 `E0_UNAVAILABLE`
- `verdict`: `Unverifiable`（论文性能/创新实证）。新回放不是 paper-ready。

证据：`artifacts/analysis/vec_competition_v1_20261012_4fd8549/` 的 `scenario_manifest.json`、`summary.json`、`event_ledger.jsonl`；旧正式缺陷见 `BUGS.md`、`cscwd_2027_baseline_contribution_audit_20261009.md`、`mechanism_evidence_closure_results_20261005.md`。本审查不更改旧实验或算法。

## 已实现并可核验的工程能力

同一 Peachtree 源段的 120 个连续 100 ms 帧、八个一直在场的观测车辆，八个不同 Alibaba DAG job；四个推定布设 RSU。场景记录 NGSIM 坐标/速度单位转换、所选车辆、`Section_ID/Lane_ID/Direction` 的观测转移图、输入 SHA-256、catalog 指纹与全部受控资源参数。统一时钟中八车竞争缓存容量、无线/回传吞吐和每站一个计算槽；模型加载需要实际推进的时间，前驱收据跨站传送后才可运行后继，最终结果通过下行才计为交付。受控单窗回放记录 70/70 节点完成、8/8 交付、17 次 handoff、10 次 state receipt transfer、6 次模型对象加载。**这些数值只证明该场景下工程链路可执行，不表示真实系统时延或算法收益。**

## 缺陷总表

| 领域 | 当前状态 | 未达到最终论文工程态的原因 / 必须补足的证据 |
|---|---|---|
| 真实轨迹与时间 | 新链路闭合单源、同窗、八车、单位与 ID；旧正式链未修复 | 新场景只有一个按位移择优的 12 秒窗口。旧 v70 frozen plan 与实际源段错配；formal/holdout 原始区间互斥仍须重建。 |
| 真实路网 | 仅八车观测车道/区段转移图 | 不是可地图匹配的完整物理拓扑；缺真实道路底图、坐标参考、道路边、转向/限行以及 RSU 对道路的覆盖关系。LuST 是仿真网，不能冒称 NGSIM 测量路网。 |
| 四 RSU | 数量、局部坐标和容量显式 | 四站沿轨迹主轴推定；缺真实站址、天线/高度、覆盖测量、站间回传图与部署可行性。 |
| 八车竞争 | 新内核同时竞争并有共享槽/链路 | 旧正式训练和 benchmark 仍是单 workflow/序列化样本；八车环境尚未被全部策略、checkpoint、评估器消费。 |
| DAG 数据与 AI 任务 | Alibaba 真实 DAG 拓扑，八个不同 job | Alibaba batch task 并非车载 AI 推理；任务与车辆是人工配对，I/O 大小用受控换算；缺真实输入、前驱中间结果、正确性标签及到达联合分布。 |
| typed base/adapter cache | 容量、版本/兼容、驻留和非零加载传输进入新内核 | catalog 标记 `controlled_validation_only`；没有真实权重、host/GPU 层级、加载/切换测量、cache 命中对真实推理的效应。旧 native typed cache 与新内核未统一成正式动作合同。 |
| 无线与回传 | 每站有限吞吐，八车等分争用；跨站传输占源/目标回传端 | 无真实无线信道、SINR、干扰、重传、车速影响或链路可靠性；站间采用抽象直达，不含实测回传路径/拓扑。受控 MB/s 不能称观测。 |
| 计算与队列 | 单站有限 compute slots、共享排队、节点计算时间进入时钟 | `compute_mb_per_s` 是受控换算；没有 GPU/CPU profile、batching、warmup、模型复杂度或能耗测量。 |
| handoff 与状态 | 旧站运行中任务可完成；新站后继等待收据跨站到达；下行交付单独记账 | 新收据只携带对象身份和受控大小，不是可恢复的应用运行状态/KV/tensor；旧 production action 4 仍只 prepare，独立两节点技术见证未接正式环境。 |
| 动作、观测、奖励 | `prepare(vehicle,target,node)` 可选未来节点，内核自动放置就绪节点 | 没有冻结新的 4×8 多主体 observation/action/reward，缺学习控制器、合法动作 mask、预测信息隔离、慢/快时间尺度更新。不能把旧 PPO/MAPPO/SA 结果直接迁移到新环境。 |
| baseline 与公平性 | 新场景只有工程默认规则，无方法比较 | 旧 SA/DT/PPO 更新预算和 checkpoint 选择不同；新环境需要同输入、同预算、同选择规则的强基线与消融。 |
| 划分、统计、artifact | 新单窗有 manifest/hash/账本；正式链缺失 | 旧 mixed/full 原始区间重叠；新单窗没有独立 formal/holdout/support、多 seed、window-outer 统计、正式 checkpoint、command log 和模型正确性原始结果。不能评分或晋级 canonical。 |
| 扩展与稳健性 | 只做一个 4×8 / 120 帧配置 | 缺 2/4/8 RSU、1/4/8/16 车辆、不同容量/带宽/预测误差、不同道路段/时段的预注册 stress，缺 wall-clock/内存和失败边界分析。 |

## 最近邻与 claim 边界

已有 VEC dependency-aware migration、service cache/offloading、DAG 与 mobility/MARL 组合，不能宣称这些组合本身空白。POLAR 直接研究 adapter caching/routing 的双时间尺度，S-LoRA、Torpor 和 FAST 2026 表明模型/adapter 加载、GPU 共享及 I/O 延迟需要实测或诚实的受控标记；Strata 的 KV cache 与本项目 adapter 权重不可等同。详见 `literature_reference_table.md`。当前可安全表述为“构建了一个使用实测车辆轨迹及真实 DAG 拓扑、四个推定 RSU 和八车竞争的可回放工程环境”；禁止表述为“真实 RSU/真实 AI 工作流/真实模型状态迁移”“PPO 优于强基线”“paper-ready”。

## 复审前必要条件

先补可核验道路与站点、真实 AI 任务和兼容权重的测量，再冻结多主体训练与评估合同；随后在互斥原始时间区间上以同预算重训强基线、完成机制消融及独立 formal/holdout，发布完整 checkpoint、manifest、命令和原始结果。没有这些输入，补代码或继续扩大受控仿真都不能关闭论文阻断项。

评分：七维均为 `N/S (not scored)`；缺正式 artifact、独立划分与公平 baseline，按审查政策优先 `Unverifiable`，不打伪精确分数。TITS/TVT 亦不能用降低证据要求替代上述闭环。
