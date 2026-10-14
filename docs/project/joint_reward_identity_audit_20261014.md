# 三算法奖励完全相同的独立核查（2026-10-14）

- `reviewed_at`: 2026-10-14（Asia/Shanghai）
- `literature_cutoff`: 2026-10-14；本轮为实现/原件审计，未重新评价 novelty
- `target_venue`: IEEE TMC（内部证据门槛）
- `artifact_run_id`: `20261014T074430Z_development`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: 原运行 `6d66ec5ca8191587b48a846eb70ac1aa8046155d`；独立回放代码 `031ada4`（两者仿真与算法源码一致，后者只追加文档）
- `evidence_level`: 开发结果和执行轨迹已独立重放；完整论文证据仍为 `E0_UNAVAILABLE`，缺 formal/holdout/support
- `verdict`: 同分可复现；该开发比较对算法奖励优势缺乏辨识力，优劣与收敛结论 `Unverifiable`

## 检查结果

证据位于 `artifacts/joint_reward_identity_audit_20261014/audit_identity.json`，核查程序为同目录 `audit_identity.py`；只读原 checkpoint 和原 train/dev 合同，没有训练、调参或开启 formal/holdout。

1. **模型身份与训练更新**：9 个最终网络的 tensor SHA-256 全部不同；SHA 覆盖 tensor 名称、形状和数值，排除仅 checkpoint 元数据不同的假区别。SA/MAPPO/缓存 PPO 的网络参数分别为 169,096 / 42,952 / 48,584。9 个模型从 24 到 96 回合均改变参数，最大单参数变化为约 0.297–0.397，最终 update count 均为 24。未发现共用同一权重或训练没有更新。
2. **实际加载**：分别构建对应 registry agent 并加载其 checkpoint；9 个评估网络的完整 tensor SHA-256 均与各自文件严格相同，排除 loader 未恢复权重。
3. **逐窗、逐帧及未舍入回报**：8 个窗口中，每窗的 9 次原 greedy 评估均有相同物理轨迹哈希；哈希覆盖车、节点、服务站、执行状态、工作进度、状态版本及完成标记。每窗从帧收据重建的未舍入回报也完全相同，因此不是平均值或六位小数舍入造成的假同分。动作字段确实不同，不同 actor 不是同一动作文件的简单副本。
4. **独立全量回放**：从各自最终 checkpoint 重新运行 3 算法 × 3 seed × 8 dev 窗口，共 72 次 greedy 评估，所有奖励与上述物理轨迹哈希严格重现原结果。源码的 `_episode()` 每次创建新环境、实际调用 agent 和 `step_actions()`，未发现复用同一结果的路径。

## 反事实动作检查

第一开发窗口 offset 2760，从相同冷缓存起点重置后执行固定规则：

| 固定规则 | 未舍入奖励 | 完成节点 | 完整请求 | 动作次数 |
|---|---:|---:|---:|---|
| 始终 action 0 | 41.92351660156253 | 51 | 2/8 | 0：855 |
| 冷态 action 0、热态 action 3 | 41.92351660156253 | 51 | 2/8 | 0：32，3：823 |
| 始终 action 3，诊断性关闭 cold mask | −9.936679687499982 | 0 | 0/8 | 3：960 |

最后一行是明确改变合法域的负对照，不用于公平算法排名；它证明冷态下不拉取模型确实会改变环境与奖励，`step_actions()` 没有整体忽略动作。前两行在原物理配置下同分，直接证明本窗口热态的 action 0/3 效果退化为相同执行路径。此等价只对已检查轨迹成立：action 0 还会 touch LRU，不能未经检查宣称在所有缓存竞争场景中都等价。

## 对上一轮解释的修正

容量不足解释低完成率，不能单独解释不同策略逐帧完全相同。精确同分的直接证据是：cold mask 强制避开缺模型的 action 3；已训练策略没有使用 action 1/4；车端 action 2 缺预载 adapter 而不可用；热态主要选择已证实在本批窗口上产生同一结果的 action 0/3。其余计算由关联站、每帧固定容量和按车辆顺序轮转的调度决定，策略没有执行站或计算调度的选择权。

因此完整性检查通过不足以使这个实验成为有效算法比较。原始数值保留，可用于合同诊断；它不能支持三算法性能相当、SA 最高奖励或训练已收敛。下一实现任务应给所有算法可达候选站执行、明确模型对象 admission/预置与状态迁移控制，并在固定规则和反事实动作之间先证明不同决策能产生可计账的结果差异。设计入口为 `joint_control_v3_design_20261014.md`。

**Hard blockers**：奖励对实际采用的策略动作缺乏辨识力，formal/holdout/support 不全。**Major concerns**：固定排队、稀少迁移机会、单 base 输入、未校准成本和缓存方向基线特征错配。**Scorecard**：七维 `N/S`。**Safe claims**：本轮同分及动作反事实可复现。**Prohibited claims**：本轮同分证明一般算法等价或 SA 优势。**Required actions**：先实现共同可控合同并验证动作敏感性，再同预算重训；不在本审查轮修改算法。**TITS/TVT fit note**：未评价。
