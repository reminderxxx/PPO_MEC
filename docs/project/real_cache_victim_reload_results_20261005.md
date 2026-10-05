# 真实缓存 victim→reload 最小实测结果

## 审查身份与结论

- `reviewed_at`: `2026-10-05`
- `literature_cutoff`: `2026-10-05`（本轮未重新评价 novelty）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `real_cache_victim_reload_20261005_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- 解析起点：`54b5cfb754cf5c3b3191172704f6e74df3f90ec9`
- 固定执行提交：`f31024d957bd07718c4087fa55d8f7bb707e0f6e`
- `evidence_level`: `E2_ARTIFACT_AUDITED`（bounded same-host real-adapter lifecycle；网络仍为假设）
- `verdict`: `REAL_VICTIM_RELOAD_VERIFIED / DECISION_EXTERNALITY_NOT_DIFFERENTIAL / NOT_PAPER_READY`

唯一一次 clean execution 在 `64.309679 s` 内完成 `12/12` 次 `generate`，无训练、下载、old holdout、自动重试或
追加实例。两个 condition 的 restart/recovery 完成量、n0、n1 输入/token、n2 输入/token 与 native cache event 均严格一致；
22 个 manifest 文件独立复算 `22/22` size/SHA-256 通过。

真实 victim 条件确认完整事件链：logical resident `base+ALPR` → native legal victim `adapter:alpr` → PEFT 中 ALPR
`780 tensors / 154,308,608 B` 变为未注册且 0 tensor → 从既有本地文件加载 Helmet `520 tensors / 9,568,256 B` →
执行 n1 → 后续 n2 请求 ALPR，合法驱逐 Helmet并使其 `520→0` tensor → 实际 `load_adapter` 重载 ALPR 为
`780 tensors / 154,308,608 B` → 执行 n2。restart 与 recovery 都使用同一 lifecycle；无驱逐对照在 action window
没有任何 load/unload。

两条件事前和事后都选择 recovery，没有边界翻转。合法 victim→reload 因对两臂相同，直接 lifecycle 成本分别为 restart
`1.022245 s`、recovery `1.026997 s`，在动作差值中应自然抵消，不能只向 recovery 计费。该结果验证真实机制存在，但没有
证明 eviction externality 改变 restart/recovery 的相对选择。

## 双方全路径与模型生命周期

每个 condition 中，processor、共享 base 和初始 resident adapter 都在每个 arm 的 action window 前真实准备并单独记录。
只有对象、时点和调用相同的这部分 setup 从比较中排除；不是因为正文称其为“共同成本”就假定抵消。

| 路径 | action 前模型 | action 内当前请求 | 计算 | 后续请求 | 动态输入 |
|---|---|---|---|---|---:|
| restart / victim | base+ALPR | 卸载 ALPR、加载 Helmet | n0+n1 | 卸载 Helmet、重载 ALPR、执行 n2 | 原图 192,757 B |
| recovery / victim | base+ALPR | 卸载 ALPR、加载 Helmet、校验状态 | n1 | 卸载 Helmet、重载 ALPR、执行 n2 | 状态包 2,195 B |
| restart / control | base+ALPR+Helmet | Helmet hit | n0+n1 | ALPR hit、执行 n2 | 原图 192,757 B |
| recovery / control | base+ALPR+Helmet | Helmet hit、校验状态 | n1 | ALPR hit、执行 n2 | 状态包 2,195 B |

共享 base 权重为 `1,015,025,832 B`，Helmet/ALPR adapter 权重为 `9,641,944/154,423,432 B`。victim 容量精确为
`base+ALPR=1,169,449,264 B`；control 容量为 `base+ALPR+Helmet=1,179,091,208 B`。容量来自真实权重，不含 dummy
padding。base 在 action 中始终依赖安全且保持 runtime loaded；两臂每个 action 都有 2 次 adapter load、2 次 unload，
读取本地 adapter 权重共 `164,065,376 B`。control action load/unload 均为 0。

setup 也没有隐藏：victim restart/recovery 分别记录 processor `0.074476/0.075023 s`、base
`0.265429/0.297845 s`、初始 ALPR `0.795693/0.816355 s`。control setup 另加载 Helmet，四个 setup adapter load
原始值全部保存在 arm receipt 中。所有静态权重在动作前后 size/hash 相同；磁盘文件没有删除。OS 文件缓存未清空且不可控，
因此这些是 warm-OS-cache 可能存在的 local-file load，不是 disk-cold 或无线加载。

完整 6 个 scientific processes 各调用一次 base load，共 6 次；adapter load 共 12 次：两个 source 各一次 Helmet，
victim 两臂各一次 setup ALPR 加两次 action load，control 两臂各两次 setup adapter load。真正由 cache event 触发的
action-window adapter load 为 4 次，其中后续 ALPR reload 为 2 次。没有 action-window base reload。

## 逐事件卸载、加载与重载

| arm | 当前 ALPR unload | 当前 Helmet load | 后续 Helmet unload | 后续 ALPR reload | action lifecycle 总计 |
|---|---:|---:|---:|---:|---:|
| restart | 0.012657 s | 0.284485 s | 0.012192 s | 0.712910 s | 1.022245 s |
| recovery | 0.012693 s | 0.292683 s | 0.012181 s | 0.709439 s | 1.026997 s |

每次 unload 的 object registration 与对应 tensor count/bytes 都降为零，每次 load 后都恢复到固定 count/bytes；这比 RSS
变化更直接。native ledger 在 runtime 成功后才提交，且 preview/commit 的 transaction status、victim 与 admission 完全一致。
无驱逐对照的两条请求在两臂均为 `noop_all_resident`，runtime event list 长度为 0。

负例先于科学执行检查：base/dependency victim 被 `adapter-only` bridge 拒绝，missing adapter 被 native catalog 拒绝，
拒绝前后 logical resident 保持 `base+ALPR`。局部合同另覆盖未映射非法 victim；production action-4 的既有测试继续覆盖
缺目标模型时不转移执行权、不推进节点。失败不会提前写成功 logical commit。

## 完成、保真、成本和预测误差

restart 调用 `n0/n1/n2=1/1/1`，recovery 为 `0/1/1`；source 每 condition 仅执行一次 n0。不存在重复或遗漏节点。
两 condition 均满足 source/restart n0 text+token 完全相同、recovery 消费 source n0、restart/recovery 的 n1 和 n2
输入对象完全相同且输出 token 完全相同。技术图像没有适用任务标签，`task_correctness=unavailable`。

下表 `measured` 为 local action wall，加 recovery state serialize/save，再加同一冻结模拟动态网络项；网络不是实测。

| condition | arm | 预测 s | action wall s | prefix s | n1 / n2 s | load / unload s | scored 实测 s | 预测-实测 s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| victim→reload | restart | 19.553682 | 11.259267 | 7.021436 | 0.140676 / 2.882903 | 0.997395 / 0.024849 | 11.294688 | +8.258994 |
| victim→reload | recovery | 8.509755 | 4.135223 | 0 | 0.143976 / 2.873619 | 1.002123 / 0.024874 | 4.156329 | +4.353426 |
| no eviction | restart | 18.684786 | 10.557728 | 7.299024 | 0.137762 / 2.911111 | 0 / 0 | 10.593149 | +8.091637 |
| no eviction | recovery | 7.640859 | 3.152671 | 0 | 0.154113 / 2.877928 | 0 / 0 | 3.173724 | +4.467135 |

state package 是 `2,195 B`，原输入是 `192,757 B`；对应模拟网络项为 `0.020176/0.035421 s`。真实网络、模型远端
传输、queue、丢包和重传全部 unavailable。本地 adapter load bytes 与动态 state/input bytes 分列，不相加冒充无线实测。

冻结的 victim lifecycle 增量估计是 `0.868896 s`，两臂直接实测为 `1.022245/1.026997 s`，分别低估
`0.153349/0.158101 s`。victim 相比 control 的完整 scored wall 差为 restart `+0.701539 s`、recovery
`+0.982605 s`；单次顺序执行还混有 inference 波动，不能把该差值全部归因于 load。两条件的预测和实测较优动作均为
recovery，未跨边界且不追加实例。

## 对论文主张的影响

- **支持**：存在可追溯的 native legal adapter victim→PEFT runtime object removal→later local-file reload→future node
  execution；模型文件保留、共享 base 不重载、logical/runtime/disk/OS-cache 四层明确分开。
- **支持但收窄**：真实 cache lifecycle 可以加入中心机制实验；本轮仅证明机制兑现和约 1.02 s 的同机 lifecycle 成本。
- **削弱/删除**：删除“驱逐外部代价只属于 recovery”或“真实 victim 已改变正确动作”的暗示。两臂在同一节点依赖下都支付
  同一 lifecycle；正确比较自然抵消该项，旧不对称公式只能保留为 frozen development method。
- **不支持**：原创缓存算法、优于正确两步前瞻、决策边界校准、无线收益、交通任务正确性、统计泛化、paper-ready。

## 顶刊政策摘要与下一步取舍

Evidence inventory：frozen config/plan、clean execution commit、6 个独立 scientific process receipts、2 个 condition
receipt、raw measurements、terminal、negative checks 与 22-file integrity manifest 齐全。formal/holdout、真实无线/queue、
任务标签和多实例统计不存在，因此完整投稿判断仍不是本轮目标。

最小取舍不是继续堆同侧重复：若论文中心主张仅为“真实生命周期连接与成本可追溯”，本轮已经闭环；若主张必须是“缓存外部
代价改变 recovery 决策”，则需要一个语义上确实让两条合法路径产生不同 resident 转移、且事前冻结的真实系统场景。本轮结果
表明不能靠对同一当前/未来节点依赖进行不对称计费制造该差异，也不应扩建多租户平台或重启 RL。

## Artifact inventory

- `artifacts/analysis/real_cache_victim_reload_20261005_v1/terminal_receipt.json`
- `artifacts/analysis/real_cache_victim_reload_20261005_v1/all_measurements.json`
- `artifacts/analysis/real_cache_victim_reload_20261005_v1/preflight_negative_checks.json`
- `artifacts/analysis/real_cache_victim_reload_20261005_v1/{adapter_victim_reload,no_eviction_control}/`
- `artifacts/analysis/real_cache_victim_reload_20261005_v1/integrity_manifest.json`
