# Workload v0.1 self-consistency and cost-mismatch audit

## 审查元数据

- `reviewed_at`: `2026-10-05`
- `literature_cutoff`: `2026-09-30`（本轮不评价 novelty，不检索或新增文献）
- `target_venue`: `IEEE TMC`（只使用 claim-boundary policy）
- `artifact_run_id`: `measurement_calibrated_vec_workload_20261005_v4`；新增执行待定
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: 审查基线 `f6aa3d7da86a2f2e05a052cf32c33eddf7a7e3e8`
- `evidence_level`: `E2_BOUNDED_ARTIFACT_AND_SOURCE_AUDITED`
- `verdict`: `SELF_CONFIRMING_LOCAL_EXACT_COMPARISON / ORIGINAL_RESULTS_RETAINED / ROBUSTNESS_EXECUTION_PENDING`

本报告只审查 workload v0.1 的决策证据并冻结后续最小验证。原 72 行、失败记录、历史 artifact 和七个用户修改均不覆盖。
本轮不训练、不运行 RL、不读取旧 holdout、不下载资源，也不新增交通任务质量 claim。

## 范围限定只读结论

### 1. local 实际可见信息

`mechanism_aware_local_incremental_cost` 实际读取固定的 prefix compute、input bytes、state bytes、link 和
`restore_cost` 标签。它没有读取逐事件 cache resident state，也没有读取 target-model ready 字段。
`restore_cost` 标签又直接选择执行模拟器中的 low measured 值或 14 s high 值，因此决策侧与执行侧共享同一个实现成本真值。
原文“B 不读取未来实现值”不成立；`local=exact 24/24` 存在自证循环。

### 2. exact 的动作空间、目标与结构

exact 对三个 workflow 的 `restart/recover` 做 `2^3=8` 枚举，目标为完成量最大、deadline violation 最小、总完成时间、
总传输、重算量的字典序。当前动作不改变后续模型需求、cache identity、容量或可行集；每个 workflow 的二选一只改变
该次增量时长、dynamic bytes 和重算量。更短路径还会单调减少后续 serial queue wait。因此主要选择由同一个局部阈值
决定；8 组枚举没有形成真正的跨事件组合优化。exact 只能标为读取实现值的事后 oracle，不能当信息公平在线 baseline。

### 3. seed 的有效输入

seed 17/29/43 只改变三个 arrival jitter 和由 seed 派生的 ID。三组 seed 不改变成本参数、模型/adapter identity、请求顺序、
动作、完成量、传输、重算量或 cache prepare count。按决策有效输入去重后，24 个实例只有 low/high 两类阈值输入；按完整
实现输入计入 arrival 后仍是 8 factor cells × 3 arrival profiles。原结果保留，但三个 seed 不能解释为三组独立成本复测。

### 4. sharing-on/off 语义

- sharing-on：三个 adapter 的 `base_model_id` 都是 `abstract_base_shared`。
- sharing-off：不是“三个互不兼容 base”；实际为 `abstract_base_a → abstract_base_b → abstract_base_a`，A0/A1 仍共享
  `abstract_base_a`。它表达的是两组 family-specific base identity 的 A→B→A 切换。
- 三个 adapter identity 均不同；initial ready 时逐事件加载两个 adapter，missing 时加载三个。adapter 从未共享。
- base 逐事件加载计数符合该 identity：on ready/missing 为 0/1；off ample ready/missing 为 1/2；off tight ready/missing
  为 2/3。原字节账本与这一较窄语义一致，但“sharing-off”必须附带上述兼容关系，不能写成完全禁用任意 base 共享。

### 5. 两次 6-call real run

v1 与 v2 使用相同 plan SHA-256 `8a0d030b…22d1d`、同一模型/adapter/input 和各 6/6 generate。v1 的执行 commit 为
`6c07005e…`，v2 为 `94600ded…`；其间修改了 real runner 的 queue-wait 输出和 workload evaluator 的计时/queue 字段。
因此它们是两个 version-specific technical run，不是同版本独立重复测量。两者均验证输出一致和 suffix-only 执行；
成本只能逐次报告，不能由这两次生成 CI 或稳定性结论。

## 冻结的最小后续验证

### 成本失配矩阵

冻结配置：`configs/experiment/workload_v0_1_cost_mismatch_robustness_v1.json`。共 12 点，local 的事前估计在所有点固定；
执行侧独立使用 measured-version-specific 值或明确标注的 break-even/hypothesis 值。范围只覆盖 model load、恢复/重算、
50/100/200 Mbps 有效速率和 target-ready 正确/假阳性/假阴性。保留 recovery 划算、不划算和边界附近点。exact 重命名为
`post_hoc_exact_oracle`。该矩阵 0 model call，不调参、不增加算法。

### 固定实现独立配对复测

冻结配置：`configs/acceptance/production_action4_independent_repeat_v1.json`。计划 3 次；每次先执行 source，再按
`restart→target`、`target→restart`、`restart→target` 交错两臂。每次调用量为 1+2+1=4，总计 12 generate，硬上限 12，
无自动重试，预计 2--4 分钟。所有 role 为新进程、模型不跨进程驻留；不清空操作系统文件缓存，明确记为未控制。
这只是同机 local-host repeat，不称真实 RSU 网络测量。

## 当前安全结论

- 原 workload 可支持 base identity/capacity 字节账本和技术链正确性边界。
- 原 `local=exact` 不能独立支持局部规则稳健性或方法改进空间；一致性主要由共享真值和可分结构决定。
- 在新增结果完成前，成本误差下的有效区间、退化代价和是否存在方法改进空间均保持 `UNVERIFIED`。
- RL 继续禁止；若简单阈值在独立实现成本下足够，应保留阈值。
