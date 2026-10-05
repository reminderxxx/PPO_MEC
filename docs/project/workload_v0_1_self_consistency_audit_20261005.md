# Workload v0.1 self-consistency and cost-mismatch audit

## 审查元数据

- `reviewed_at`: `2026-10-05`
- `literature_cutoff`: `2026-09-30`（本轮不评价 novelty，不检索或新增文献）
- `target_venue`: `IEEE TMC`（只使用 claim-boundary policy）
- `artifact_run_id`: `measurement_calibrated_vec_workload_20261005_v4` +
  `workload_v0_1_cost_mismatch_robustness_20261005_v1` +
  `production_action4_independent_repeat_20261005_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: 审查基线 `f6aa3d7da86a2f2e05a052cf32c33eddf7a7e3e8`；robustness
  `3bf08e6bd2e430212181438bf53839f953dcabe3`；固定实现复测
  `4a6d670605b624eb6aac4fbaf3c05868ab11c8c4`
- `evidence_level`: `E2_BOUNDED_ARTIFACT_AND_SOURCE_AUDITED`
- `verdict`: `SELF_CONFIRMING_ORIGINAL_COMPARISON / SIMPLE_THRESHOLD_CONDITIONALLY_SUFFICIENT / NO_RL`

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

## 已执行结果

### 12 点成本失配

执行 commit `3bf08e6…`，36/36 方法行，0 model call，wall 0.0066 s。第一次手工命令写错 expected full SHA，
在 output 创建前 fail-fast，调用和结果为 0；随后以 Git 实际 SHA 一次性执行冻结矩阵。所有方法均完成 3/3 workflow、
6/6 nodes，service failure=0。local 的事前估计固定，全部 12 点均选择三次 recovery；事后 oracle 在 8 点 recovery、
4 点 restart。完整逐行结果位于
`artifacts/analysis/workload_v0_1_cost_mismatch_robustness_20261005_v1/all_method_results.json`。

| 点 | 实现条件 | local / oracle | local 总完成时间代价 | makespan 代价 | 传输 / 重算交换 |
|---|---|---|---:|---:|---:|
| r01--r05 | 两次旧实测低成本、ready/missing 信息正确或错误 | recover / recover | 0 | 0 | 0 |
| r06 | 11.05 s restore, 50 Mbps | recover / recover | 0 | 0 | 0 |
| r07 | 11.05 s restore, 100 Mbps | recover / recover | 0 | 0 | 0 |
| r08 | 11.05 s restore, 200 Mbps | recover / restart | +0.042439 s | +0.021219 s | −571,716 B / −33.105912 s |
| r09 | 11.00 s restore, 100 Mbps | recover / recover | 0 | 0 | 0 |
| r10 | 11.10 s restore, 100 Mbps | recover / restart | +0.296701 s | +0.148351 s | −571,716 B / −33.105912 s |
| r11 | 14.00 s restore, 100 Mbps | recover / restart | +17.696701 s | +8.848351 s | −571,716 B / −33.105912 s |
| r12 | 10.80 s recompute, 11.00 s restore | recover / restart | +1.108525 s | +0.554263 s | −571,716 B / −32.400000 s |

负传输/重算“代价”表示 local recovery 仍少传状态字节、避免重算，但因字典序先最小化完成时间而选错。四个失配点的
deadline violation 与 oracle 相同；没有用 deadline 制造差距。target-ready 假阳性/假阴性改变绝对 model-prepare 时间和
deadline，但在当前两臂都必须准备同一目标模型的合同下是共同成本，不改变二选一。

决策开销 mean/max：current `0.118/0.208 µs`，local `0.142/0.292 µs`，事后 oracle
`76.524/90.958 µs`。oracle 读取实现值，只用于计算选错代价。

阈值为
`recovery_overhead < prefix_recompute + 8×(input_bytes-state_bytes)/effective_rate`；相同 fixed latency 在两臂抵消。
以 192,757/2,185 B、prefix 11.035304 s 为例，50/100/200 Mbps 的 break-even 分别是
11.065796/11.050550/11.042927 s。只要估计和实现的 margin 在阈值同侧，简单规则仍有效；当恢复/重算误差或链路速率
把实现 margin 推到另一侧时退化。模型加载误差和 ready 信息在当前合同下不会改善动作判别。

### 三次固定实现配对复测

执行 commit `4a6d670…`，同一 model/adapter/input/generation 配置，12/12 generate、无重试、wall 117.375 s；
9 个 child PID 全部不同。每次 source 后交错执行两臂，输出 prompt/rendered/input IDs/hash/token IDs 一致，restart
执行 n0+n1，recovery 只执行 n1。每次完整结果如下，不以平均值替代：

| repeat / 顺序 | restart path wall | recovery path wall | recovery−restart | restart model load / n0 / n1 | recovery model load / validate / rebuild / n1 |
|---|---:|---:|---:|---:|---:|
| rep01 / R→recovery | 33.137546 | 22.269451 | −10.868095 | 1.156670 / 10.741790 / 3.796527 | 1.152671 / 0.000538 / 0.002638 / 3.796373 |
| rep02 / recovery→R | 31.375792 | 20.725828 | −10.649965 | 1.151947 / 10.845278 / 3.793490 | 1.145808 / 0.000265 / 0.002508 / 3.832070 |
| rep03 / R→recovery | 31.856374 | 21.004044 | −10.852330 | 1.139647 / 10.729573 / 3.793990 | 1.140711 / 0.002795 / 0.002495 / 3.807233 |

三次中 recovery 都更快，差值范围 10.650--10.868 s。公式字节为 restart 192,757 B、recovery 2,185 B；没有实际
网络传输。每个 role 是冷新进程、模型不跨进程驻留；OS 文件缓存未清空、缓存状态未控制，首个 source load 也明显较高。
因此只可称固定实现同机复测，不是独立主机、真实 RSU、无线、排队或稳态测量。任务质量仍 unavailable。

## 最终安全结论

- 原 workload 可支持 base identity/capacity 字节账本和技术链正确性边界；原 72 行与失败记录完整保留。
- 原 `local=exact 24/24` 不能独立支持局部规则稳健性或方法改进空间；一致性主要由共享实现真值和可分结构决定。
- 在本机低恢复成本复测区间，恢复相对重跑有约 10.65--10.87 s 的宽 margin，简单阈值足够。
- 在 break-even 附近或高恢复成本下，固定低成本估计会退化；最大观察到的时间优先选错代价是总完成时间
  17.696701 s、makespan 8.848351 s，同时仍节省 dynamic bytes 和重复计算，必须按 Pareto/字典序解释。
- 真实方法改进空间首先是独立成本校准、margin/uncertainty guard 和明确的 readiness contract，不是 RL。当前二选一、
  共同 model-prepare 和可分结构下不引入 RL；保留简单阈值。
- 外部 RSU/network、production KV/tensor state、队列/丢包、独立主机复现和任务质量仍未执行，不产生 paper-ready claim。
