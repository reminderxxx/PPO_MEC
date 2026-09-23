# G14R21 统计生成器纠错、正式结果勘误与 holdout 执行合同审查

- `reviewed_at`: `2026-09-23`
- `literature_cutoff`: `2026-09-21`（沿用 G14A01；本轮未作新颖性评价）
- `target_venue`: `IEEE Transactions on Mobile Computing (TMC)`
- `artifact_run_id`: `typed_model_cache_post_ablation_20260921_g14e07_pending`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `engineering_baseline_commit`: `1770401ccc2f4ce15b5d5adcabdecb386fd90a05`
- `implementation_commit`: `8fb40515f4587320ae7985d058393b7b7aae817e`
- `evidence_level`: `E3_TARGETED_REPRODUCED` 仅适用于冻结 formal 原始行上的 mean、nullable coverage、window sign test
  与 fixed-family Holm；bootstrap CI/effect size 经 old/new 全项一致性核验，但未由第二实现独立重算。完整 paper-ready
  证据因 holdout 未开启仍为 `Unverifiable`
- `verdict`: 统计修复与 formal 勘误通过；holdout 申请为 `NOT_ISSUED_BLOCKED`，不是 READY

## 结论

G14R21 没有训练、重新选模、正式 rollout 或 holdout 消费。原 G14E07 root、统计、gate、ledger、模型和七个用户
文件保持只读。纠正版只写入新的 analysis root：
`artifacts/analysis/typed_model_cache_g14r21_corrected_statistics_holdout_audit_20260923_v1/`。

复算结果为 `0 supported / 72 mixed / 12 contradicted`。12 个 contradicted 全部来自
`transfer_mb_per_request` 的 candidate-adverse signed CI；另两个 transfer 比较跨 0。84 项 window-level exact
sign test 经 Holm 后均未通过 0.05。该结果与 G14A01 的独立对照一致；不导入生产统计 helper 的第二实现从
三份冻结 controller CSV 复算 mean、nullable coverage、window sign test 与 Holm，`mismatch_count=0`。第二实现
未重复 bootstrap CI/effect size，因此不把该部分表述为独立复算。

## 修复内容

1. `raw_mean_delta_candidate_minus_baseline` 永远表示 candidate minus baseline；`mean_delta` 只转换一次并始终以
   正值表示 candidate 有利。raw CI 由 signed CI 的确定性逆变换得到，不再产生第二套随机边界。
2. hierarchical bootstrap、BCa/percentile CI 与效应量规则保持不变；exact sign test 改为先在每个 outer
   raw-time window 内聚合 capacity/seed/workflow，再以 window mean 计 win/tie/loss。
3. 输出同时保留 row-level diagnostics 与 window-level test counts，并显式记录有效窗口、tie、sign-test
   denominator、低窗口数、nullable drop counts、84 项 preregistered/available family size。
4. Holm 消费未舍入 exact p；展示 p 保留 6 位。LRU transfer 为 exact p=`0.00390625`、Holm=`0.328125`。
   冻结文本同时写有“84 项 preregistered family”和“available finite p-values”，对 unavailable 项如何占位存在歧义；
   本轮采用不受结果驱动的保守解释：family 恒为 84，unavailable 比较自身 p/Holm 为 null，但仍占一个 family slot。
5. claim helper 直接消费 `signed_positive_favors_candidate=true` 的 CI，不再按 lower-is-better 二次翻转。
   `formal_gate.passed=true` 仍只表示完整性；纠正版科学 classification 是独立产物。

原 formal gate 的科学 map 为 `72 mixed / 12 supported`；纠正版为 `72 mixed / 12 contradicted`，逐项差异文件
明确记录 `72 mixed→mixed` 与 `12 supported→contradicted`，而不是用修复后的 helper 覆盖旧错误状态。

## Formal 纠正版事实

- 相对 LRU 的 transfer：raw candidate-minus-baseline `+2.440697 MB/request`；signed `-2.440697`，BCa 95% CI
  `[-4.951683, -1.307530]`；窗口 W/T/L=`0/3/9`。含义是 candidate 多传输，不是节省传输。
- service-ready、joint hit、request-ready、continuity 的点估计略正，但 CI 下界为 0，窗口 W/T/L 主要为
  `2/10/0`，不能声称显著优势。
- delay 只在 completed workflow 条件下可用：SA-GHMAPPO 为 `220/540`，只覆盖 `9/12` 外层窗口；15 个 agent
  availability mask 相同，14 个配对 delay effect 全为 0。null 未插补为零。
- 576 MB 与 864 MB 的 2,700 对结果在六个 primary endpoints 和 total reward 上差异数均为 0，只能报告
  saturation/non-discrimination，不能声称 576→864 的扩展收益。
- 14 个 transfer 比较中 12 个 CI 位于 candidate-adverse 方向、2 个 mixed；其他 primary comparisons 全部 mixed。
- typed semantics 预注册 6 个 level，仅 `typed_full` 与 `no_prediction` 可执行；其余 4 个均为
  `unavailable_pre_execution`。局部 readiness/handoff 收益与 transfer/backhaul 代价必须同时报告。
- 不显著不等于等效，完整性 gate passed 不等于算法优势。

## Holdout 未签发审查

只读取 seal、四个 split 的 metadata 和实现代码。train/dev/formal/sealed-holdout=`24/12/12/12`；按
`source_segment_run_id` 的 raw frame/time interval 复核无重叠，最小非重叠 gap 为 24 raw frames。没有读取 reward、
ranking、cache outcome 或 agent outcome，没有调用 policy/episode consumer。

当前不具备申请资格，blocker 为：

1. one-time execution token 状态仍为 `not_issued_in_G14B`；
2. Protocol 2.9 capability registry 明确 `holdout_capability=false`；
3. 没有冻结 dedicated end-to-end public holdout runner/command plan；
4. output root、append-only opening ledger、completion receipt 与 integrity consumer 尚未冻结成一个事务；
5. 精确运行量与经过测试的 wall-clock 上界尚未冻结。
6. 本轮按 metadata-only 边界只核对 150 个 checkpoint 的 manifest/source-reference 绑定，没有重新读取并散列
   checkpoint bytes；opening gate 仍必须执行 byte-level hash validation。

seal 的 outer plan canonical hash=`3570…` 与 split semantic hash=`aa9a…` 已按 expected/observed/pass 机器复核；模型
来源、150 个 checkpoint 的 source-reference/seed/provenance manifest、六个 immutable source files、catalog
fingerprint、预算和 Protocol 在 manifest 层核对通过。checkpoint bytes 与未来最终 commit/package hash 仍是
unchecked opening conditions，这些正向条件不消除上述 blocker。历史 formal cache-policy + controller 的 `5h12m`
和 statistics/gate 的约 `15m` 只作为粗略下界；未测上界保持 `null`。本轮没有签 grant、没有打开 holdout。

## 验证与证据

- 后台固定作业：`return_code=0`、`artifact_validation_passed=true`、自动重试 0；起止时间、PID、命令、日志和
  allowed write roots 见 `local_job/completion_receipt.json`。
- 独立 raw-row 关键纠错复算：84 项、`mismatch_count=0`、`0/72/12`、Holm 显著数 0；命令、环境与 audit
  script SHA-256 写入 `independent_recalculation.json`。
- 定向统计回归 62 passed；加入 live statistics consumer 与 env contract 后最终 scoped suite 为 77 passed；
  compile/import 与 `scripts/smoke_test.py` 通过。
- 主要机器证据：`corrected_statistics/`、`corrected_claim_map.json`、`old_to_new_difference.json`、
  `independent_recalculation.json`、`scientific_limitations.json`、`holdout_interval_audit.json`、
  `holdout_application_unsigned.json`、`source_provenance.json`、`artifact_integrity_manifest.json`。

## 距离论文结果定稿

先冻结并独立审查 dedicated holdout executor、一次性 opening/ledger/receipt/integrity transaction、精确工作量与失败
处置；签发后才可一次性开启。holdout 完成后必须按本轮已冻结的窗口单位、84 项 family、nullable/delay 和 trade-off
规则复算，不得按结果换 checkpoint、指标、窗口或 family。若论文保留 base sharing、workflow-state migration、
eviction-policy 或 supervised predictor 的机制主张，还需补齐匹配消融；否则应收缩对应 claim。不得自动标记
paper-ready。
