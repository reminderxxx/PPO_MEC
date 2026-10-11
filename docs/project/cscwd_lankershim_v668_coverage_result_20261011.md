# Lankershim v668 positive-prepare coverage 结果

- entry commit: `ccef09eb992ead8e8debef3f66be63c42d6c26f5`
- parent metadata SHA-256: `7aaad8cec66908e5e4044e20f0dd3e1086eafbf925686f84da6947c35829fdec`
- train window: vehicle 668，Global_Time `1118936135400–1118936254200`，Frame_ID `4555–5743`
- scope: `development_unknown_not_independent_generalization`

独立 metadata 核验确认原 CSV SHA、1,189 行/118.8 s、Frame +1、time +100 ms、有限坐标及与已登记保护区间无重叠。
selection v19 与 check v749 未加载 trace、label 或 reward；历史 check 暴露不完整且三段同一 Lankershim recording，不构成
独立泛化证据。

一次性 192-step train-only probe 已完成：0 optimizer、0 parameter update、0 checkpoint、0 formal/holdout。真实 SA consumer
与 direct label 的 mismatch 为 0；执行动作 0/1/2/3/4=`70/8/76/23/15`，但 base label 为 abstain/serve/prepare=
`177/15/0`。原因是 `action4_not_legal=131`、`decisive_public_comparison_unavailable=46`、
`action4_fails_current_service=15`。candidate `prepare→abstain=0`。

因此 `coverage_gate_passed=false`、`matched_training_may_be_frozen=false`。动作 4 可执行不等于存在正 prepare 监督；不得启动
4-arm×5-seed pilot，不得换下一车辆、扩预算、改 reward/layout/label 或读取 selection/check 寻找机会。本轮停止。

Artifact root：`artifacts/analysis/cscwd_lankershim_v668_positive_prepare_coverage_probe_20261011_v1/`；integrity/summary
SHA-256=`e3a5445288166f750d3f8ae46d8cfe141dfd6a10a8093ba0acf3024ba0fdd00e`/
`4215dd977c969df1bbedad698664272658bf25d873c7672ee66fcfff1aff8986`。

