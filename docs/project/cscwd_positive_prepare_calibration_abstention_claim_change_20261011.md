# A 论文线主张与证据变更：正 prepare 校准来源 abstention（2026-10-11）

本说明独立交付，不修改 A 线主论文稿。绑定实现 commit
`f500f5ef1e96f3c39b0a99f2a93adb6655c6b75f`；evidence level 为
`E2_IMPLEMENTATION_AND_STATIC_RECOMPUTE / no training`。

可新增的事实主张：现有 public-causal 正 prepare 标签未检查 predictor calibration provenance；已实现一个默认关闭的门，
仅 literal `confidence_calibrated=true` 保留正 prepare supervision。11 个既有公共状态静态重算只出现 1 个
`prepare→abstain`，7 个 serve 全部保持。Raw producer 合同固定 false，所以当前候选实际撤销该 profile 的全部正 prepare
auxiliary supervision。

必须保留的边界：未运行新训练、未产生 checkpoint、未读 formal/holdout，也未证明该候选改善 completion、failure、成本、
迁移或 SA 相对基线表现。不得写成奖励已修复、退化根因已证实、fine-grained uncertainty control、SA 稳定领先或论文贡献已
验证。旧 pilot 结果与 checkpoint 只作历史描述，不能充当新候选的匹配 control。

下一步只有一个冻结 A/B；后续已获一次性 `execution_authorized=true` 授权：原 public-v3 SA 与新 gate SA 单变量比较，并在同一次新 run 中
重训 PPO/MAPPO、重评 public rules；共同 23,040 steps、5 seeds、selected/update24 双视角。任何服务恶化或 Raw 正 prepare
仍被监督均直接否证；development 支持也不自动晋级 formal/holdout。

该 A/B 不检验“三头结构优于公平基线”、真实跨 RSU 净收益或 sharing×migration 交互。现有本地 metadata 只有已暴露的
Lankershim/US-101 合格来源；Peachtree 时间单位待核验、I-80 属另一封存协议，二者均未获得当前 CSCWD 118.8 s 合同的
可用资格；anonymous prefix 无 raw time mapping，旧 sealed holdout 不得解封。故独立泛化仍需新 source-disjoint 数据资格与
明确授权；新增 seed 不构成独立证据。
