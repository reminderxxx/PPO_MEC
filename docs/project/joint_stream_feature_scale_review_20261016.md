# 持续请求学习初始共享状态尺度复核

reviewed_at: 2026-10-16 Asia/Shanghai；literature_cutoff: 2026-10-15；target_venue: IEEE TMC；policy_version: tmc_review_policy_v3_20260621；Git: 3357f74（实现dirty）；evidence_level: raw first-observation witness / paper E0_UNAVAILABLE；artifact_run_id: joint_stream_learning_smoke_v3。

八配置短联调、账本/重放通过后，在真实输入首次观测直接读取到：本地rsu_a capacity4096，其他三个站capacity尚不可见；新学习候选用缺省1字节作模型尺寸分母，造成base/adapter特征最大2048。不是神经网络图训练失败证据，但会在冷启动把未知观测放大至饱和，必须在正式科学训练前修正。

独立最小修复：已知容量继续使用实际可见容量；未知容量只用静态公共catalog最大合法依赖闭包字节作**归一化尺度**，不推断真实容量、不使未知容量的预取变可行。所有算法同样应用，当前模型保护/mask/reward不变。补目标回归和一轮有限软件全链验收，旧三轮smoke原件保留，无按得分选择参数。
