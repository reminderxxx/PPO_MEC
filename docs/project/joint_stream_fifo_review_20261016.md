# 同帧FIFO顺序独立审查

reviewed_at: 2026-10-16（Asia/Shanghai）；literature_cutoff: 2026-10-15；target_venue: IEEE TMC；policy_version: tmc_review_policy_v3_20260621；Git: 19a1cf1；artifact_run_id: artifacts/joint_stream_workloads；evidence: bounded source witness, paper E0_UNAVAILABLE/Unverifiable。

独立源审查发现StreamEnv按(frame,vehicle,request_id)排序，而外部清单按wave0、wave1发布。反例：z_r0与a_r1同车同帧，清单先z后a，实际先a后z。queued_batch和stream虽为相同任务集合，却可能出现不同执行顺序，不能把差异仅归到到达时刻。容量下界也用了名字顺序。该问题在提交19a1cf1的dirty smoke未覆盖；属于生命周期接口错误，与policy结果无关。

已停止原216工程运行；partial原件和stop_record保留，不进入完整矩阵比较，不据其得分选规则/场景。实际完成数以stop_record为准，另最多1个未记录工程单元，RL训练仍0。旧命令日志保留。

独立修复任务：同frame同车保持发布清单顺序，环境与FIFO必要下界一致；新增词典序相反的同帧请求及queued/stream同顺序验证；新workload/protocol标v2，使用独立root joint_stream_workloads_v2，全部216从头运行，不拼接partial。已提交的v1为历史工程，不作为有效算法证据。其余矩阵、资源、费用、规则与预算保持原计划，不按得分调整。
