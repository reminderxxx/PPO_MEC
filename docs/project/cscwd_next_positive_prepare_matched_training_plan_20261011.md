# 下一轮 positive-prepare 匹配训练最小计划（未授权）

当前 training-source qualification 为 `gate_passed=false, training_authorized=false`。本文件只冻结最小计划，不启动训练。

## 数据资格先于训练

1. 另立全新 development split；每个窗口必须有原始 `source_segment_id / vehicle_id / Global_Time / Frame_ID`。
2. train、checkpoint-selection、development-check 三种角色按原始 time/frame interval 严格互斥，并登记与全部历史已暴露区间的
   overlap count。US-101 当前窗口已作 selection，不能移作训练后仍称独立。
3. 候选来源按 `(source_segment_id, Global_Time_start, vehicle_id)` 固定排序；最多 192 个总 env steps，固定 seed/action-cycle，
   所有通过和排除行均保留。只允许读取 public mechanism eligibility、base label/reason/weight，不读取方法 reward、排名、
   checkpoint 或 future truth。
4. 使用真实 SA `_event_auxiliary_target` 复核 direct label。门槛事前固定为：至少 1 条 base
   `prepare, weight>0`，且候选在同一条上产生 `abstain, weight=0`；consumer mismatch 必须为 0。第一条按固定顺序通过的
   training source 才可冻结，不能按 reward 或效果择优。
5. development-check 不参与调整 coverage 门槛或来源搜索；门失败只记未覆盖并停止。不得下载新数据、解封 I-80/holdout、
   修改 RSU layout、reward 或 action 权限来制造机会，除非中央另行批准新的数据资格任务。

## 通过后才可申请的匹配训练

- 唯一变量：`mechanism_aux_positive_prepare_calibration_abstention_enabled=false/true`。
- 两臂共同：同一 train/selection/check split、seeds `[7,17,29,43,61]`、每 cell 1,152 env steps、24 updates、每 update
  8 optimizer steps、selected 与 fixed-update24 双视角；不自动 retry。
- 主要指标：workflow completion、on-time completion、service failure；共同报告 reward、elapsed、transfer、recompute、
  base-positive exposure、candidate prepare→abstain 数和 consumer match。
- 否证/停止：任一视角 completion/on-time 下降或 failure 增加即否定；实际 supervised intervention 再次为 0 则判
  `UNINFORMATIVE` 并停止。PASS 资格本身不授权训练，仍须中央单独批准。

