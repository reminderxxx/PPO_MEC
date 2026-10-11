# CSCWD phase-v2 公共消费者验收与一次性 pilot 协议（2026-10-11）

## 身份与当前裁决

- `reviewed_at`: 2026-10-11 Asia/Shanghai
- `literature_cutoff`: 2026-10-11（本轮未检索新文献）
- `target_venue`: IEEE TMC
- `artifact_run_id`: 预留 `cscwd_phase_v2_causal_public_pilot_20261011_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `Git commit`: estimator v3=`cfb0c23057a9ce8443feb65afbb3258288d60c47`；phase-v2 env=
  `366c951b35df4c8e5f36d52e437cf4d88890e4ac`；最终启动提交由 supervisor/run manifest 精确记录
- `evidence_level`: `E1_INTERFACE_CONFORMANCE / development protocol frozen; performance UNTESTED before launch`
- `verdict`: `ACTUAL-CLASS CONSUMER PASS / ONE-SHOT DEVELOPMENT PILOT AUTHORIZED`

本轮没有修改 reward、action mask、网络参数量、PPO/optimizer、DAG、deadline 或两个冻结来源窗口。唯一 SA A/B 变量是
`mechanism_aux_causal_public_prepare_advantage_enabled=false/true`。候选概念仍是 public-causal 三态 event supervision；
因为 phase-v2 新增了公开阶段输入，estimator/event/rule 与 checkpoint 语义递增为 v3，不是另加算法变量。

## 实际类验收

输入身份：phase contract SHA-256=`2a0cee4b...6343`；phase-v2 env SHA-256=`5a5b888f...b683`；public estimator
SHA-256=`a6b49bd2...1790`；A 最终 public fixture SHA-256=`1e7da4d9...d8c2`；内嵌 public-info SHA-256=
`0edb2a05...20f`，独立重算一致。

在实际 `RawNGSIMPhaseV2Env`、US101 长开发轨迹、`dev_01/default` 上：

- reset 公共信息与冻结 fixture 完全一致；estimator/event/rule schema 分别为
  `causal_public_action_estimator_v3`、`causal_public_prepare_advantage_v3`、
  `causal_public_rule_selection_v3`；
- label=`prepare`，public immediate/two-step 均选 action 4；phase-v2 two-step 因公共状态没有二维速度/目标进入时间，
  明确退化为相同 immediate score，不能计为独立强基线样本；
- action 4 实际完成当前服务与迁移，耗时 `1.961859672 s`；phase elapsed、环境时钟和 backhaul byte split 守恒；
- `clone_for_decision_model()` fail-closed，未给公共规则或 learned policy 隐藏实际 future contact；
- SA candidate 前向动作在公共 mask 内、log probability 有限；实际四 learned cell 均完成 48-transition rollout、
  truncation/bootstrap 与 8 optimizer-step 接口回归；旧 v2 candidate checkpoint 继续 fail-closed 拒绝。

本门只证明接口/权限/核算一致，不证明候选优于 control、PPO 或 MAPPO。

A 随后按冻结上限完成一次 60-cell fixed-policy mechanism matrix：60 episode、522 real steps、731 preview upper bound；
manifest/ledger SHA-256=`cb581f68…bc2c`/`bfb2d674…4028`，独立 audit SHA-256=`30d821d4…db8d` 且
`all_checks_passed=true`。v2 的 18 次 action4 为 9 delivered/9 target-departed；重复 US101 轨迹跨 method/condition 不是独立样本。
default public immediate/two-step 均 2/2 on-time，但 two-step 是同分 fallback；该矩阵只关闭 data/mechanism gate，不形成 learned 排名。

## 唯一匹配 pilot

| 项目 | 冻结值 |
|---|---|
| learned cells | SA control、SA candidate、PPO、controller-level MAPPO × 5 seeds=`20` |
| seeds | `7,17,29,43,61` |
| 每 cell | 48 transitions/update × 24 updates=`1,152` env steps；4 PPO epochs；minibatch 32；192 optimizer steps |
| 总预算 | `23,040` env steps；`3,840` optimizer steps；无 retry |
| 训练 | Lankershim/default、`dev_01`、全部从随机初始化 |
| 选模 | US101/default；updates `6,12,18,24`；只按 on-time/completion/failure/unfinished/bytes，reward 与最终 case 不参与 |
| 最终双视角 | selected 与固定 update24 |
| 固定 cases | phase-v2 的 default/adverse_link/ample_cache + phase-v1 mechanism-off，各用两条来源，共 8 case/view/cell |
| learned final eval | `320` episode；选模 `80` episode |
| rules | public immediate/two-step；privileged exact two-step 只在 v1 capability reference 报告，v2 不可用 |
| 数据边界 | 两个 118.8 s 已暴露 development 来源；不是独立 test，不做显著性主张 |

失败/停止条件事前冻结：candidate 在 selected 或 fixed-update24 任一视角 completion 下降或 failure 上升即 `FAIL`；
只在一个视角/少数 seed 改善，或仅形成服务—资源交换，即 `MIXED`；只有双视角 completion/on-time 不降、failure 不升、
至少一个服务指标严格改善、方向至少 3/5 seed 有利且最多 1/5 不利，并且 transfer/recompute/partial-byte 不恶化，才记
`SUPPORTED_DEVELOPMENT_CANDIDATE`。任何裁决都不自动进入下一阶段，也不改 formal/holdout 或 paper-ready 状态。

## 复现与验证

预检：

```bash
/Users/howen/Projects/PPO_MEC/.venv/bin/python \
  scripts/run_cscwd_phase_v2_causal_public_pilot.py --preflight
```

预检已读取本机 raw CSV 并完成两窗口扫描，19.6 秒返回 `PASS`；同时核验 60-cell matrix manifest/ledger/audit 固定 hash 与
`all_checks_passed=true`，没有生成训练步。局部回归见
`tests/test_cscwd_phase_v2_causal_public_pilot.py`。正式启动只能在 final commit 已 push、checkout clean、三个 create-only
root 都不存在时，由 durable launcher 执行一次；ACK 后不持续轮询。
