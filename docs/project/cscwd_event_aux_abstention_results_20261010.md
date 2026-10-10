# Event auxiliary abstention 匹配开发结果（2026-10-10）

## 审查身份与裁决

- `reviewed_at`: `2026-10-10`（Asia/Shanghai）
- `literature_cutoff`: `2026-09-28`；本轮未做新文献检索
- `target_venue`: `IEEE TMC`
- `artifact_run_id`: `cscwd_event_aux_abstention_ab_20261010_v1`
- `policy_version`: `tmc_review_policy_v3_20260621`
- `git_commit`: `abca7e047d672644f3e31d821831e23f80723a52`
- `implementation_commit`: `46a68f11c289ccc304b88cdfed01c34ba5f61c3d`
- `evidence_level`: `E2_ARTIFACT_AUDITED_DEVELOPMENT_ONLY`
- `verdict`: `MIXED_STOPPED / not an algorithm-success or paper-ready result`

唯一 scientific child terminal=`PASS`：5 cells、28,800 environment steps、3,840 optimizer steps、200 新评价，0 retry、
0 formal/holdout、0 download。scientific run、原 analysis 和 supervisor 的 integrity 分别为 `40/40`、`8/8`、`11/11`，
全部 size/SHA-256 独立复算一致。create-only v2 只读消费 1,040 evaluation rows、3,840 optimizer rows，生成 200 对完整
episode 配对；新增 training/evaluation/reselection=`0/0/false`。

按事前门槛最终为 `MIXED`，不得因 selected 视角较好而晋级。候选支持“服务失败可靠性改善伴随 model preparation/传输成本
增加”的 tradeoff，不支持“整体算法成功”、SA 稳定领先、formal/holdout 或论文优势。

## 两视角总结果

failure episode 与 failed-service attempt 分开报告；completion 在两臂、两视角均为 100/100。

| view | arm | on-time | failure episodes | failure attempts | total transfer MB | recompute s | completed elapsed s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selected | control | 38/100 | 32 | 50 | 13,092.625 | 2,725.595 | 7,549.875 |
| selected | abstention | 45/100 | 0 | 0 | 30,209.709 | 2,506.142 | 7,392.217 |
| update96 | control | 43/100 | 26 | 39 | 17,772.839 | 2,591.527 | 7,345.649 |
| update96 | abstention | 44/100 | 2 | 4 | 29,039.714 | 2,769.662 | 7,581.444 |

selected 的 on-time `+7`、failure episode `-32`、attempt `-50` 与 elapsed/recompute 下降是正信号；但平均 total transfer
增加 `171.171 MB/episode`。fixed update96 仍减少 failure episode/attempt `26→2`/`39→4`，但平均 elapsed、recompute 和
total transfer 分别增加 `2.358 s`、`1.781 s`、`112.669 MB/episode`。这不是单调改善。

## split 与 seed 门禁

- selected regression on-time `29→31/60`、failure episodes `19→0`；frozen-check `9→14/40`、`13→0`。
- update96 regression on-time `32→30/60`、failure episodes `17→2`；frozen-check `11→14/40`、`9→0`。因此 fixed96 的
  regression on-time 已直接下降。
- selected 每 seed on-time delta：seed 7/17/29/43/61=`+.30/-.05/+.10/0/0`。
- update96 每 seed on-time delta：`+.30/-.05/-.15/0/-.05`；17、29、61 三个 seed 下降。预注册 seed gate 要求每视角
  至少 3/5 改善且至多 1/5 恶化，fixed96 仅 2 个归为 improved、3 个 worsened/tradeoff，故失败。

不得用 selected checkpoint 的服务改善覆盖 fixed endpoint 的不稳定性，也不得事后改变选模、挑 seed 或重跑评价。

## 动作、状态与监督机制

- selected current-missing action4 `46→0`，action0 总数 `24→124`，failed attempts `50→0`；candidate state commits 为
  144（control 166），说明可靠性改善不是来自更多 state commit。update96 current-missing action4 `36→2`，attempt `39→4`。
- selected action0/action4 的整体变化为 `24→124`/`253→175`；update96 为 `49→97`/`248→186`。候选没有删除 action4，
  只是明显转向当前服务填充。
- 3,840 optimizer-step 原始日志的 event eligible/supervised/abstained=`110,072/79,796/30,276`，直接监督比例
  `0.724944`；各 seed 为 `.712055–.753230`。所有计数满足 eligible=supervised+abstained，auxiliary loss 和加权梯度均 finite。
- 历史 control 没有该 supervision fraction 字段，故不伪造旧比例；这里只把 candidate 的直接日志用于机制兑现。

## 解释边界与停止决定

结果支持 abstention 改变了 action4/current-service 行为并显著减少服务失败，但同时增加 model preparation/transfer；fixed96 的
按期率和成本不稳定说明它是可靠性—成本 Pareto 交换，不是完整算法成功。旧 hard-zero 的 `MIXED_STOPPED`、no-aux 负结果、
两步规则的 exact-transition/lexicographic 权限和所有旧 checkpoint 保持不变。

本 candidate 已触发预注册停止：不重训、不重评、不扩 seed/预算、不改选模、不串行搜索第二候选。是否存在实现缺陷须等待 A
对原生加载/重算事件链、fallback 权限与计费对称性的独立报告；只有固定合同/数学定义与代码执行明确不一致才可称 bug。若只有
合法 Pareto 权衡，则无需“修复”，下一步只能另立预注册多目标设计，而非继续找 SA 优势。

机器证据：`artifacts/analysis/cscwd_event_aux_abstention_ab_20261010_v1_analysis_v2/`，其中
`arm_view_split_seed_summary.csv` 72 行覆盖两 arm×两视角×全部 split/seed，`paired_episode_rows.csv` 为完整 200 对，
`event_supervision_summary.csv` 为五 seed+combined，`final_verdict.json` 固定 `MIXED_STOPPED`。
