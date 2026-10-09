# Strong-baseline A/B 支线只读比较边界

- `reviewed_at`: `2026-10-09`
- `audited_commit`: `d18e7bfc1c81dc2fd99166e61489d66e0c4c0b65`
- `scope`: public capability/config only；未运行 preflight、训练或评价

该提交新增 development-only strong-baseline runner/config。其 learned methods 为 SA-GHMAPPO、controller-MAPPO、PPO 和
project-native `dt_handoff_drl`；另含 prediction-aware、per-instance reset 的 stateful popularity heuristic，以及使用 exact
environment transition clone 的 model-based two-step planner。`dt_handoff_drl` 是文献启发的项目实现，不是论文原方法精确复现；
two-step rule 拥有两步精确模型和字典序目标权限，不能与 model-free policy 写成同能力方法。

配置固定 `original_reward_v1`、`critic_target_normalization=raw_disabled`、seeds `7/17/29/43/61`，learned cell 为
1,440 steps、24 updates、60 transitions/update、4 epochs、batch 32、192 optimizer steps，candidate updates 为
`6/12/18/24`；36 个实例全部为已消费 development data。配置仍为 `execution_authorized=false`。

这条 A 线不能与本支线的 PopArt A/B 合并排名：A 线使用 original reward + raw critic，方法集合也增加了 DT/heuristic/rule；
本支线使用 service-aligned reward 并比较 raw critic 与 PopArt。reward、critic contract、方法集合和可能的 checkpoint selection
路径均不同，跨表排序不具有共同 estimand。安全用法是分别报告：A 线回答 original-reward 下 strong-baseline capability；B 线只回答
service-aligned reward 下 PopArt 的 critic-scale 与服务充分性。不得把两表拼成单一 leaderboard，亦不得用任一线的最终评价
反选另一线 checkpoint 或候选。
