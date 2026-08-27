# Formal Agent Order Contract

- contract version：`1.0.0`
- active Protocol：`1.7.0`
- order semantic SHA-256：`82e562755dadd4341c950bf71efc488d3527b7f45b7f02512f8064d189b655e0`
- Protocol semantic SHA-256：`5a1c2070529674ecf65c8b836706849f0937853a59b6dfbc3b987d88ac4f50a5`
- evidence boundary：non-formal execution-contract validation；无正式性能结论

## 权威序列

Reactive顺序：

1. `reactive_lru`
2. `reactive_fifo`
3. `reactive_lfu`
4. `reactive_aging_lfu`
5. `reactive_random`

Learned顺序：

1. `sa_ghmappo`
2. `ppo`
3. `mappo`
4. `dqn`
5. `dueling_dqn`
6. `qmix`
7. `controller_mat`
8. `dag_offload_drl`
9. `cache_offload_drl`
10. `dt_handoff_drl`

Main benchmark顺序严格为上述5+10。`popularity_cache_heuristic`是matched report-only heuristic；
`exact_oracle_h1/h3/h6/h12`是exact-oracle report cells。它们都不属于15-agent主benchmark，也不需要
learned checkpoint。

## 身份规则

- JSON object key insertion order只用于存储，永不推导科学、执行、artifact或论文顺序。
- alphabetical sort不能替代权威序列。
- 相同集合但顺序不同与缺失、重复、额外、未知、角色交换一样，必须在执行前fail-fast。
- active producer/consumer通过`src.runtime.formal_agent_order.resolve_formal_agent_order`读取并验证合同。
- Protocol、scientific config、fairness manifest与包含`--agents`的command template必须逐元素一致。
- checkpoint map可以按agent key存储，但candidate、selection、freeze、manifest输出和论文表格遍历必须使用
  权威learned/main顺序。
- pairwise统计固定candidate=`sa_ghmappo`，baseline顺序是main order删除candidate后的14项；输入row重排
  不得改变pair、Holm family或显示顺序。

## Consumer边界

Dev selector不再读取`list(training_budget.agent_configs)`；Protocol v1.7必须提供order contract path，
并在任何nested benchmark前验证scientific config、Protocol matrix、reactive baseline和fairness controller
顺序。`enforce_benchmark_args`继续使用exact list equality，不降级为set equality。

Execution Binding与Resolved Context绑定order semantic hash；training checkpoint metadata、dev candidate、
selection/freeze companion、benchmark raw row/aggregate/statistics和claim/display均携带或验证同一身份。
Protocol v1.0–v1.6只允许历史审计，不得启动新的active formal执行。

## G14C v7失败登记

`typed_model_cache_formal_20260826_233222_g14c_v7`永久状态为
`invalid_after_training_before_dev_performance_execution`。其150/150 training cells与1,200 candidates不能
被解释为可复用checkpoint资产；dev rows/selected/frozen/formal=`0/0/0/0`。旧run、checkpoint、candidate、
partial dev input、ledger和marker禁止resume、finalize、salvage或reuse；active runner、dev、freeze和benchmark
消费者均拒绝该root引用。

## 验收与结论边界

clean detached候选无本地`.venv`，完成Protocol 186-command dry-run、150/150 training command identity、
24个dev nested command/1,200 candidates、三档formal controller/cache-policy/support顺序审计；完整NGSIM
11,850,526 rows解析为73,871 provider frames，60/60 frozen windows可达，clean tests 1077项通过。

non-formal链路新训练10个tiny checkpoints，由实际dev selector生成nested benchmark，实际
`benchmark_main_results.py`与fairness exact gate输出15条agent raw rows；selection/freeze为10/10。
synthetic统计对两种input mapping/row order输出相同14个comparison。所有输出均标记`formal=false`、
`performance_evidence=false`。

Readiness v9为`READY_FOR_G14C_V8_CLEAN_TRAIN_AND_FORMAL`。这只授权未来独立任务从最终Commit A8 clean
worktree新建G14C v8 run；不表示G14完成、formal完成、holdout evidence、算法优势或paper-ready。本任务未
启动G14C v8、G14D或G15；正式training/checkpoint/performance仍为0，holdout保持sealed/unopened。
