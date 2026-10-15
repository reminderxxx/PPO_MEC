# DQN终止信用接口的独立审查

- reviewed_at：2026-10-15；literature_cutoff：2026-10-15；target_venue：IEEE TMC；policy_version：tmc_review_policy_v3_20260621。
- artifact_run_id：`joint_scheduled_control/{main,baselines,stability}/seed{7,17,27}`（仍在执行）；Git：aa8676a；evidence level：源码/代数合同witness，论文E0_UNAVAILABLE / Unverifiable。

1. artifact：原九配置尚未全部回收，禁止提前排名；原DQN继续保留，不删除、不挑seed或checkpoint。
2. provenance：共同credit定义为半local return加半team return/初始请求数。PPO的episode消费者显式保留已完成请求的team tail；DQN把下一帧无自身请求直接设terminal，终止target只剩当前帧reward，漏掉该tail。
3. 公平性：这是目标消费接口不一致，不能用原DQN排序证明主方法优势。终止时应补入观测到的discounted team tail（training-only标签，不读未来给actor），非终止仍为正常target-network TD。独立重训DQN三个seed各128，旧结果单列legacy-terminal诊断，不按新旧得分选择。
4. witness：两请求、当前local=team=1，下一帧team=1，gamma=.99，scale=.1。共同MC target=.5*.1+.5*(.1+.99*.1)/2=.09975；旧DQN终止target=.5*.1+.5*.1/2=.075，漏.02475。共享cache行为会影响完成后的其他请求，不能默认该tail与已完成请求的动作无关。
5. claim：此witness证明接口差异，不证明其是DQN后期回退的唯一原因。修复只改DQN终止bootstrap credit；环境/资源/预测/共同PPO逻辑与其他方法不变，Git共享模块逐文件hash核对后再合并。原comparison_plan不覆盖修改；另写append-only amendment明确替换身份与384episode额外成本。论文主比较仍未验证。
6. novelty：无新文献检索，不作novelty评价。
