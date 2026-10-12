# 四站八车冻结工程场景 v2（2026-10-12）

状态：`engineering_replay`。此记录取代 v1 的工程验收结论，不改变 Protocol 2.9 或旧正式结果。v1 选车器把相同 `Location + Vehicle_ID + Global_Time` 的重复原始行覆盖到同一轨迹；其中部分行相距数百米且方向不同。故 v1 的车辆连续性、handoff 数和 8/8 交付验收均**作废**，历史 artifact 只作错误复核材料，不能引用为合格轨迹证据。

## 冻结输入与单位

- 配置：`configs/experiment/vec_competition_peachtree_4rsu_8vehicle_v2.json`。NGSIM Peachtree 原始毫秒 `1163570000` 起，120 个连续 100 ms 帧（11.9 s），车辆 `740,829,832,835,837,839,847,897`，同一来源、不同 ID、各时刻恰好一行；发现重复、时间缺口、方向改变、超限位移或非法观测道路转移即拒绝运行。
- 时间统一秒；`Local_X/Y` 英尺乘 `0.3048` 为米，`v_Vel` 英尺/秒乘 `0.3048` 为米/秒；Alibaba task 大小乘配置中的 1000 转为**受控字节**，不是原生测量字节。原始数据、catalog 和配置 SHA-256 写入 manifest。
- `Section_ID/Lane_ID/Direction` 的同窗观测边作为合法转移检查，**不是完整 GIS 路网**。不把 LuST 仿真路网与 NGSIM 测量轨迹拼称联合实测。

## 冻结资源与执行

- 四个 RSU 局部坐标、半径、每站 512 MB 缓存、1 个计算槽、20 MB/s 无线、40 MB/s 回传端口以及 a-b-c-d 的 40 MB/s 站间链路写入配置。站址和链路是受控推定，并非真实部署或无线测量。每帧导出 8 车接入时间线，断覆盖注入使 8 个请求全部保持 unfinished。
- 八个不同 Alibaba DAG 分配给八车，全部在 0 s 到达，截止 11.9 s。共享无线、回传、云端口、缓存和计算槽；实际传输字节跨帧累积。节点需等待前驱完成及收据到达、输入和模型就绪及计算槽。旧站已开始的节点续算；结果经站间转发和车辆下行才计交付。负载扰动可延迟其他车，两个大 uplink 可互相延迟。
- 模型对象有容量预留、在途加载、依赖满足后的 resident 提升和依赖安全驱逐；运行对象被锁定。控制器只接收当前及过去状态，可发 `prepare` 或 `place`；预测记录目标、ETA、误差及新增错误预取字节。默认状态 codec 只序列化可核验元数据收据，**不是模型/KV/tensor 实际状态**。
- `src/evaluators/vec_competition_audit.py` 独立复算到达分母、交付、DAG/model 因果、每 tick 资源容量、逐 flow 字节守恒、计算和排队等待。输出完整 ledger、summary、audit、站点时间线和离线 replay HTML。结果只证明受控场景内工程合同，不能直接声称算法优越或现实服务时延。

## 仍未关闭的最终论文阻断项

1. NGSIM 对齐的可核验 GIS 底图、坐标参考和真实 RSU 站址/天线、覆盖/无线及回传测量；当前只是短路段观测转移图和推定站点。
2. 有许可且可核验的车载 AI 输入/标注、真实 base/adapter、真实计算/加载/状态字节与推理正确性；Alibaba 只提供真实 DAG 结构，大小是受控参数。
3. 新 `4×8` observation/action/reward 与 PPO/MAPPO/基线的训练、评估、checkpoint、benchmark 消费端；v2 的插件接口和规则策略尚不构成公平学习方法比较。
4. 原始时间区间互斥的多窗口、多 seed formal/holdout/support、同预算强基线、消融、原始结果和正式 checkpoint。单窗 8/8 不属于论文统计结论。

运行：`/Users/howen/Projects/PPO_MEC/.venv/bin/python scripts/run_vec_competition.py --output-dir artifacts/analysis/<new_run_id> --controller no_prefetch_v1`；也可选 `constant_velocity_prefetch_v1`。工作区必须 Git clean、输出目录必须不存在；原始数据不可用或 hash 不符时 fail closed。验证命令：`/Users/howen/Projects/PPO_MEC/.venv/bin/python -m pytest tests/test_vec_competition_v2.py -q`。

本地预提交诊断的独立审计通过，8/8 请求同时到达、70/70 节点、8/8 截止前交付、16 次接入切换、5 次状态收据迁移，队列等待累计 24.5 s；按接入周期去重的预测解析 13 次，仍为 8/8 交付。预提交输出含旧 HEAD，不作冻结 artifact；须在 clean commit 后重新由入口生成。两种规则都是同一受控单窗，不能推断预测器效果。逐车及全局 `sent_transfer_bytes_by_kind` 包括已发送但后续取消的字节；共享模型加载按触发车辆记账，不能解释为其独享资源消耗。
