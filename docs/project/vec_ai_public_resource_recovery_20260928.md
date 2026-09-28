# 既有数据包找回与公开车联网AI工作负载资源复核

- reviewed_at / literature_cutoff: 2026-09-28
- target_venue: IEEE TMC；policy_version: tmc_review_policy_v3_20260621
- git_base: 1350a40；artifact_run_id: none_public_discovery_only
- evidence_level: 本地登记/源码及公开官方页面核对，未验收新payload
- verdict: REUSE_CANDIDATES_FOUND；不必从零构造AI语义；无完整joint VEC cache trace资格结论

## 1. 找回了什么

此前G11已有19项登记，原件为 `model_cache_dataset_discovery_audit_20260819.md`、
`configs/data/model_cache_dataset_registry.json`、`DATASET_SOURCES.md`。G11明确记录未下载原始payload、
未实现importer；G13仍为metadata-only。不存在“此前找到=已下载=已融合”的等价关系。
阶段一只查本地模型权重，不足以否定这些外部数据候选；本次修正遗漏，不改写旧登记。

本地 `data/raw/model_cache/huggingface_model_cache_sources.json`中的五项HF候选是文件大小/内容或空仓库，
不是AI workflow请求数据。qwen/cbow/bert未知许可边界保持，不因名称含model-cache而启用。
本轮以名称搜索data目录未发现BurstGPT/Mooncake/Qwen-Bailian/DriveLM payload；不作全机不存在断言。

## 2. 核验候选与用途（页面可用不等于数据已验收）

| 资源 | 一手入口及研究出处 | 可复用内容 | 缺口/决定 |
|---|---|---|---|
| DriveLM | [ECCV 2024论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06870.pdf)、[作者仓库](https://github.com/OpenDriveLab/DriveLM)、[数据格式](https://github.com/OpenDriveLab/DriveLM/blob/main/docs/data_prep_nus.md) | 驾驶场景中有逻辑依赖的感知/预测/规划QA；提供scene/frame及QA JSON结构 | 首选车载AI语义候选；QA依赖不自动等于可执行任务DAG；示例con_up/con_down可为null，不能假定边全有；没有RSU缓存/迁移成本 |
| V2X-Seq | [CVPR 2023论文](https://openaccess.thecvf.com/content/CVPR2023/html/Yu_V2X-Seq_A_Large-Scale_Sequential_Dataset_for_Vehicle-Infrastructure_Cooperative_Perception_and_CVPR_2023_paper.html)、[作者仓库](https://github.com/AIR-THU/DAIR-V2X-Seq) | 真实车端/路端序列与协同感知/预测任务 | 更贴车路协同，作第二候选；不含本项目model/adapter cache和跨RSU状态成本；不先下载大规模图像点云 |
| BurstGPT | [作者仓库](https://github.com/HPMLL/BurstGPT)、KDD 2025 DOI 10.1145/3711896.3737413 | 请求时间、会话、模型类别、token数、整体elapsed及失败记录；有request generator示例 | 优先到达/会话校准候选；不是DAG/车载trace；保留含失败版本，elapsed不能直接当本地推理耗时 |
| Mooncake FAST25 traces | [官方trace说明](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/README.md)、[FAST 2025论文](https://www.usenix.org/conference/fast25/presentation/qin) | real conversation/tool-agent与另列synthetic；到达、token长度、prefix hashes | 首选AI数据/KV复用候选；hash不恢复真实文本，不等同adapter；需模型配置才能估KV bytes |
| Qwen-Bailian | [官方仓库](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon) | chat/parent/turn、相对时间、token长度、16-token block hashes；官方replayer指针 | 保留session-aware备选；跨源hash不可比较；不代表车辆/RSU identity或精确模型revision |
| WfCommons | [官方项目](https://wfcommons.org/publications)、FGCS 2022 DOI 10.1016/j.future.2021.09.043 | 工作流格式、实例、生成与验证方法 | 复用方法，不把科学计算workflow直接称车载AI |
| OpenDAG-Agent | [作者工具仓库](https://github.com/ANRGUSC/opendag-agent) | 异构edge/cloud/API任务图与调度模拟工具 | README表明cluster execution/profiling仍为roadmap；不是已验证生产trace，论文/固定revision待核，不作为首选数据 |
| TraceLab | [作者仓库](https://github.com/uw-syfi/TraceLab)、[预印本](https://arxiv.org/abs/2606.30560) | coding-agent trace收集/分析方法 | 非驾驶应用；保留方法线索，正式venue/下载内容/许可未核，不引入主线 |

DriveLM仓库声明代码/资产默认Apache-2.0，语言数据CC BY-NC-SA 4.0，nuScenes继承自己的许可。
实际采用前需逐文件复核条款；不是将代码许可扩展到所有数据。其v1.0准备文档只声明公开训练集，
不能凭下载入口宣称有独立测试标签；challenge版本另核。本轮未提交表单、接受条款、下载权重或图像。
其他候选的G11许可记录是历史记录，下载前重新锁定revision并检查对应文件许可，不将仓库badge当数据许可结论。

## 3. 决定：复用后补缺，而非找不到才全部自造

第一优先检查DriveLM的小型公开标注：它比阶段一自拟文本链更直接支撑驾驶AI语义。
阶段一文本服务模板保留为fallback，不再视为已选定最终应用；以样本结构/许可/执行成本判定，不看算法效果。

- 主线A（workflow）：DriveLM提供驾驶任务语义，核实/重建明确标注的依赖；本地测量模型与状态成本。
- 主线B（cache支持）：Mooncake或Qwen-Bailian提供prefix复用，BurstGPT提供会话/到达特征。
  先作为独立支持场景，不能声称它们的请求来自DriveLM车辆。
- NGSIM继续保留移动来源；若与nuScenes/DriveLM跨源组合，车辆ID和共同时间不能虚构匹配。
  若选用同一原始驾驶数据的ego时空信息，也仍需另建RSU/链路模型，不能冒充真实网络日志。
- Alibaba保留结构基准，不必让新AI节点语义继续由其task_type决定；新旧工作负载结果分开报告。

自己新增的部分应是可解释的RSU/资源场景、profiling、状态语义、转换器和可复现生成规则，
不是把DriveLM/Mooncake改名发布。数据贡献声明应写清派生关系、归属、许可及新增验证价值。
本轮检索没有核验出一包同时包含真实车辆/RSU、AI DAG、adapter依赖、缓存事件和迁移成本的资源，
这是有界检索结果，不是全世界不存在的证明。

## 4. 下一步最小资格步骤

1. 锁定DriveLM标注及Mooncake trace的具体公开revision/文件、许可证和size；只选小型标注或有界trace。
2. 核对scene/frame、QA语义、实际依赖字段覆盖率；null依赖不伪造为观测边；定义派生边依据。
3. 核对trace时间单位/相对时间、原始失败保留、token和block语义；不混合16/512-token hash空间。
4. 样本资格通过才决定应用模型与profiling；先复用公开可运行baseline，成本超出设备时先报告。
5. 固定download/hash/local sample审计后再写importer。当前只完成发现/方案，未下载、安装、生成数据或运行policy。

搜索范围包括vehicular edge AI workflow dataset、agent workflow traces、DriveLM、V2X-Seq、AWTO artifact
及既有G11来源。AWTO的GitHub搜索出现同名Kafka账号，不能当论文代码；未核实其可用artifact，不断言无开源。
V2X-QA、DriveNetSim、AgentEval等进一步线索保留为待核；驾驶接管handover不能等同RSU handoff。
研究身份、内容、资源大小及许可在下载前还需检查；本报告不授予数据再分发或真实holdout权限。
