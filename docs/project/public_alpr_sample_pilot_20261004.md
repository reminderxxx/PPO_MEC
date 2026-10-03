# 公开 ALPR 示例与有界任务检查

日期：2026-10-04。此任务由用户“收集数据集后开始下一步”授权。

## 输入与用途

- 数据：UniqueData/license_plates，固定 revision `5c1678c7350bdc7f9d674156bb1926567c6f9a30`。
- 来源：https://huggingface.co/datasets/UniqueData/license_plates
- 仓库实际内容为四国家各 25 张示例，不是完整商业数据集。README 前部的旧配置描述不作为实际文件清单。
- 发布者声明 CC-BY-NC-ND-4.0；本轮仅本地非商业分析，不上传或再分发原图、标签或预测车牌；不保存裁剪、增强图。底层网页图片权利未被独立认证，不能将该声明推广为无条件再发布许可。
- 仅下载原图、TSV 和 README，不执行第三方数据集脚本。
- 原件位于主仓库 `data/raw/ai_task_acceptance/unique_data_license_plates_5c1678c_20261004/`，使用本机 Git exclude 排除；所有逐样本标签、预测、日志和计划在被忽略的 `artifacts/analysis/alpr_public_sample_20261004_v1/`。

## 预固定检查

下载前落盘 design.json；逐文件验证固定 revision 的 size 及 LFS SHA-256/Git blob SHA-1，额外保存 SHA-256。图像解码、标签一一对应，按相同国家+规范化车牌、相同字节或 64-bit dHash 距离 <=4 建立连通分组。该启发式不保证排除全部近重复，更不等同物理车辆/video ID。

按 country、SHA256(1401:path) 排序，每国取前三个不同分组：第一个 development、另两个 locked_check，共 4+8。无结果驱动的样本筛选、提示词修改或重试。两组均无模型训练独立性保证，不称为正式 holdout。

复用现有 SmolVLM base 与 ALPR LoRA，不下载模型，不训练。全原图、CPU float32、单线程、greedy、max_new_tokens=32、12 次 generate、最长 1800 秒。固定提示词见脚本与 design.json。模型进程只读取剔除 reference 的输入表，评分在子进程完成后进行。

评分为完整输出 NFKC、uppercase、alphanumeric 后的 exact match 和总编辑距离/总参考字符数 CER。不做 O/0 猜测、不从解释文本挑车牌子串、不在失误后裁剪重跑。只报告描述性计数，无显著性和泛化 claim。

## 入口

`scripts/run_public_alpr_pilot.py prepare`：一次性收集与冻结。

`scripts/run_public_alpr_pilot.py launch`：一次性启动独立后台宿主（内部 `supervise`），执行最多一次 offline infer，保存 child stdout/stderr、return code、completion_receipt.json、完整性清单。独占创建日志拒绝重复 launch；不得重复使用原输出目录。AI 无需持续轮询；宿主本身异常退出或断电仍可能无终态，需要只读核对，不自动重启。

科学解释器固定为 `/Users/howen/Projects/PPO_MEC/artifacts/environments/adapter_state_acceptance_py39_v1/bin/python`。轻量无模型 pytest 使用已有主 `.venv`，因为独立推理环境未安装 pytest；不新增依赖。

## 结论边界

本轮只回答“现有真实 ALPR adapter 在带标签公开小样本上是否能输出正确车牌”。不回答缓存收益、迁移收益、算法优越、跨 RSU 部署或论文就绪。即使正确率低也原样保留，不强行构造 DAG。任务完成后记录实际结果。
