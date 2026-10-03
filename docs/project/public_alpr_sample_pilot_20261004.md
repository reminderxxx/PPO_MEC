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

## 收集与启动回执

- 102 个下载文件：100 JPG + 原始 TSV + README，合计 18,939,410 bytes；所有源哈希、大小通过。
- Bahrain/Ireland/Norway/USA 各 25 张；100/100 解码、标签绑定通过。当前重复规则 0 edges、100 components；不声称彻底排除未知近重复。
- 已固定 4 development + 8 locked_check，并一次启动后台 supervisor PID 1096；科学执行代码提交 `b3e825c0f4fc75801e095ae5d7824c829152d9da`。此条是启动记录，不是成功完成或准确率声明。
- `tests/test_public_alpr_pilot.py tests/test_env_contract.py`：17 passed；smoke：6 nodes completed；AST syntax 和 diff-check 通过。独立推理环境首次尝试 pytest 因未安装 pytest 退出；随后使用已有主 `.venv` 跑测试通过，未安装新依赖。
- 终态唯一入口：主仓库 `artifacts/analysis/alpr_public_sample_20261004_v1/completion_receipt.json`；成功后分数为 `task_results_redacted.json`。原始预测和标签只保留本地。
- 首次 `git push -u origin codex/alpr-task-pilot` 因 GitHub 443 连接失败；本地提交完整，运行不依赖 GitHub。

## 2026-10-04 完成读回与下一步

原轮次正常完成，child rc=0，耗时 146.3395 秒，12/12 generate、无 token limit 截断。development：0/4 exact、CER 5/28；locked_check：4/8 exact、CER 13/53。七文件保护通过，原始完整性清单重新逐字节核验。低正确率不能归因为程序失败；不为结果改提示词或换样本。

下一步是原始 base vs ALPR adapter 的固定 12 图配对诊断，不是新的独立测试。原 adapter 结果只读复用；新 base-only 进程仅加载同一 base，不挂 adapter；prompt、processor、CPU float32、seed、32-token 上限与样本顺序不变。新增最多 12 generate，1800 秒上限，不训练、下载、重新选图或自动重试。比较是看到 adapter 结果后决定开展的探索性分析，不进行显著性/泛化宣传。

准备入口：`scripts/run_public_alpr_pilot.py prepare_base --base-comparison`。
一次后台启动：`scripts/run_public_alpr_pilot.py launch --base-comparison`。
独立本机输出：`artifacts/analysis/alpr_public_base_comparison_20261004_v1/`。
消费 `paired_comparison_redacted.json` 的 both-correct、adapter-only-correct、base-only-correct、both-wrong 计数；保留所有输赢，不选胜例进入后续证据。

审查元数据：reviewed_at=2026-10-04；literature_cutoff=2026-10-04（本轮不评价 novelty）；target_venue=IEEE TMC（研究目标，不是就绪结论）；artifact_run_id=alpr_public_sample_20261004_v1；policy_version=tmc_review_policy_v3_20260621；原科学 commit=b3e825c0f4fc75801e095ae5d7824c829152d9da；证据范围=局部任务原件核验，不晋级完整论文 E2，也不形成缓存/迁移/算法 claim。

## 配对完成与筛选决定

base-only 科学执行 commit `2a35f96`，child rc=0，85.916 秒，12/12 完成；原 adapter 为 146.340 秒。两轮都无 token-limit 截断。只读审计工具 `scripts/audit_public_alpr_pair.py` 用单独的二维编辑距离实现逐项复算，24 行评分 mismatch=0；两份完整性清单、输入 hash、模型目录清单、环境版本及生成设置一致。

| split | adapter exact | base exact | adapter CER | base CER |
|---|---:|---:|---:|---:|
| development | 0/4 | 3/4 | 5/28 | 1/28 |
| locked_check | 4/8 | 4/8 | 13/53 | 13/53 |

检查集是同样四图双方正确、同样四图双方错误；开发集三图仅 base 正确、一图双方错误。12 图中六个规范化输出相同，不能将汇总相同误写为所有输出相同。已消除大小写、空格和标点差异，剩余错误不是仅靠这一格式规范化可消除。没有独立重标图像，不能断定所有来源标签绝对正确。

观察到单次生成中位耗时 adapter=11.669 秒、base=6.803 秒；顺序运行、未随机化和重复测量，不作为稳健时延比较或因果开销结论。ALPR 模型卡明确说 unknown dataset，任务域和训练输入格式不明；这可能影响适配性，但根因尚未确证。本次缺少独立 active-tensor fingerprint，不把输出变化单独当成全面加载正确性的证明。

**决定：不将此 ALPR adapter+全图样本组合晋级为缓存收益实验的任务。** 保留其技术兼容和负向任务证据；不重新挑图、调提示词、重开所谓 holdout 或启动 RL。下一候选应先提供清楚的任务域/训练说明及可追溯标签，并通过同信息输入的任务正确性与 base 对照，再做缓存/迁移的匹配实验。当前 base 也仅 4/8 检查图正确，不认定为可靠部署模型。

新只读报告：主仓库 `artifacts/analysis/alpr_public_pair_review_20261004_v1/review_redacted.json` 与 `integrity.json`。本轮模型调用新增 0，未下载新数据或模型。评分与配对工具测试 7 passed；无训练、formal 或旧 holdout 操作。总体论文机制/算法优势仍未由此证据建立。
