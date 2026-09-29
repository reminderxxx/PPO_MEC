# 驾驶小实验资源准备记录

2026-09-29，用户明确允许下载。基线commit为3040a5f；本轮无科学推理或训练。

## 模型与隔离环境

`scripts/download_driving_pilot_model.py`按固定revision与13个根目录文件allowlist下载，
不会取得ONNX全仓或执行模型。元数据总量上限1,030,000,000 bytes，权重SHA/size必须匹配；
小文件核验Git blob身份，逐文件另记录SHA256。未通过校验的文件保留.partial，不成为完成文件。
没有自动下载重试或旧文件覆盖；输出root必须不存在。

本轮模型位置：`data/raw/ai_workflow_pilot/model_a7da5b9_20260929/`。
完成/失败回执为该目录`download_receipt.json`；status=complete且13项哈希读回通过才算取得模型。
本记录写入时权重下载仍进行中；12个配套文件已验证，不能把已有.partial算作完成。
入口进程PID56983、工具会话55214仅为本次观察，不是稳定身份或自动重启依据。

独立环境：`artifacts/environments/driving_pilot_py39_v1/`，由现有Python创建新venv，不启用system-site-packages。
依赖版本见`configs/experiment/driving_pilot_requirements.txt`；通过PyPI安装，旧`.venv`未安装或修改。
安装进程PID58105、工具会话33236；当时仍运行。安装完成后需pip check、实际import及离线模型加载，
不以安装命令已启动代替可用性。全部原始文件/环境均排除Git，不能push权重或原图。

后续同轮验收：安装已成功退出0，`pip check`无冲突；在HF/Transformers offline模式下，
AutoProcessor实际读取本地配置成为Idefics3Processor，AutoConfig.model_type=idefics3，
Torch2.8.0/Transformers4.49.0/MPS可用且sys.prefix为独立环境。权重未加载，推理0次。
记录一项环境警告：系统Python使用LibreSSL2.8.3，urllib3 v2给出NotOpenSSLWarning；
离线processor读取通过不证明该HTTPS栈兼容。正式模型入口应保持offline，联网下载本轮使用stdlib urllib。

## 图像条款：核清但等待负责人接受

本轮通过浏览器读到完整[nuScenes条款](https://www.nuscenes.org/terms-of-use)，页面标注2021-11-16更新。
它规定从网站或其他来源下载/使用均受CC BY-NC-SA 4.0及附加条款约束，包含免责声明、赔偿及争议解决。
此前仅文本抓取空壳的问题已解决，但用户的“允许下载”不被擅自记录为已接受刚展示的完整条款。
已向负责人单独展示链接与条件；在确认前图像下载为0，未代登录、点击同意或使用gated账户。
本轮不宣称可再分发或法律合规审查已完成。

## 验证与后续

- `.venv/bin/python -B -m pytest tests/test_driving_pilot_model_download.py tests/test_drivelm_pilot_plan.py tests/test_drivelm_workflow_preview.py tests/test_drivelm_workflow_sample_audit.py -q`：54 passed。
- `.venv/bin/python -B scripts/smoke_test.py`：6/6 toy节点完成。
- `git diff --check`通过；七个保护文件SHA与既有基线一致。
- 新下载器及测试编译通过；独立环境`pip check`和offline processor/config加载通过。
- 未执行真实模型、图像解码、16-call pilot、完整仓库测试或旧holdout。

下一次集中检查download_receipt及环境终态；不反复轮询下载进度，不因失败覆盖.partial或冒充完成。
模型字节校验、环境实载、图像许可与输入解码通过后才进入已设计的有界小实验。
