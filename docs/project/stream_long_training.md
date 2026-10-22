# 统一预算的持续请求长训入口

本入口准备一个独立的开发实验：检验完整场景重复暴露后的学习表现。入口交付、观察器资格通过和更长预算本身均不表示算法不稳已修复。运行前必须独立回收诊断，并留下明确的干预理由与验证证据；不能仅因日志结构完整而启动。

方法顺序固定为 SA、MAPPO、PPO、DAG、去预测 SA、cache、DT，各使用 seed 7/17/27、504 会话、每4会话一次原 update，算法 `training_budget=504`。七个完整72-case轮转使用同一日程；从头训练，固定最终504模型，不按峰值选择、排名或尾稳失败追加预算。未修复的 DQN 不加入该比较。

训练10584会话、2646次learner调用，最多5080320物理帧。完整开发48条件在0/72/…/504共8点；原固定probe12在0/24/…/504共22点，重合检查点复用full48里的12行，不重复环境。实际唯一评估11592会话，最多5564160帧。两面板分别标识和作图：原probe12仅含family=1、capacity=4096，不代表完整面板。

主图依据固定full48，保留各seed及均值；probe仅辅助。原始训练回报及24会话移动均值可另行展示，不用任务组成变化证明收敛。开发面板只有原两段时间会话，条件数、seed和重复checkpoint均不增加独立时间cluster。尾部稳定性与初末性能分别报告；低性能平台不能称学习成功。

内部描述性尾稳规则事前固定在full48的360/432/504：reward相对极差不超过1%、请求完成率极差不超过1个百分点、汇总模型KiB/完成请求相对极差不超过2%，总体及三个seed分别检查；未定义分母判失败。这些是内部检查规则，不是数学收敛证明或期刊录用标准。

```text
python scripts/train_stream_long.py prepare --source_protocol <original_v2_manifest> --output_root <fresh_private_root>
python scripts/train_stream_long.py check --protocol <fresh_private_root>/manifest.json
python scripts/train_stream_long.py run --protocol <fresh_private_root>/manifest.json --decision_receipt <independent_root_decision.json>
```

prepare只读取并校验已有资产，不调用旧数据准备、原始数据下载或predictor拟合；check为标准库预检。run必须绑定清洁源码、协议SHA与独立诊断回收决策。原算法源码保持基线，具体决策须明确这是有界覆盖预算检验或已验证的非算法协议修复，不把它包装成已证实的算法修复。

单进程CPU一线程，最长七天、RSS最多8GiB，入口及各cell前剩余磁盘至少30GiB，另有逐调用和输出硬上限。21个learner、462个独立评估副本、22176个随环境构造的固定predictor分别计数，predictor拟合为零。评估保存后使用副本，完整恢复Python/NumPy/PyTorch RNG，并核对训练模型、双optimizer及控制器不变。

checkpoint保存前按16MiB上界预扣输出额度且不退款，并限制实际文件位置和多次保存的累计写入；流式输出和元数据按实际累计写入计费。该计数是保守资源额度，不是目录当前驻留大小。

输出仅写新的私有ignored artifacts；checkpoint、训练/开发原始行、最终事件/费用压缩日志及失败现场都保留。独占claim拒绝恢复、覆盖与自动重试。软件测试 `tests/test_train_stream_long.py` 使用合成或mock调用，不是科学训练验证；正式formal/holdout/canonical不由本入口晋级。
