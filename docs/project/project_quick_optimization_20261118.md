# 项目运行效率与数据预检优化

日期：2026-11-18 Asia/Shanghai。用户明确本轮覆盖项目本身；本记录补足上一轮偏重论文管理的范围。实现基线 Git `4a25030`；本轮只修改两个 trainer、数据就绪脚本和独立测试，原七个算法/profile/兼容性未提交文件保持不变。

## 已落地的改动

### 采样复用已返回的 value

`OnPolicyTrainer` 和 `MARLOnPolicyTrainer` 原来用 `action_info.get("value", self._estimate_value(...))` 取值。Python 会先执行默认参数，即使动作结果已有 value，也会再次调用价值网络。现改为先检查 key：有 value 时直接使用，没有时才调用 fallback；`0.0` 也正确复用。终止/截断及下一状态 bootstrap 路径保留。

与固定基线 trainer 在相同 toy 环境、seed=7、默认 agent 下对照：

| agent | episode 步数 | evaluate_value 调用：修改前 → 后 |
| --- | ---: | ---: |
| PPO | 6 | 6 → 0 |
| MAPPO | 6 | 11 → 5 |
| SA-GHMAPPO | 6 | 11 → 5 |

三种方法的 observation、next_observation、动作、奖励、终止/截断、value/next_value、log_prob、advantage、return 逐项一致；相同更新随机种子下，一次真实 `learn()` 后的全部网络参数也逐项相等。五次短运行计时仅保留在本地回执，不据此外推长训练加速比或算法效果。测试另覆盖缺少 value、零 value、非零 value、终止与截断 bootstrap。

### 主线数据预检可用于运行门禁

`scripts/check_data_ready.py` 新增：

- `--scope mainline`：只执行 NGSIM + Alibaba 检查，不读取 LuST/highD/HF metadata。
- `--strict`：所选检查任一未就绪时退出码为 1，否则为 0。
- `--json`：stdout 仅输出原有 `{"results": [...]}` 结构，便于脚本消费。

NGSIM/Alibaba 检查拒绝空文件、目录、不可读文件及未还原的 Git LFS 指针；NGSIM 检查默认排序首个 CSV，与实际加载器一致，不默默跳到另一个有效文件。最多读取文件开头 256 字节，不全量扫描、下载、解压或覆盖数据。默认无参数仍检查六项并返回 0，保留历史盘点模式。

此门禁只检查本地文件可用性，不验证 CSV schema、数据版本、完整性哈希或正式 split。实际解析仍用 sample/dry-run；显式传入其他数据路径时须单独验证该路径，不把默认目录检查当作其验收。

## 验证

```bash
.venv/bin/python -m pytest tests/test_trainer_value_reuse.py tests/test_data_ready_cli.py tests/test_env_contract.py tests/test_checkpoint_compat.py tests/test_model_cache_dataset_registry.py tests/test_algo_pool_contract.py -q
.venv/bin/python scripts/smoke_test.py
.venv/bin/python scripts/check_data_ready.py --scope mainline --strict --json
.venv/bin/python scripts/run_ngsim_sample.py --max_rows 500
.venv/bin/python scripts/run_alibaba_sample.py --limit_jobs 3 --min_tasks 5 --max_tasks 20
.venv/bin/python scripts/run_real_sample_dryrun.py --mobility_source ngsim --workflow_source alibaba --max_mobility_rows 1500 --max_workflows 3 --workflow_selector ordered --rsu_layout auto_dominant_tight --window_selector max_handoff_candidate --window_length 24 --max_steps 12
git diff --check
```

- 工作区六组测试 309 项通过；独立导出 `4a25030`、只叠加本轮五个代码/测试文件后，同六组 305 项通过。差四项是未纳入的原有算法候选测试，不依赖原 dirty 修改才能通过。
- smoke 完成六节点；NGSIM 读取 500 条车辆记录、Alibaba 返回 3 个 workflow；真实 dry-run 的九节点 DAG 在 10 步内完成，仅用于链路联调。
- 主线严格预检退出 0；缺失/空文件/目录/不可读/LFS 指针、默认文件选择、可选源隔离及历史无参数模式均有测试。
- 语法检查通过：`PYTHONPYCACHEPREFIX=/private/tmp/ppo_mec_quick_pycache_20261118 .venv/bin/python -m py_compile scripts/check_data_ready.py src/trainers/on_policy_trainer.py src/trainers/marl_on_policy_trainer.py`。缓存写入临时目录；默认 macOS 全局缓存不可写不是源码错误。

本地证据：`artifacts/analysis/project_quick_optimization_20261118_v1/` 的 `baseline_trainer_probe.json`、`trainer_comparison.json`、`clean_verification.json`。原始材料不进入 Git。原七个 dirty 文件及论文关键文件的 SHA-256 保护检查通过。

## 未覆盖与后续

本轮未变更 reward、action/observation/checkpoint schema、网络结构或 baseline 权限；只做 toy 一次更新对照，没有长训练或正式 benchmark，不能据此宣称回报提高或收敛。机制状态预测、v122 候选闭环和完整实验归档继续按[项目优化清单](project_optimization_backlog.md)推进，论文确认另由论文台账管理。
