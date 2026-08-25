# Typed Model-Cache Formal Execution Environment Contract

- `formal_execution_environment_contract_version`: `1.0.0`
- `execution_environment_resolver_version`: `1.0.0`
- Protocol：`typed_model_cache_formal_protocol_version=1.4.0`
- 状态：`READY_FOR_G14C_V5_CLEAN_TRAIN_AND_FORMAL`

## Identity 与 runtime location

科学 identity 包含 Python implementation/version、platform system/architecture、installed-package
fingerprint、Torch/关键库版本、Commit A5 逻辑绑定与 source-root identity，并计算 environment fingerprint。
它不包含 host-specific Python、venv、site-packages、clean worktree 或 cwd 绝对路径。后者只进入 runtime
audit；因此共享虚拟环境 relocation 不改变科学 identity。

Resolver 优先级固定为：显式 `--python-executable`、显式 environment manifest、当前 runner
`sys.executable`、协议允许候选。解释器必须存在且可执行，版本与 dependency fingerprint 必须匹配；全部
preflight/tests/train/dev/formal/support/statistics/integrity/gate 子命令使用同一绝对解释器。冻结模板只使用
`{python_executable}`，不再含相对 `.venv/bin/python`。

## Clean import gate

子进程固定 `PYTHONPATH=<clean_worktree_root>`、`PYTHONNOUSERSITE=1`，并记录 `sys.executable`、
`sys.version`、`sys.path`、`src.__file__`、关键模块 origin、Torch 版本、cwd 与 execution binding。
`src` 和项目模块必须位于 clean worktree；依赖可以位于共享 venv。任何项目 import 指向 clean root 外部，
包括主 dirty worktree或 editable-install 优先路径，立即失败。

G14R4+ no-.venv rehearsal 中 resolved Python 为主 worktree 共享 venv，但 `src` 与 formal execution module
均从 `/private/tmp/ppo_mec_g14r4_clean_git_20260825` 加载；child parity 与 environment fingerprint 通过。
主机绝对路径只作为本次 runtime audit，不是 protocol semantic identity。

## 边界

不允许用手工 symlink 修复正式合同。两个 G14C v4 run 均不能使用新 resolver 事后 salvage。环境 readiness
不是正式性能证据；holdout 仍 sealed/unopened。
