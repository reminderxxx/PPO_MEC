# Git 工作流规范

用途：PPO_MEC 项目的 Git 版本控制规范，确保代码变更可追溯、可审查、可回滚。

## 分支策略

### 主分支

- `main`：稳定主线，只接受通过验证的代码
  - 所有提交必须通过测试验证
  - 禁止直接推送，必须通过 PR/MR 合并
  - 每次合并后打标签记录版本

### 开发分支

- `develop`：日常开发集成（可选，单人项目可省略）
  - 功能完成并验证后合并到 main

### 临时分支

- `feature/<name>`：新功能开发
- `fix/<name>`：Bug 修复
- `exp/<name>`：实验性改动
- `docs/<name>`：文档更新

命名规范：
- 使用小写 `snake_case`
- 名称简洁描述改动目的，如 `feature/controller_mat_baseline`、`fix/memory_leak`

## 提交规范

### 提交信息格式

```
<type>: <subject>

<body>

<footer>
```

### 类型（type）

| 类型 | 用途 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat: 新增 Controller-MAT baseline` |
| `fix` | Bug 修复 | `fix: 修复 handoff 状态迁移错误` |
| `docs` | 文档更新 | `docs: 更新 RUNBOOK 训练命令` |
| `style` | 代码格式 | `style: 统一缩进为 4 空格` |
| `refactor` | 重构 | `refactor: 简化 action mask 逻辑` |
| `test` | 测试相关 | `test: 新增 checkpoint 兼容性测试` |
| `chore` | 构建/工具 | `chore: 更新 requirements.txt` |
| `exp` | 实验性改动 | `exp: 尝试新的 reward shaping` |

### 主题（subject）

- 不超过 50 个字符
- 使用祈使句，如"新增"、"修复"、"更新"
- 结尾不加句号

### 正文（body）

- 可选，用于详细说明
- 每行不超过 72 个字符
- 说明改动的动机和与之前行为的对比

### 页脚（footer）

- 可选，用于引用 Issue 或 Breaking Change
- `Breaking Change:` 标记不兼容改动

### 示例

```
feat: 新增 controller_mat baseline 支持

实现 controller-level Multi-Agent Transformer baseline，包括：
- 新增 mat_agent.py 实现
- 新增 controller_mat.yaml 配置
- 接入 registry 和 learned suite
- 通过 smoke test 验证

与 PPO/MAPPO 使用相同的 controller-level contract，
但采用 transformer-based critic 替代 MLP critic。

验证命令：
python scripts/train_algo_pool_real_sample.py --agent_name controller_mat --profile smoke
```

## 提交前检查清单

### 必需检查项

- [ ] 代码可以运行，无语法错误
- [ ] 相关测试通过
- [ ] `git status` 检查只包含本次改动相关文件
- [ ] 未跟踪文件、数据、checkpoint、缓存未混入提交
- [ ] 提交信息符合规范

### 代码改动检查项

- [ ] 最小改动原则：只改必要文件
- [ ] 新增文件使用 `snake_case` 命名
- [ ] 未重构无关模块
- [ ] 未移动已归档数据或历史产物

### 文档同步检查项

- [ ] 修改 schema/manifest/checkpoint/接口字段时，检查生产端和消费端
- [ ] 修改目录结构/主入口/产物路径时，更新 README.md、DIRECTORY_STRUCTURE.md、RUNBOOK.md
- [ ] 修改模块职责或依赖方向时，更新 CODE_MODULE_MAP.md
- [ ] 涉及长期设计取舍时，更新 DECISION_LOG.md

### 验证执行检查项

- [ ] 语法或 import 检查通过
- [ ] 最小 smoke 通过：`python scripts/smoke_test.py`
- [ ] 环境契约测试通过：`python -m pytest tests/test_env_contract.py`
- [ ] 目标链路局部验证通过（如适用）

## 提交流程

### 标准流程

```bash
# 1. 检查当前状态
git status

# 2. 查看改动内容
git diff

# 3. 添加本次改动文件（不要 git add .）
git add <specific-files>

# 4. 再次确认状态
git status

# 5. 提交
git commit -m "type: subject"

# 6. 推送到远程
git push origin <branch>
```

### 功能开发流程

```bash
# 1. 从 main 创建功能分支
git checkout -b feature/<name> main

# 2. 开发并提交（可多次提交）
git add <files>
git commit -m "feat: xxx"

# 3. 开发完成后，rebase 到最新 main
git fetch origin
git rebase origin/main

# 4. 推送分支
git push origin feature/<name>

# 5. 创建 PR/MR（如适用）
# 6. 审查通过后合并到 main
```

### 紧急修复流程

```bash
# 1. 从 main 创建修复分支
git checkout -b fix/<name> main

# 2. 修复并提交
git add <files>
git commit -m "fix: xxx"

# 3. 验证修复
# 4. 快速合并到 main（可省略 PR 流程）
git checkout main
git merge fix/<name>
git push origin main

# 5. 删除修复分支
git branch -d fix/<name>
```

## 版本标签

### 标签格式

- `v<major>.<minor>.<patch>`，如 `v1.2.3`
- 重大版本：不兼容的 API 变更
- 次要版本：向下兼容的功能新增
- 补丁版本：向下兼容的问题修复

### 打标签流程

```bash
# 1. 确保在 main 分支且代码已推送
git checkout main
git pull origin main

# 2. 创建标签
git tag -a v1.2.3 -m "版本 1.2.3：新增 Controller-MAT baseline"

# 3. 推送标签到远程
git push origin v1.2.3

# 4. 推送所有标签
git push origin --tags
```

### 标签场景

| 场景 | 标签示例 | 说明 |
|------|----------|------|
| 论文投稿 | `v2026.05.13_submission` | 标记投稿版本 |
| 实验冻结 | `v2026.05.10_exp_freeze` | 标记实验协议冻结 |
| 审稿回复 | `v2026.05.20_rebuttal` | 标记审稿回复版本 |
| 正式发布 | `v1.0.0` | 项目正式发布 |

## 冲突解决

### 预防冲突

- 频繁从 main 拉取更新：`git pull origin main`
- 功能分支生命周期尽量短
- 大改动先沟通再实施

### 解决步骤

```bash
# 1. 拉取最新代码
git fetch origin

# 2. 尝试 rebase
git rebase origin/main

# 3. 如有冲突，编辑冲突文件
# 冲突标记：<<<<<<< HEAD / ======= / >>>>>>> branch

# 4. 标记冲突已解决
git add <resolved-files>

# 5. 继续 rebase
git rebase --continue

# 6. 如需放弃 rebase
git rebase --abort
```

## 历史管理

### 查看历史

```bash
# 简洁历史
git log --oneline -20

# 图形化分支历史
git log --oneline --graph --all -20

# 查看具体提交
git show <commit-hash>

# 查看文件历史
git log -p <file-path>
```

### 撤销操作

```bash
# 撤销工作区改动（未 add）
git checkout -- <file>

# 撤销暂存区
git restore --staged <file>

# 撤销本地工作区改动
git restore <file>
```

### 清理未跟踪文件

清理前必须先预览，不直接执行破坏性删除：

```bash
git clean -nd
```

确认只包含缓存、临时文件或已明确不入库的本地报告后，才允许清理指定路径。不要使用宽泛的 `git clean -fdx` 清理真实数据、checkpoint、历史 artifact 或未归档报告。

## PPO_MEC 项目约束

- 提交前必须执行 `git status`，只 stage 本次任务相关文件。
- 禁止使用 `git add .` 混入真实数据、checkpoint、缓存或历史实验产物。
- 完成代码、配置、脚本或长期文档更新后，必须在匹配验证通过后提交并 push。
- 若 push 失败，最终记录必须说明失败命令、错误原因、本地 commit 状态和用户需要执行的后续命令。
- `artifacts/` 和 `data/` 默认不入库；论文结论只引用已在 `docs/project/ARTIFACT_RECORDS.md` 记录的正式产物。
- 新增长期文档优先放入 `docs/project/`，并在 `docs/project/README.md` 增加入口。

## 推荐提交粒度

- 文档清洗和结果审查：单独使用 `docs:` 提交。
- 代码修复：按问题域拆分，避免把无关重构、实验结果和文档草稿放在同一提交。
- 实验产物变更：通常只更新 manifest、报告和长期记录；不要提交大型 checkpoint 或原始数据。
