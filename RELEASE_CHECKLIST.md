# 发布检查清单

## 发布流程文档

此文档详细说明了 `jorek_postproc` 包的发布流程和检查清单。

## 前置条件

- [ ] 拥有 PyPI 账户并配置 API 令牌
- [ ] 拥有 GitHub 仓库的维护者权限
- [ ] 本地 Git 仓库已更新到最新代码

## 版本发布步骤

### 1. 准备代码更改

- [ ] 确保所有功能已实现并测试完成
- [ ] 提交所有改动到开发分支
- [ ] 创建 Pull Request 到 `main` 分支
- [ ] 确保所有 CI/CD 检查通过
- [ ] 获得代码审查批准
- [ ] 合并 PR 到 `main` 分支

### 2. 更新版本号

- [ ] 更新 `jorek_postproc/__version__.py` 中的版本号
  ```python
  __version__ = "X.Y.Z"  # 使用语义化版本
  ```
- [ ] 确保版本号遵循 [语义化版本](https://semver.org/lang/zh-CN/) 规范
  - 主版本号 (X)：不兼容的 API 改动
  - 次版本号 (Y)：向下兼容的功能新增
  - 修订号 (Z)：向下兼容的问题修复

### 3. 更新 CHANGELOG

- [ ] 添加新版本条目到 `CHANGELOG.md`
- [ ] 按如下格式组织更新内容：
  ```markdown
  ## [X.Y.Z] - YYYY-MM-DD
  
  ### Added
  - 新增功能说明
  
  ### Changed
  - 改动说明
  
  ### Fixed
  - 修复说明
  
  ### Deprecated
  - 弃用说明
  ```
- [ ] 检查所有改动是否准确记录
- [ ] 将 [Unreleased] 部分清空或保留用于下一个版本

### 4. 更新项目元数据

- [ ] 检查 `pyproject.toml` 中的项目信息
- [ ] 更新任何相关的文档
- [ ] 检查 `README.md` 中的安装说明

### 5. 代码质量检查

- [ ] 运行本地测试：`pytest tests/ -v`
- [ ] 运行覆盖率分析：`pytest tests/ --cov=jorek_postproc --cov-report=html`
- [ ] 运行代码质量检查：`tox`
- [ ] 确保所有代码质量指标达标：
  - [ ] Flake8 无错误
  - [ ] Black 代码格式一致
  - [ ] MyPy 类型检查通过
  - [ ] Pylint 评分 ≥ 7.0
- [ ] 运行预提交钩子检查：`pre-commit run --all-files`

### 6. 标记和发布

- [ ] 创建 Git 标签：
  ```bash
  git tag -a vX.Y.Z -m "Release version X.Y.Z"
  ```
- [ ] 推送标签到 GitHub：
  ```bash
  git push origin vX.Y.Z
  ```
- [ ] 在 GitHub Releases 页面创建新发布：
  - [ ] 选择刚刚推送的标签
  - [ ] 发布标题：`Release vX.Y.Z`
  - [ ] 发布描述：复制 CHANGELOG 中的相应内容
  - [ ] 标记为最新版本（如果适用）

### 7. 自动发布到 PyPI

- [ ] 确保 GitHub Actions 工作流已配置
- [ ] 检查发布工作流执行情况
- [ ] 验证包已成功发布到 PyPI：
  ```bash
  pip install jorek_postproc==X.Y.Z
  ```

### 8. 发布后验证

- [ ] 验证新版本在 PyPI 上可用：https://pypi.org/project/jorek_postproc/
- [ ] 测试从 PyPI 安装：
  ```bash
  pip install --upgrade jorek_postproc
  python -c "import jorek_postproc; print(jorek_postproc.__version__)"
  ```
- [ ] 验证包导入和基本功能正常
- [ ] 检查 GitHub Releases 页面

### 9. 发布通知

- [ ] 更新项目网站（如有）
- [ ] 通过电子邮件通知用户列表（如有）
- [ ] 在项目讨论版/论坛公布新版本（如有）
- [ ] 考虑发布博客文章或发行说明（对于主要版本）

## 版本号策略

按照 [语义化版本 2.0.0](https://semver.org/lang/zh-CN/) 规范：

### 主版本号增加 (X.0.0)
- 发布包含不兼容改动的版本
- 已弃用的功能被移除
- 现有 API 的行为发生重大改变

### 次版本号增加 (Y.0.0)
- 新增向下兼容的功能
- 新增功能设计为扩展现有功能
- 改进既有功能（向下兼容）

### 修订号增加 (Z)
- 修复向下兼容的缺陷
- 性能优化和改进
- 文档更新

## 快速发布命令

```bash
# 1. 更新版本号
# 编辑 jorek_postproc/__version__.py

# 2. 更新 CHANGELOG
# 编辑 CHANGELOG.md

# 3. 提交更改
git add -A
git commit -m "chore: release version X.Y.Z"

# 4. 创建标签
git tag -a vX.Y.Z -m "Release version X.Y.Z"

# 5. 推送代码和标签
git push origin main
git push origin vX.Y.Z

# 6. GitHub Actions 将自动发布到 PyPI
```

## 回滚发布

如果发布后发现严重问题：

1. 在 PyPI 上标记版本为已取消使用（Yanked）
2. 立即发布补丁版本（X.Y.Z+1）修复问题
3. 在 CHANGELOG 中说明问题和修复方案
4. 通知用户升级到修复版本

## CI/CD 自动化

### GitHub Actions 工作流

#### tests.yml
- 在每次推送和 PR 时运行
- 在多个 Python 版本（3.7-3.11）和操作系统上测试
- 执行代码质量检查
- 上传覆盖率报告到 Codecov

#### publish.yml
- 在创建 GitHub Release 时自动触发
- 或通过手动工作流调度触发
- 构建并验证分发包
- 上传到 PyPI

## 文档更新

发布新版本时应更新：

- [ ] 版本号：`__version__.py`
- [ ] 更新日志：`CHANGELOG.md`
- [ ] 在线文档（如 Sphinx 文档）
- [ ] 快速开始指南（如版本有重大改动）
- [ ] API 文档（如 API 有改动）

## 常见问题

### Q: 如果 CI/CD 检查失败，我应该怎样做？
A: 检查失败的工作流日志，修复代码问题，推送修复后的代码，再次尝试发布。

### Q: 如何为开发版本发布？
A: 在版本号中使用 pre-release 标识符，例如 `0.2.0-alpha.1` 或 `0.2.0-beta.1`

### Q: PyPI 上的包需要多久才能更新？
A: 通常在几分钟内，但可能需要最多 15 分钟才能完全同步到全球 CDN。

## 联系方式

对于发布相关的问题，请联系：
- **包维护者**：Allen Cheng (Allencheng@buaa.edu.cn)
- **GitHub Issues**：在项目仓库中报告问题
- **讨论**：使用 GitHub Discussions 进行讨论

## 相关文档

- [CHANGELOG.md](../CHANGELOG.md) - 版本历史
- [CONTRIBUTING.md](../CONTRIBUTING.md) - 贡献指南
- [pyproject.toml](../pyproject.toml) - 项目配置
