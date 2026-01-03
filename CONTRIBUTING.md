# 贡献指南

感谢对 jorek_postproc 的兴趣！本指南描述了如何对项目做出贡献。

## 行为准则

本项目遵循开源社区的行为准则。通过参与，您同意遵守这些准则。

## 如何贡献

### 报告Bug

在报告bug时，请提供以下信息：
- 清晰的bug描述
- 复现步骤
- 预期行为和实际行为
- Python版本、OS、依赖版本等环境信息
- 任何相关的代码片段或日志

### 提出功能建议

功能建议应该：
- 清晰地解释增强方案
- 提供尽可能多的细节和上下文
- 列出一些现有的类似功能（如果有）

### 提交拉取请求 (Pull Request)

1. **Fork仓库**
   ```bash
   git clone https://github.com/yourusername/jorek_postproc.git
   cd jorek_postproc
   ```

2. **创建开发分支**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **安装开发依赖**
   ```bash
   pip install -e ".[dev]"
   ```

4. **编写代码**
   - 遵循PEP8风格指南
   - 为新功能添加类型提示
   - 编写全面的docstring

5. **代码检查**
   ```bash
   # 格式化代码
   black jorek_postproc tests
   isort jorek_postproc tests
   
   # 检查风格
   flake8 jorek_postproc tests
   
   # 类型检查
   mypy jorek_postproc
   ```

6. **编写测试**
   ```bash
   # 在 tests/ 目录添加测试
   pytest tests/
   ```

7. **更新文档**
   - 更新相关的docstring
   - 更新README.md（如果需要）
   - 添加示例（如果适用）

8. **提交更改**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

9. **创建Pull Request**
   - 提供清晰的PR描述
   - 关联相关的issue
   - 确保CI/CD检查通过

## 开发工作流

### 环境设置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装包（开发模式）
pip install -e ".[dev]"
```

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_io.py

# 显示覆盖率
pytest --cov=jorek_postproc

# 在多个Python版本上测试
tox
```

### 代码风格

本项目使用以下工具：
- **black** - 代码格式化
- **isort** - import排序
- **flake8** - 风格检查
- **mypy** - 类型检查

### 提交消息格式

遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

类型：
- `feat` - 新功能
- `fix` - bug修复
- `docs` - 文档更改
- `style` - 代码风格更改（不影响功能）
- `refactor` - 代码重构
- `perf` - 性能改进
- `test` - 添加或更新测试
- `chore` - 其他更改（不修改src或test文件）

示例：
```
feat(geometry): add CFETR device support

Add mask definitions and viewing angles for CFETR tokamak.

Closes #123
```

## 项目结构

```
jorek_postproc/
├── jorek_postproc/          # 主包目录
│   ├── __init__.py
│   ├── data_models.py       # 数据模型
│   ├── io.py               # 文件I/O
│   ├── reshaping.py        # 数据重整化
│   ├── processing.py       # 数据处理
│   ├── geometry.py         # 装置定义
│   ├── plotting.py         # 可视化
│   ├── config.py           # 配置
│   ├── cli.py              # 命令行接口
│   ├── logging.py          # 日志
│   ├── diagnostics.py      # 诊断工具
│   └── __version__.py      # 版本
├── tests/                   # 测试目录
├── docs/                    # 文档目录
├── setup.py / pyproject.toml
└── ...
```

## 文档贡献

### 更新文档

1. 修改对应的 `.md` 文件或docstring
2. 确保语法正确
3. 在PR中清晰描述更改

### 编写示例

- 在 `example.py` 中添加新示例
- 确保示例可以独立运行
- 提供清晰的注释

## 发布流程

只有维护者可以发布新版本。流程包括：

1. 更新版本号
2. 更新CHANGELOG.md
3. 运行所有测试和检查
4. 创建Git标签
5. 构建分布包
6. 上传到PyPI

## 许可证

通过提交贡献，您同意您的贡献在 MIT 许可证下获得许可。

## 联系方式

- **邮件**：Allencheng@buaa.edu.cn
- **问题/讨论**：通过GitHub Issues
- **讨论区**：通过GitHub Discussions

---

感谢您的贡献！ 🎉
