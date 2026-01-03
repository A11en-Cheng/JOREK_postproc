# ⚡ 快速开始指南

## 5 分钟快速上手 jorek_postproc

### 对于用户

#### 安装
```bash
pip install jorek_postproc
```

#### 基本使用
```python
from jorek_postproc import read_boundary_file, reshape_to_grid, plot_surface_3d

# 读取数据
data = read_boundary_file('boundary_quantities.dat')

# 重塑网格
grid_data = reshape_to_grid(data, grid_shape=(20, 30))

# 绘图
plot_surface_3d(grid_data)
```

#### 命令行使用
```bash
# 查看帮助
jorek-postproc --help

# 处理数据
jorek-postproc process input.dat output.dat
```

---

### 对于开发者

#### 1️⃣ 设置开发环境 (3 分钟)

```bash
# 克隆仓库
git clone https://github.com/yourusername/jorek_postproc.git
cd jorek_postproc

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# venv\Scripts\activate  # Windows

# 安装
pip install -e .
pip install -r requirements-dev.txt

# 安装预提交钩子
pre-commit install
```

#### 2️⃣ 运行测试 (1 分钟)

```bash
# 快速测试
pytest tests/ -v

# 覆盖率报告
pytest tests/ --cov=jorek_postproc
```

#### 3️⃣ 代码质量检查 (1 分钟)

```bash
# 全自动检查
pre-commit run --all-files

# 或单独运行
black jorek_postproc tests
flake8 jorek_postproc tests
mypy jorek_postproc
```

---

## 常见任务

### 添加新测试
```python
# tests/test_my_feature.py
import pytest
from jorek_postproc import my_function

def test_my_feature(sample_grid_data):
    """测试我的功能"""
    result = my_function(sample_grid_data)
    assert result is not None
```

运行: `pytest tests/test_my_feature.py -v`

### 添加新模块
```bash
# 1. 创建文件
touch jorek_postproc/my_module.py

# 2. 在 __init__.py 中导出
# from .my_module import my_function
# __all__.append('my_function')

# 3. 编写测试
# tests/test_my_module.py

# 4. 运行测试
pytest tests/test_my_module.py -v
```

### 创建 Pull Request
```bash
# 1. 创建分支
git checkout -b feature/my-feature

# 2. 编写代码和测试
# ... 编辑文件 ...

# 3. 提交
git add .
git commit -m "feat: add my feature"

# 4. 推送
git push origin feature/my-feature

# 5. 在 GitHub 创建 PR
```

### 发布新版本
```bash
# 1. 更新版本
# 编辑 jorek_postproc/__version__.py
__version__ = "0.2.0"

# 2. 更新 CHANGELOG.md
# 添加新的版本条目

# 3. 创建标签
git tag -a v0.2.0 -m "Release v0.2.0"

# 4. 推送
git push origin main v0.2.0

# 5. 完成！GitHub Actions 会自动发布到 PyPI
```

---

## 📚 更多信息

| 需要什么 | 看这个文档 |
|---------|-----------|
| 如何贡献 | `CONTRIBUTING.md` |
| 深入开发指南 | `DEVELOPER_GUIDE.md` |
| 如何发布版本 | `RELEASE_CHECKLIST.md` |
| API 使用 | `README.md` + 代码注释 |

---

## 🆘 故障排除

### 问: 测试失败，怎么办？
```bash
# 1. 检查环境
python -c "from jorek_postproc import run_diagnostics; run_diagnostics(verbose=True)"

# 2. 重新安装依赖
pip install -r requirements-dev.txt --upgrade

# 3. 运行单个失败的测试
pytest tests/test_file.py::test_name -vv

# 4. 查看详细错误
pytest tests/ -vv --tb=long
```

### 问: 代码风格检查失败？
```bash
# 自动修复大多数问题
black jorek_postproc tests
isort jorek_postproc tests

# 检查剩余问题
flake8 jorek_postproc tests
```

### 问: 导入错误？
```bash
# 确保包安装在开发模式
pip install -e .

# 验证包可导入
python -c "import jorek_postproc; print(jorek_postproc.__version__)"
```

---

## ✅ 常用命令速查表

```bash
# 开发
pytest tests/ -v                          # 运行测试
pytest tests/ --cov                       # 覆盖率报告
tox                                       # 多环境测试
pre-commit run --all-files                # 代码检查

# 代码质量
black jorek_postproc tests                # 格式化代码
isort jorek_postproc tests                # 排序导入
flake8 jorek_postproc tests               # 质量检查
mypy jorek_postproc                       # 类型检查
pylint jorek_postproc                     # 复杂度检查

# Git
git checkout -b feature/name              # 创建分支
git add . && git commit -m "msg"          # 提交
git push origin feature/name              # 推送
git tag -a vX.Y.Z -m "msg"                # 创建标签

# 包管理
pip install -e .                          # 开发模式安装
pip install -r requirements-dev.txt       # 安装开发依赖
pip install jorek_postproc                # 正式安装
```

---

## 🎯 下一步

- 📖 阅读完整的 `DEVELOPER_GUIDE.md`
- 🧪 查看 `tests/` 目录了解测试框架
- 💬 在 GitHub Issues 提问
- 🚀 开始贡献！

---

**快乐编码！** 🚀
