# 开发人员指南

欢迎成为 `jorek_postproc` 项目的贡献者！本文档将帮助您了解项目的开发流程、最佳实践和工作方式。

## 目录

- [项目概述](#项目概述)
- [开发环境设置](#开发环境设置)
- [项目结构](#项目结构)
- [开发工作流](#开发工作流)
- [编码规范](#编码规范)
- [测试指南](#测试指南)
- [文档编写](#文档编写)
- [调试技巧](#调试技巧)
- [性能优化](#性能优化)

## 项目概述

`jorek_postproc` 是一个用于处理和可视化 JOREK 核聚变模拟数据的 Python 包。

**主要目标**：
- 简化 JOREK 输出数据的处理
- 提供高质量的数据可视化
- 支持多种设备位形配置
- 提供易用的命令行和 Python API

**技术栈**：
- Python 3.7+
- NumPy：数值计算
- Matplotlib：数据可视化
- SciPy：科学计算
- Click：命令行界面

## 开发环境设置

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/jorek_postproc.git
cd jorek_postproc
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用 venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或
venv\Scripts\activate  # Windows

# 或使用 conda
conda create -n jorek_postproc python=3.10
conda activate jorek_postproc
```

### 3. 安装开发依赖

```bash
# 安装包和开发依赖
pip install -e .
pip install -r requirements-dev.txt

# 安装预提交钩子
pre-commit install
```

### 4. 验证安装

```bash
# 验证包可以导入
python -c "import jorek_postproc; print(jorek_postproc.__version__)"

# 运行诊断
python -c "from jorek_postproc import run_diagnostics; run_diagnostics(verbose=True)"

# 运行测试
pytest tests/ -v
```

## 项目结构

```
jorek_postproc/
├── __init__.py              # 包初始化和 API 导出
├── __version__.py           # 版本信息
├── config.py                # 配置和参数解析
├── cli.py                   # 命令行界面
├── data_models.py           # 数据模型和数据类
├── diagnostics.py           # 诊断和验证工具
├── example.py               # 使用示例
├── geometry.py              # 几何和位形定义
├── io.py                    # 文件 I/O 操作
├── logging.py               # 日志配置
├── plotting.py              # 绘图和可视化
├── processing.py            # 数据处理和转换
├── reshaping.py             # 数据重构和插值
└── PACKAGE_STRUCTURE.md     # 包结构文档

tests/
├── conftest.py              # pytest 配置和共享固件
├── test_basic.py            # 基本功能测试
├── test_cli.py              # CLI 测试
├── test_io.py               # I/O 操作测试
├── test_plotting.py         # 绘图功能测试
├── test_processing.py       # 处理功能测试
└── test_reshaping.py        # 重塑功能测试

docs/                         # 文档（计划中）
├── index.rst
├── api/
├── guide/
└── examples/

root files/
├── pyproject.toml           # 现代包配置
├── CHANGELOG.md             # 版本历史
├── CONTRIBUTING.md          # 贡献指南
├── RELEASE_CHECKLIST.md     # 发布清单
├── DEVELOPER_GUIDE.md       # 本文件
├── requirements-dev.txt     # 开发依赖
├── tox.ini                  # 多环境测试配置
├── .flake8                  # 代码质量配置
├── .pre-commit-config.yaml  # 预提交钩子配置
└── README.md                # 项目首页
```

## 开发工作流

### 1. 创建分支

```bash
# 从 develop 分支创建特性分支
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name

# 分支命名规范
# feature/xxx      - 新功能
# fix/xxx          - 问题修复
# docs/xxx         - 文档更新
# refactor/xxx     - 代码重构
# test/xxx         - 测试相关
```

### 2. 开发和提交

```bash
# 编辑代码
# 运行预提交钩子检查
pre-commit run --all-files

# 或让 git 自动运行
git add .
git commit -m "feat: add new feature"

# 提交消息格式（Conventional Commits）
# feat: 新功能
# fix: 修复
# docs: 文档
# style: 格式
# refactor: 重构
# perf: 性能
# test: 测试
# chore: 杂项
```

### 3. 推送和 Pull Request

```bash
git push origin feature/your-feature-name

# 在 GitHub 上创建 PR
# - 选择 develop 作为目标分支
# - 写清楚描述
# - 链接相关 Issue
```

### 4. 代码审查

- 等待其他维护者审查
- 根据反馈进行修改
- 保持对话清晰
- PR 批准后合并

## 编码规范

### 代码风格

遵循 PEP 8 和项目约定：

```python
# 良好的代码示例
import numpy as np
from typing import Tuple, List

def process_data(
    data: np.ndarray,
    scale: float = 1.0,
    method: str = 'linear'
) -> Tuple[np.ndarray, dict]:
    """
    处理数据的简短描述。
    
    更详细的描述...
    
    Args:
        data: 输入数据数组
        scale: 缩放因子，默认 1.0
        method: 处理方法，'linear' 或 'cubic'
    
    Returns:
        处理后的数据和元数据字典
    
    Raises:
        ValueError: 如果方法无效
    
    Examples:
        >>> data = np.random.rand(100)
        >>> processed, info = process_data(data, scale=2.0)
    """
    if method not in ['linear', 'cubic']:
        raise ValueError(f"Unknown method: {method}")
    
    processed = data * scale
    metadata = {'scale': scale, 'method': method}
    
    return processed, metadata
```

### 关键约定

1. **导入顺序**：标准库 → 第三方 → 本地导入
2. **行长度**：最多 100 个字符（由 Black 强制）
3. **命名**：
   - 类：`PascalCase`
   - 函数/变量：`snake_case`
   - 常量：`UPPER_CASE`
4. **类型提示**：使用类型注解
5. **文档字符串**：使用 Google 风格

### 代码检查

自动运行检查：

```bash
# 代码风格（Black）
black jorek_postproc tests

# 导入排序（isort）
isort jorek_postproc tests

# 代码质量（Flake8）
flake8 jorek_postproc tests --max-line-length=100

# 类型检查（MyPy）
mypy jorek_postproc --ignore-missing-imports

# 代码复杂度（Pylint）
pylint jorek_postproc
```

## 测试指南

### 编写测试

```python
# tests/test_example.py
import pytest
from jorek_postproc import some_function

class TestSomeFunction:
    """测试 some_function"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        result = some_function(5)
        assert result == 10
    
    def test_edge_case(self):
        """测试边界情况"""
        with pytest.raises(ValueError):
            some_function(-1)
    
    @pytest.mark.parametrize('input,expected', [
        (1, 2),
        (5, 10),
        (10, 20),
    ])
    def test_parametrized(self, input, expected):
        """参数化测试"""
        assert some_function(input) == expected
```

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_example.py::TestSomeFunction::test_basic_functionality

# 运行并显示覆盖率
pytest tests/ --cov=jorek_postproc --cov-report=html

# 运行特定标记的测试
pytest -m "not slow"

# 多环境测试
tox
```

### 测试约定

- 测试文件以 `test_` 开头
- 测试类以 `Test` 开头
- 测试方法以 `test_` 开头
- 目标：≥80% 代码覆盖率
- 使用固件共享测试数据（见 `conftest.py`）

## 文档编写

### 模块文档

```python
"""模块简短描述。

更详细的说明...

Example:
    使用此模块的基本示例::

        from jorek_postproc import module_name
        result = module_name.function()

Attributes:
    CONSTANT_NAME: 常量说明
"""
```

### 函数文档

遵循 Google 风格指南：

```python
def function_name(param1, param2):
    """简短一句话描述。
    
    更详细的描述，如果需要的话...
    
    Args:
        param1: 参数1说明
        param2: 参数2说明
    
    Returns:
        返回值说明
    
    Raises:
        ExceptionType: 异常说明
    
    Examples:
        >>> function_name(1, 2)
        3
    """
    pass
```

### 类文档

```python
class ClassName:
    """类描述。
    
    详细说明...
    
    Attributes:
        attr1: 属性说明
        attr2: 属性说明
    
    Example:
        >>> obj = ClassName()
        >>> obj.method()
    """
    
    def __init__(self, attr1, attr2):
        """初始化说明"""
        self.attr1 = attr1
        self.attr2 = attr2
```

## 调试技巧

### 使用日志

```python
from jorek_postproc.logging import setup_logging

# 设置日志
logger = setup_logging(verbose=True)

# 在代码中使用
logger.debug("调试信息")
logger.info("信息")
logger.warning("警告")
logger.error("错误")
```

### 使用 Python 调试器

```python
import pdb

def problematic_function():
    x = 5
    pdb.set_trace()  # 程序会在此暂停
    y = x + 2
    return y

# 调试命令：
# n - 下一行
# s - 进入函数
# c - 继续执行
# p variable - 打印变量
# h - 帮助
```

### 使用 IPython

```bash
pip install ipython

# 在代码中使用
from IPython import embed
embed()  # 启动交互式调试会话
```

## 性能优化

### 分析性能

```bash
# 使用 cProfile
python -m cProfile -s cumulative script.py

# 使用 line_profiler
pip install line_profiler
kernprof -l -v script.py

# 使用 memory_profiler
pip install memory_profiler
python -m memory_profiler script.py
```

### 优化建议

1. **使用向量化操作**：NumPy 优于 Python 循环
2. **避免不必要的复制**：使用视图而不是副本
3. **缓存结果**：使用 `functools.lru_cache`
4. **并行处理**：考虑使用 `concurrent.futures` 或 `joblib`
5. **Cython**：关键代码可用 Cython 加速

## 常见任务

### 添加新模块

1. 在 `jorek_postproc/` 创建新文件
2. 在 `__init__.py` 中导出主要函数/类
3. 在 `tests/` 中创建对应测试
4. 更新 `PACKAGE_STRUCTURE.md`
5. 添加使用示例到 `example.py`

### 修复 Bug

1. 创建 `fix/bug-name` 分支
2. 编写测试重现 bug
3. 修复代码
4. 验证测试通过
5. 更新 CHANGELOG
6. 创建 PR

### 添加新功能

1. 创建 `feature/feature-name` 分支
2. 编写设计文档
3. 实现功能和测试
4. 编写文档和示例
5. 更新 CHANGELOG
6. 创建 PR

## 获取帮助

- **GitHub Issues**：报告 bug 或请求功能
- **GitHub Discussions**：讨论设计和想法
- **Email**：Allen Cheng (Allencheng@buaa.edu.cn)

## 许可证

本项目采用 MIT 许可证。贡献代码即表示同意此许可证。

---

感谢您对 `jorek_postproc` 的贡献！
