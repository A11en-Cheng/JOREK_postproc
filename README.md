# JOREK Post-Processing Package (`jorek_postproc`)

一个用于处理、分析和可视化 **JOREK 等离子体边界量数据** 的高可扩展 Python 工具包。通过此库，你可以将 JOREK 复杂的非结构化边界点云数据重整为结构化网格，并自动生成高质量的三维可视化图像与能量沉积分析报告。

## 核心功能

- 📊 **数据IO与重整化**：解析 `.dat` 格式边界量文件，支持重心排序与含 X 点（X-points）的复杂剖分。
- 🗼 **装置位形掩膜提取**：内置 **EXL50U**、**ITER** 真实装置几何掩膜，方便分部件观测（UO, LO, UI, LI等）。
- 🎨 **高级 3D 可视化**：支持表面图 (Surface) 与散点图 (Scatter)，自动标记最值，自由调节观测视角与渲染上限。
- ⚙️ **能量分析与终端CLI**：内建热通量运算功能，支持从命令行一键出图，流水线自动化处理多时间步。

## 安装指南

使用 `pip` 本地安装以进入开发模式：

```bash
# 确保你位于项目根目录，有 setup.py 和 pyproject.toml
pip install -e .
```

## 快速入口

- **完整说明书**: 请参阅 [`USAGE_GUIDE.md`](USAGE_GUIDE.md)。
- **示例代码**: 可运行 `python -m jorek_postproc.example` 快速验证。
- **CLI命令范例**:
  ```bash
  python -m jorek_postproc.cli -f boundary_data.dat -t 4200 -n heatF_tot_cd --log_norm
  ```

## 目录结构

- `jorek_postproc/`: 核心包，包含 IO、重整(Reshaping)、位形(Geometry)、绘图(Plotting) 等模块
- `tests/`: 单元测试，可通过 `pytest tests/` 运行
- `redundent/`: 用于存放历史临时开发和探索性脚本（即将弃用或仅作参考）
- `USAGE_GUIDE.md`: 本包的使用手册、API示例及CLI命令说明
