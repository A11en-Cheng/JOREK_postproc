# jorek_postproc 包 - 使用说明

## 📋 概述

**jorek_postproc** 是一个为JOREK等离子体模拟输出数据设计的后处理包，专门用于处理和可视化**边界量**数据。

### 核心功能
- ✅ 读取JOREK边界量文件
- ✅ 将非结构化点云数据重整化为结构化网格
- ✅ 生成高质量的3D散点图和表面图
- ✅ 内置多个装置位形定义（EXL50U、ITER等）
- ✅ 支持灵活的数据处理和可视化配置
- ✅ 提供命令行和Python API两种使用方式

## 🚀 快速开始

### 步骤1：安装包

```bash
cd /home/ac_desktop/utils/plot_tools_py
pip install -e .
```

### 步骤2：最简单的使用示例

```python
from jorek_postproc import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取数据
col_names, blocks, _ = read_boundary_file('boundary_quantities_s04200.dat')

# 重整化为网格
data = reshape_to_grid(
    blocks['004200'], 
    col_names, 
    ['R', 'Z', 'phi', 'heatF_tot_cd'],
    iplane=1080
)

# 绘制3D表面图
fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')
config = PlottingConfig(log_norm=True, data_limits=[1e5, 3e8])
plot_surface_3d(data, fig, ax, config=config)
plt.show()
```

### 步骤3：针对特定位置的绘图

```python
# 获取装置几何信息
device = get_device_geometry('EXL50U', data.R, data.Z)

# 对特定位置的掩膜进行绘图
mask = device.masks['mask_UO']  # 外部上端
angle = device.view_angles['mask_UO']

fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')
plot_surface_3d(data, fig, ax, config=config, mask=mask, view_angle=angle)
plt.show()
```

## 📦 包结构

```
jorek_postproc/
├── data_models.py       # 数据模型定义
├── io.py               # 文件读取
├── reshaping.py        # 数据重整化
├── processing.py       # 数据处理流程
├── geometry.py         # 装置位形管理
├── plotting.py         # 3D绘图
├── config.py           # 配置和参数解析
├── cli.py              # 命令行接口
├── example.py          # 使用示例
└── README.md等文档
```

## 💻 使用方式

### 方式1：Python代码调用

```python
from jorek_postproc import read_boundary_file, reshape_to_grid, plot_surface_3d

# 读取 → 处理 → 绘图
```

### 方式2：命令行使用

```bash
python -m jorek_postproc.cli \
    -f boundary_quantities_s04200.dat \
    -t 4200 \
    -n heatF_tot_cd \
    --device EXL50U \
    --log-norm \
    --limits 1e5 3e8
```

### 方式3：运行示例脚本

```bash
python -m jorek_postproc.example
```

## 📚 主要API

### 读取函数
```python
read_boundary_file(file_path, debug=False)
    → (col_names, blocks, t_mapping)
```

### 重整化函数
```python
reshape_to_grid(block, col_names, names, iplane=None, xpoints=None, debug=False)
    → BoundaryQuantitiesData
```

### 绘图函数
```python
plot_surface_3d(data, fig, ax, config=None, mask=None, view_angle=(30, 30), ...)
plot_scatter_3d(data, fig, ax, config=None, mask=None, view_angle=(30, 30), ...)
```

### 装置几何
```python
get_device_geometry(device_name, R, Z, debug=False)
    → DeviceGeometry
```

## 🔧 配置对象

### PlottingConfig - 绘图配置
```python
config = PlottingConfig(
    log_norm=True,              # 对数色图
    cmap='viridis',             # 色图
    dpi=300,                    # 分辨率
    data_limits=[1e5, 3e8],    # 数据范围
    find_max=True               # 标记最大值
)
```

### ProcessingConfig - 处理配置
```python
config = ProcessingConfig(
    file_path='data.dat',
    timesteps=['4200', '4650'],
    device='EXL50U',
    data_limits=[1e5, 3e8],
    log_norm=True,
    debug=False
)
```

## 🎨 装置定义

### 内置装置

**EXL50U** - EXL50-U托卡马克
- mask_UO (44, 15)：外部上端
- mask_LO (-44, -15)：外部下端
- mask_UI (24, 168)：内部上端
- mask_LI (-24, -168)：内部下端

**ITER** - ITER装置
- mask_UO (40, 45)：外部上端
- mask_LO (-40, -45)：外部下端
- mask_UI (24, 150)：内部上端
- mask_LI (-20, -150)：内部下端

### 添加新装置

在 `geometry.py` 中添加新的掩膜生成函数，然后在 `get_device_geometry()` 中注册。

## 📖 文档

| 文档 | 内容 |
|------|------|
| [README.md](jorek_postproc/README.md) | 包总体说明和功能介绍 |
| [USAGE_GUIDE.md](jorek_postproc/USAGE_GUIDE.md) | 详细的使用指南和示例 |
| [QUICK_REFERENCE.md](jorek_postproc/QUICK_REFERENCE.md) | 快速参考 |
| [PACKAGE_STRUCTURE.md](jorek_postproc/PACKAGE_STRUCTURE.md) | 包结构详细说明 |

## 🔍 常见用法

### 处理多个时间步
```python
from jorek_postproc import process_multiple_timesteps

data_dict = process_multiple_timesteps(
    ['4200', '4650', '5000'],
    file_dir,
    col_names,
    ['R', 'Z', 'phi', 'heatF_tot_cd'],
    iplane=1080
)

for ts, grid_data in data_dict.items():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_surface_3d(grid_data, fig, ax)
    plt.savefig(f'plot_{ts}.png')
    plt.close(fig)
```

### 对比多个位置
```python
device = get_device_geometry('EXL50U', data.R, data.Z)

fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                         subplot_kw={'projection': '3d'})
for ax, (name, mask) in zip(axes.flat, device.masks.items()):
    angle = device.view_angles[name]
    plot_surface_3d(data, fig, ax, mask=mask, view_angle=angle)
```

### 处理X点数据（撕裂模）
```python
import numpy as np

xpoints = np.array([
    [0.75, -0.8],   # 下X点
    [0.73, 0.877]   # 上X点
], dtype=float)

data = reshape_to_grid(block, col_names, names, xpoints=xpoints)
```

## 🐛 调试

启用详细输出：

```python
grid_data = reshape_to_grid(block, col_names, names, debug=True)
```

输出示例：
```
[Reshaping] Detected 1080 toroidal planes (phi slices).
[Reshaping] Phi=0.00000: Centroid at (R=0.744, Z=0.000), Points=1134
[Reshaping] Reshaped grid size: (1080, 1134)
```

## 🎯 使用案例

### 案例1：生成单个热流图
```python
col_names, blocks, _ = read_boundary_file('boundary_quantities_s04200.dat')
data = reshape_to_grid(blocks['004200'], col_names, 
                      ['R', 'Z', 'phi', 'heatF_tot_cd'])

fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')
config = PlottingConfig(log_norm=True, data_limits=[1e5, 3e8])
plot_surface_3d(data, fig, ax, config=config)
plt.savefig('heatflux.png', dpi=300)
```

### 案例2：批量处理和保存所有位置
```bash
python -m jorek_postproc.cli \
    -f boundary_quantities_s04200.dat \
    -t 4200 \
    -n heatF_tot_cd \
    -o output_dir \
    --log-norm
```

### 案例3：对比不同时间步的演化
```python
data_dict = process_multiple_timesteps(['4200', '4650', '5000'], ...)

fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                         subplot_kw={'projection': '3d'})
for ax, (ts, data) in zip(axes, data_dict.items()):
    plot_surface_3d(data, fig, ax)
plt.tight_layout()
plt.savefig('evolution.png')
```

## ⚡ 性能提示

- **大文件**：使用 `debug=False` 减少日志开销
- **多图像**：预先分配Figure，复用Axes
- **多时间步**：使用 `process_multiple_timesteps()` 进行批处理
- **内存**：及时删除不用的大数组

## 🔌 扩展

包设计具有高度的可扩展性：

1. **新装置**：在 `geometry.py` 中添加掩膜函数
2. **新物理量处理**：在 `processing.py` 中添加计算函数
3. **新绘图类型**：在 `plotting.py` 中添加绘图函数
4. **自定义数据格式**：扩展 `data_models.py` 中的数据类

## 📝 版本信息

- **版本**：0.1.0
- **Python**：3.7+
- **依赖**：numpy, matplotlib, scipy
- **开发状态**：Alpha

## 📧 获取帮助

```python
# 查看函数说明
help(read_boundary_file)
help(plot_surface_3d)

# 查看类说明
help(BoundaryQuantitiesData)
help(PlottingConfig)

# 运行示例
python jorek_postproc/example.py
```

## 📚 更多资源

- 完整文档：见各 `.md` 文件
- 示例代码：`example.py`
- 命令行帮助：`python -m jorek_postproc.cli -h`

---

## 总结

现在你已经拥有一个完整的、模块化的、易于扩展的JOREK后处理包！

**关键特点：**
- ✅ 开箱即用的函数
- ✅ 统一的数据格式
- ✅ 灵活的配置系统
- ✅ 多装置支持
- ✅ 详细的文档
- ✅ 调试工具

**立即开始：**
```bash
cd /home/ac_desktop/utils/plot_tools_py
pip install -e .
python -m jorek_postproc.example
```

祝你使用愉快！ 🎉
