# 使用指南 - JOREK后处理包

## 目录

1. [安装](#安装)
2. [快速开始](#快速开始)
3. [基本概念](#基本概念)
4. [使用示例](#使用示例)
5. [高级功能](#高级功能)
6. [扩展指南](#扩展指南)

## 安装

### 方式1：开发模式安装（推荐用于开发）

进入包所在目录：

```bash
cd /home/ac_desktop/utils/plot_tools_py
pip install -e .
```

这样安装后，修改源代码会立即生效，无需重新安装。

### 方式2：直接导入

不安装，直接在Python中添加路径：

```python
import sys
sys.path.insert(0, '/home/ac_desktop/utils/plot_tools_py')
```

### 检验安装

```python
import jorek_postproc
print(jorek_postproc.__version__)
```

## 快速开始

### 最简单的使用方式

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from jorek_postproc import (
    read_boundary_file,
    reshape_to_grid,
    get_device_geometry,
    plot_surface_3d,
    PlottingConfig,
)

# 1. 读取文件
col_names, blocks, t_mapping = read_boundary_file('boundary_quantities_s04200.dat')

# 2. 重整化数据
names = ['R', 'Z', 'phi', 'heatF_tot_cd']
grid_data = reshape_to_grid(blocks['004200'], col_names, names, iplane=1080)

# 3. 获取装置位形
device = get_device_geometry('EXL50U', grid_data.R, grid_data.Z)
mask = device.masks['mask_UO']
angle = device.view_angles['mask_UO']

# 4. 绘图
fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')
config = PlottingConfig(log_norm=True, data_limits=[1e5, 3e8])
plot_surface_3d(grid_data, fig, ax, config=config, mask=mask, view_angle=angle)
```

## 基本概念

### 处理流程

```
原始文件 → 读取 → 数据提取 → 重整化 → 应用掩膜 → 绘图
  .dat    read  block    reshape  geometry  plot
```

### 坐标系统

- **R**：主要半径（m）
- **Z**：竖向坐标（m）
- **Phi**：环向角（弧度或度数）
- **Data**：物理量值（与列名相关）

### 装置位形

每个装置有多个"位置"（region），对应托卡马克的不同部分：

- **UO (Upper Outer)**：上外侧
- **LO (Lower Outer)**：下外侧
- **UI (Upper Inner)**：上内侧
- **LI (Lower Inner)**：下内侧

## 使用示例

### 示例1：基本的3D可视化

```python
from jorek_postproc import (
    read_boundary_file, reshape_to_grid, plot_surface_3d, PlottingConfig
)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取和处理数据
col_names, blocks, _ = read_boundary_file('data.dat')
grid_data = reshape_to_grid(blocks['004200'], col_names, ['R','Z','phi','heatF_tot_cd'])

# 绘图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
config = PlottingConfig(log_norm=True, dpi=300)
plot_surface_3d(grid_data, fig, ax, config=config)
```

### 示例2：应用掩膜来聚焦特定位置

```python
from jorek_postproc import get_device_geometry

device = get_device_geometry('EXL50U', grid_data.R, grid_data.Z)

# 获取特定位置的掩膜和视角
for region_name in ['mask_UO', 'mask_UI']:
    mask = device.masks[region_name]
    angle = device.view_angles[region_name]
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    plot_surface_3d(grid_data, fig, ax, mask=mask, view_angle=angle)
```

### 示例3：处理多个时间步

```python
from jorek_postproc import process_multiple_timesteps

# 处理多个时间步
timesteps = ['004200', '004650', '005000']
data_dict = process_multiple_timesteps(
    timesteps,
    file_dir,
    col_names,
    ['R', 'Z', 'phi', 'heatF_tot_cd'],
    iplane=1080,
    debug=True
)

# 对每个时间步绘图
for ts, grid_data in data_dict.items():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_surface_3d(grid_data, fig, ax)
    plt.savefig(f'plot_{ts}.png')
    plt.close(fig)
```

### 示例4：自定义绘图配置

```python
from jorek_postproc import PlottingConfig

# 详细的配置
config = PlottingConfig(
    log_norm=True,              # 使用对数色图
    cmap='plasma',              # 改变色图
    dpi=300,                    # 高分辨率
    data_limits=[1e4, 1e9],    # 设置数据范围
    find_max=True               # 标记最大值
)

# 在绘图中应用配置
plot_surface_3d(grid_data, fig, ax, config=config)
```

### 示例5：处理带有X点的撕裂模

```python
import numpy as np
from jorek_postproc import reshape_to_grid

# X点坐标（双X点情况）
xpoints = np.array([
    [0.75, -0.8],   # 下X点
    [0.73, 0.877]   # 上X点
], dtype=float)

# 使用X点进行数据排序
grid_data = reshape_to_grid(
    block_data, col_names, names,
    xpoints=xpoints,
    debug=True
)
```

### 示例6：批量处理和保存

```python
import os
from jorek_postproc import (
    process_multiple_timesteps, get_device_geometry,
    plot_surface_3d, PlottingConfig
)

# 创建输出目录
output_dir = 'results'
os.makedirs(output_dir, exist_ok=True)

# 处理多个时间步
data_dict = process_multiple_timesteps(
    ['4200', '4650', '5000'],
    '/path/to/data',
    col_names, names,
    iplane=1080
)

# 获取装置几何
device = get_device_geometry('EXL50U', grid_data.R, grid_data.Z)

# 对每个时间步和每个位置绘图
config = PlottingConfig(log_norm=True, data_limits=[1e5, 3e8])

for ts, grid_data in data_dict.items():
    for region, mask in device.masks.items():
        angle = device.view_angles[region]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        save_path = os.path.join(output_dir, f'{ts}_{region}.png')
        plot_surface_3d(grid_data, fig, ax, config=config, 
                       mask=mask, view_angle=angle, save_path=save_path)
```

## 高级功能

### 调试模式

启用详细的输出来诊断问题：

```python
grid_data = reshape_to_grid(block_data, col_names, names, debug=True)
```

调试输出示例：
```
[Reshaping] Detected 1080 toroidal planes (phi slices).
[Reshaping] Phi=0.00000: Centroid at (R=0.744, Z=0.000), Points=1134
[Reshaping] Phi=0.00582: Centroid at (R=0.744, Z=0.001), Points=1134
...
[Reshaping] Reshaped grid size: (1080, 1134)
```

### 数据限制

应用上下限来集中显示重要的数据范围：

```python
from jorek_postproc import apply_data_limits

# 应用限制
limited_data = apply_data_limits(grid_data, limits=[1e5, 3e8])
```

### 自定义装置

对于EXL50U和ITER之外的装置，可以自定义位形定义：

```python
from jorek_postproc.geometry import create_mask_exl50u

# 修改geometry.py中的get_device_geometry函数
# 或创建新的掩膜生成函数

def create_mask_mydevice(R, Z, debug=False):
    masks = {
        'mask_1': (R >= 1.0) & (R <= 2.0),
        'mask_2': (R >= 0.5) & (R <= 1.0),
    }
    angles = {
        'mask_1': (30, 45),
        'mask_2': (-30, -45),
    }
    return masks, angles
```

## 扩展指南

### 添加新的数据处理函数

在`processing.py`中添加：

```python
def calculate_some_quantity(grid_data: BoundaryQuantitiesData) -> np.ndarray:
    """
    计算某个物理量
    """
    # 实现逻辑
    result = grid_data.data * some_factor
    return result
```

### 添加新的绘图函数

在`plotting.py`中添加：

```python
def plot_custom_3d(data: BoundaryQuantitiesData, fig, ax, **kwargs):
    """
    自定义绘图函数
    """
    # 实现绘图逻辑
    pass
```

### 扩展数据模型

修改`data_models.py`中的数据类：

```python
@dataclass
class BoundaryQuantitiesData:
    # ... 现有字段 ...
    custom_field: Optional[np.ndarray] = None  # 新字段
```

## 常见问题

**Q：如何处理非常大的数据文件？**
A：可以使用numpy的分块读取或多进程处理。

**Q：绘图显示不清楚？**
A：调整`PlottingConfig`中的`dpi`参数或使用更高的分辨率。

**Q：如何对比多个时间步？**
A：使用`process_multiple_timesteps()`处理多个时间步，然后并排绘图。

**Q：能否自动生成报告？**
A：可以使用matplotlib的savefig()并结合其他库（如reportlab）生成PDF报告。

## 性能优化

对于大规模数据：

1. **使用多进程**：利用`concurrent.futures`并行处理多个时间步
2. **内存管理**：及时删除不需要的大数组
3. **图像缓存**：重用figure对象而不是每次创建新的

## 更多信息

- 详见各模块的源代码docstring
- 运行`example.py`查看完整示例
- 查看`cli.py`了解命令行用法

---

**版本**：0.1.0  
**最后更新**：2024年
