# JOREK后处理包 - 边界量可视化工具

一个用于处理和可视化JOREK等离子体边界量数据的综合Python包。

## 功能特性

- ✅ **灵活的文件读取**：支持JOREK边界量文件格式
- ✅ **数据重整化**：将非结构化点云数据转换为结构化网格
- ✅ **3D可视化**：支持散点图和表面图
- ✅ **装置位形管理**：内置EXL50U和ITER等装置定义，支持自定义扩展
- ✅ **标准数据格式**：统一的数据模型贯穿整个处理管道
- ✅ **灵活配置**：通过配置对象或命令行参数控制处理过程
- ✅ **调试支持**：内置调试模式用于问题诊断

## 包结构

```
jorek_postproc/
├── __init__.py           # 包初始化和主要导出
├── data_models.py        # 数据格式定义
├── io.py                 # 文件读取函数
├── reshaping.py          # 数据重整化模块
├── processing.py         # 时间步处理和数据流程
├── geometry.py           # 装置位形定义和管理
├── plotting.py           # 3D绘图函数
├── config.py             # 配置和命令行解析
└── example.py            # 使用示例脚本
```

## 安装

### 本地开发安装

在包所在目录下运行：

```bash
pip install -e .
```

或者直接添加到Python路径：

```python
import sys
sys.path.insert(0, '/home/ac_desktop/utils/plot_tools_py')
```

## 快速开始

### 基本使用流程

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

# 2. 重整化数据到网格
names = ['R', 'Z', 'phi', 'heatF_tot_cd']
grid_data = reshape_to_grid(blocks['004200'], col_names, names, iplane=1080)

# 3. 获取装置位形信息
device = get_device_geometry('EXL50U', grid_data.R, grid_data.Z)

# 4. 选择位置并绘图
mask = device.masks['mask_UO']  # 外部上端
angle = device.view_angles['mask_UO']

fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')

config = PlottingConfig(
    log_norm=True,
    data_limits=[1e5, 3e8],
    find_max=True
)

plot_surface_3d(grid_data, fig, ax, config=config, mask=mask, view_angle=angle)
```

### 运行示例

```bash
python -m jorek_postproc.example
```

## 数据模型

### BoundaryQuantitiesData

核心数据类，表示重整化后的边界量：

```python
@dataclass
class BoundaryQuantitiesData:
    R: np.ndarray              # 主要半径坐标
    Z: np.ndarray              # 竖向坐标
    phi: np.ndarray            # 环向角
    data: np.ndarray           # 物理量数据
    data_name: str             # 物理量名称
    time: Optional[float]      # 物理时间
    time_step: Optional[str]   # 时间步标识
    grid_shape: Optional[Tuple] # 网格形状
```

### PlottingConfig

绘图配置参数：

```python
@dataclass
class PlottingConfig:
    log_norm: bool = False            # 对数色图
    cmap: str = 'viridis'             # 色图名称
    dpi: int = 300                    # 分辨率
    data_limits: Optional[List] = None # 数据范围 [min, max]
    find_max: bool = True              # 标记最大值
```

## 主要函数

### IO模块

```python
read_boundary_file(file_path, debug=False) -> (col_names, blocks, t_mapping)
```

读取JOREK边界量文件。

### 重整化模块

```python
reshape_to_grid(block, col_names, names, iplane=None, xpoints=None, debug=False) 
    -> BoundaryQuantitiesData
```

将1D点云转换为2D网格。支持标准重心排序和X点分段排序。

### 处理模块

```python
process_timestep(timestep, file_path, column_names, names, ...) 
    -> (time, grid_data)

process_multiple_timesteps(timesteps, file_addr, column_names, names, ...)
    -> Dict[str, BoundaryQuantitiesData]

apply_data_limits(data, limits=None) -> BoundaryQuantitiesData
```

### 几何模块

```python
get_device_geometry(device_name, R, Z, debug=False) -> DeviceGeometry

create_mask_exl50u(R, Z, debug=False) -> (masks, angles)
create_mask_iter(R, Z, debug=False) -> (masks, angles)
```

### 绘图模块

```python
plot_scatter_3d(data, fig, ax, config=None, mask=None, view_angle=(30, 30), ...)
plot_surface_3d(data, fig, ax, config=None, mask=None, view_angle=(30, 30), ...)
```

## 装置位形管理

### 支持的装置

- **EXL50U**：EXL50-U托卡马克
  - mask_UO: 外部上端
  - mask_LO: 外部下端
  - mask_UI: 内部上端
  - mask_LI: 内部下端

- **ITER**：ITER装置
  - mask_UO, mask_LO, mask_UI, mask_LI

### 添加新装置

创建新的掩膜生成函数：

```python
def create_mask_mydevice(R, Z, debug=False):
    masks = {
        'mask_1': (R >= 1.0) & (R <= 2.0) & (Z >= 0),
        'mask_2': (R >= 1.0) & (R <= 2.0) & (Z < 0),
    }
    angles = {
        'mask_1': (30, 45),
        'mask_2': (-30, -45),
    }
    return masks, angles

# 在geometry.py中的get_device_geometry函数中添加
if device_name_upper == 'MYDEVICE':
    masks, angles = create_mask_mydevice(R, Z, debug=debug)
```

## 配置参数

### ProcessingConfig数据类

```python
@dataclass
class ProcessingConfig:
    file_path: str
    timesteps: List[str]
    iplane: int = 1080
    data_name: str = 'heatF_tot_cd'
    device: str = 'EXL50U'
    data_limits: Optional[List[float]] = None
    norm_factor: Optional[float] = None
    plot_surface: bool = True
    plot_overall: bool = False
    log_norm: bool = False
    find_max: bool = True
    output_dir: Optional[str] = None
    xpoints: Optional[List[float]] = None
    debug: bool = False
```

## 高级特性

### X点处理（双x非凸壁位形）

对于具有双X点的非凸壁位形，可提供X点坐标以改进数据排序：

```python
xpoints = np.array([[x1, z1], [x2, z2]], dtype=float)
grid_data = reshape_to_grid(block_data, col_names, names, xpoints=xpoints)
```

### 调试模式

启用详细的日志输出：

```python
grid_data = reshape_to_grid(block_data, col_names, names, debug=True)
```

## API参考

详见源代码中的docstring文档。

## 扩展指南

### 添加新的物理量处理函数

在合适的模块中添加新函数。例如，在`processing.py`中添加能量冲击计算函数：

```python
def calculate_energy_impact(grid_data: BoundaryQuantitiesData) -> np.ndarray:
    """计算能量冲击参数"""
    # 实现计算逻辑
    pass
```

### 自定义数据格式

修改`data_models.py`中的数据类以支持新的字段。

## 常见问题

**Q: 如何处理数据中的NaN值？**
A: 绘图函数会自动忽略NaN值。可使用numpy的`np.nan_to_num()`或`np.nanmean()`等函数处理。

**Q: 如何改变图的观看角度？**
A: 使用`view_angle`参数，例如`view_angle=(45, 90)`。

**Q: 如何同时处理多个设备？**
A: 对每个设备创建对应的几何定义，然后在循环中调用`get_device_geometry()`。

## 许可证

MIT License

## 联系方式

如有问题或建议，请提交Issue。

---

**最后更新**：2024年
