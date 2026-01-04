# 包结构总结

## 目录结构

```
plot_tools_py/
├── setup.py                          # 包安装配置
├── jorek_postproc/                   # 主包目录
│   ├── __init__.py                   # 包初始化，导出主要API
│   ├── data_models.py                # 数据模型定义
│   │   ├── BoundaryQuantitiesData    # 主要数据类
│   │   ├── DeviceGeometry            # 装置位形数据类
│   │   └── PlottingConfig            # 绘图配置数据类
│   │
│   ├── io.py                         # 文件读取模块
│   │   └── read_boundary_file()      # 读取JOREK边界量文件
│   │
│   ├── reshaping.py                  # 数据重整化模块
│   │   └── reshape_to_grid()         # 1D→2D网格转换
│   │
│   ├── processing.py                 # 数据处理模块
│   │   ├── process_timestep()        # 处理单个时间步
│   │   ├── process_multiple_timesteps()  # 批量处理
│   │   └── apply_data_limits()       # 应用数据限制
│   │
│   ├── geometry.py                   # 装置几何和位形管理
│   │   ├── get_device_geometry()     # 获取装置几何信息
│   │   ├── create_mask_exl50u()      # EXL50U位形定义
│   │   └── create_mask_iter()        # ITER位形定义
│   │
│   ├── plotting.py                   # 3D可视化模块
│   │   ├── plot_scatter_3d()         # 3D散点图
│   │   └── plot_surface_3d()         # 3D表面图
│   │
│   ├── config.py                     # 配置和命令行解析
│   │   ├── ProcessingConfig          # 配置数据类
│   │   ├── parse_args()              # 命令行参数解析
│   │   └── create_debug_config()     # 调试配置
│   │
│   ├── cli.py                        # 命令行接口
│   │   ├── process_single_timestep() # CLI主处理函数
│   │   └── main()                    # 命令行入口
│   │
│   ├── example.py                    # 使用示例脚本
│   │   ├── example_basic_usage()
│   │   ├── example_with_device_geometry()
│   │   ├── example_scatter_plot()
│   │   ├── example_surface_plot()
│   │   ├── example_multiple_views()
│   │   └── main()
│   │
│   ├── README.md                     # 包总体说明
│   └── USAGE_GUIDE.md                # 详细使用指南
│
└── ... (其他源文件)
```

## 模块说明

### 1. data_models.py - 数据模型定义

定义了三个核心数据类，确保整个处理管道的数据一致性：

- **BoundaryQuantitiesData**：标准化的边界量数据格式
  - R, Z, phi: 坐标网格
  - data: 物理量数据
  - 元数据：time, time_step, grid_shape
  - 方法：is_2d_grid(), get_2d_view()

- **DeviceGeometry**：装置位形信息
  - name: 装置名称
  - masks: 位置掩膜字典
  - view_angles: 推荐观看角度

- **PlottingConfig**：绘图配置参数
  - log_norm: 对数色图开关
  - cmap: 色图选择
  - dpi: 分辨率
  - data_limits: 数据范围
  - find_max: 标记最大值

### 2. io.py - 文件读取模块

```python
read_boundary_file(file_path, debug=False)
  → (col_names, blocks, t_mapping)
```

- 读取JOREK边界量文件
- 返回列名、数据块和时间映射
- 支持调试模式

### 3. reshaping.py - 数据重整化

```python
reshape_to_grid(block, col_names, names, iplane=None, xpoints=None, debug=False)
  → BoundaryQuantitiesData
```

- 将非结构化1D数据转换为2D网格
- 支持标准重心排序
- 支持X点分段排序（双X点撕裂模）

### 4. processing.py - 处理流程

```python
process_timestep(...)           # 处理单个时间步
process_multiple_timesteps(...) # 批量处理
apply_data_limits(...)          # 应用数据限制
```

### 5. geometry.py - 装置位形管理

```python
get_device_geometry(device_name, R, Z, debug=False)
  → DeviceGeometry
```

支持的装置：
- EXL50U：mask_UO, mask_LO, mask_UI, mask_LI
- ITER：同样的四个位置

### 6. plotting.py - 3D可视化

```python
plot_scatter_3d(data, fig, ax, config=None, mask=None, ...)
plot_surface_3d(data, fig, ax, config=None, mask=None, ...)
```

- 支持掩膜过滤
- 支持自定义视角
- 支持图像保存

### 7. config.py - 配置和参数解析

```python
@dataclass
class ProcessingConfig  # 完整配置
parse_args()            # 命令行解析
create_debug_config()   # 调试配置
```

### 8. cli.py - 命令行接口

提供完整的命令行工具：

```bash
python -m jorek_postproc.cli -f file.dat -t 4200 -n heatF_tot_cd --log-norm
```

### 9. example.py - 使用示例

五个递进式示例：
1. 基本使用流程
2. 装置位形使用
3. 散点图绘制
4. 表面图绘制
5. 多位置视图

## 数据流向

```
JOREK输出文件
    ↓
[io.py] read_boundary_file()
    ↓
{时间步: numpy数组}
    ↓
[reshaping.py] reshape_to_grid()
    ↓
BoundaryQuantitiesData (2D网格)
    ↓
[geometry.py] get_device_geometry()
    ↓
DeviceGeometry (掩膜)
    ↓
[plotting.py] plot_surface_3d/scatter_3d()
    ↓
PNG图像文件
```

## 主要特性

✓ **模块化设计**：各功能独立，易于扩展和维护
✓ **统一数据格式**：BoundaryQuantitiesData贯穿全程
✓ **多装置支持**：内置EXL50U、ITER，易于添加新装置
✓ **灵活配置**：既支持代码调用，也支持命令行
✓ **调试支持**：详细的日志输出
✓ **完善文档**：docstring、README、使用指南

## 使用方式

### 方式1：作为Python包导入

```python
from jorek_postproc import (
    read_boundary_file,
    reshape_to_grid,
    plot_surface_3d
)
```

### 方式2：命令行使用

```bash
python -m jorek_postproc.cli \
    -f boundary_quantities_s04200.dat \
    -t 4200 \
    -n heatF_tot_cd \
    --device EXL50U \
    --log-norm
```

### 方式3：作为脚本模块

```bash
python jorek_postproc/example.py
```

## 安装方法

```bash
cd /home/ac_desktop/utils/plot_tools_py
pip install -e .
```

或直接添加到Python路径。

## 扩展点

1. **新装置**：在geometry.py中添加掩膜生成函数
2. **新物理量**：在processing.py中添加计算函数
3. **新绘图类型**：在plotting.py中添加图形函数
4. **新数据类型**：扩展data_models.py中的数据类

## 性能考虑

- 对于大文件：使用分块读取或内存映射
- 对于多时间步：使用multiprocessing并行处理
- 对于高精度图：设置dpi=300及以上

## 版本信息

- **版本**：0.1.0
- **Python**：3.7+
- **依赖**：numpy, matplotlib, scipy
- **开发状态**：Alpha

---

此包设计用于JOREK等离子体模拟后处理，特别是边界量数据的可视化。
可根据实际需求扩展和定制。
