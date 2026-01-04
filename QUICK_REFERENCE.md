# 快速参考指南

## 安装

```bash
cd /home/ac_desktop/utils/plot_tools_py
pip install -e .
```

## 最小示例

```python
from jorek_postproc import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取 → 重整 → 绘图
col_names, blocks, _ = read_boundary_file('data.dat')
data = reshape_to_grid(blocks['004200'], col_names, ['R','Z','phi','heatF_tot_cd'])

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')
plot_surface_3d(data, fig, ax, config=PlottingConfig(log_norm=True))
```

## 常用函数速查

### 读取文件
```python
col_names, blocks, t_mapping = read_boundary_file('file.dat', debug=False)
```

### 数据重整化
```python
grid_data = reshape_to_grid(
    block,                              # numpy数组
    col_names,                          # 列名列表
    ['R', 'Z', 'phi', 'data_name'],   # 物理量映射
    iplane=1080,                        # 环向平面数
    xpoints=None,                       # X点坐标（可选）
    debug=False
)
```

### 数据处理
```python
# 单个时间步
time, data = process_timestep(ts, file, col_names, names)

# 多个时间步
data_dict = process_multiple_timesteps(
    ['t1', 't2'], file_dir, col_names, names
)

# 应用限制
limited = apply_data_limits(data, limits=[1e5, 3e8])
```

### 装置几何
```python
device = get_device_geometry('EXL50U', grid_data.R, grid_data.Z)
mask = device.masks['mask_UO']
angle = device.view_angles['mask_UO']
```

### 绘图
```python
fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')

config = PlottingConfig(
    log_norm=True,
    cmap='viridis',
    data_limits=[1e5, 3e8],
    find_max=True
)

# 散点图
plot_scatter_3d(data, fig, ax, config=config, 
               mask=mask, view_angle=(30, 45),
               save_path='output.png')

# 表面图
plot_surface_3d(data, fig, ax, config=config,
               mask=mask, view_angle=(30, 45),
               save_path='output.png')
```

## 命令行使用

```bash
# 基本使用
python -m jorek_postproc.cli -f file.dat -t 4200 -n heatF_tot_cd

# 多个时间步
python -m jorek_postproc.cli -f file.dat -t 4200 4650 5000

# 带选项
python -m jorek_postproc.cli \
    -f file.dat -t 4200 \
    -n heatF_tot_cd \
    --device EXL50U \
    --log-norm \
    --limits 1e5 3e8 \
    -o output_dir

# 查看帮助
python -m jorek_postproc.cli -h
```

## 配置对象

### ProcessingConfig（处理配置）
```python
config = ProcessingConfig(
    file_path='data.dat',
    timesteps=['4200', '4650'],
    iplane=1080,
    data_name='heatF_tot_cd',
    device='EXL50U',
    data_limits=[1e5, 3e8],
    norm_factor=4.1006E-07,
    plot_surface=True,
    log_norm=True,
    find_max=True,
    output_dir='output',
    debug=False
)
```

### PlottingConfig（绘图配置）
```python
config = PlottingConfig(
    log_norm=False,
    cmap='viridis',
    dpi=300,
    data_limits=None,
    find_max=True
)
```

## 数据模型

### BoundaryQuantitiesData
```python
data.R              # R坐标网格 (N_phi, N_poloidal)
data.Z              # Z坐标网格
data.phi            # Phi坐标网格
data.data           # 物理量数据
data.data_name      # 物理量名称
data.time           # 物理时间
data.time_step      # 时间步标识
data.grid_shape     # 网格形状

# 方法
data.is_2d_grid()   # 检查是否为2D网格
data.get_2d_view(iplane)  # 转换为2D视图
```

### DeviceGeometry
```python
device.name                 # 装置名称
device.masks               # {'mask_UO': ..., 'mask_LO': ...}
device.view_angles         # {'mask_UO': (30, 45), ...}
```

## 装置信息

### EXL50-U
- mask_UO (44, 15)：外部上端
- mask_LO (-44, -15)：外部下端  
- mask_UI (24, 168)：内部上端
- mask_LI (-24, -168)：内部下端

### ITER
- mask_UO (40, 45)：外部上端
- mask_LO (-40, -45)：外部下端
- mask_UI (24, 150)：内部上端
- mask_LI (-20, -150)：内部下端

## 常见技巧

### 对比多个时间步
```python
timesteps = ['4200', '4650', '5000']
data_dict = process_multiple_timesteps(
    timesteps, file_dir, col_names, names
)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), 
                         subplot_kw={'projection': '3d'})
for ax, (ts, data) in zip(axes, data_dict.items()):
    plot_surface_3d(data, fig, ax)
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

### 处理X点数据
```python
import numpy as np

xpoints = np.array([
    [0.75, -0.8],   # 下X点
    [0.73, 0.877]   # 上X点
], dtype=float)

data = reshape_to_grid(block, col_names, names, xpoints=xpoints)
```

### 批量输出所有位置
```python
device = get_device_geometry('EXL50U', data.R, data.Z)
config = PlottingConfig(log_norm=True, data_limits=[1e5, 3e8])

for mask_name, mask in device.masks.items():
    angle = device.view_angles[mask_name]
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = fig.add_subplot(111, projection='3d')
    
    plot_surface_3d(data, fig, ax, config=config,
                   mask=mask, view_angle=angle,
                   save_path=f'{mask_name}.png')
```

## 调试技巧

```python
# 启用详细输出
grid_data = reshape_to_grid(block, col_names, names, debug=True)

# 检查数据范围
print(f"R: [{data.R.min()}, {data.R.max()}]")
print(f"Z: [{data.Z.min()}, {data.Z.max()}]")
print(f"Data: [{data.data.min()}, {data.data.max()}]")

# 可视化掩膜
import matplotlib.pyplot as plt
plt.imshow(device.masks['mask_UO'])
plt.colorbar()
plt.show()
```

## 性能提示

- 大文件：使用 `debug=False` 减少日志输出
- 多图：预先分配figure，复用axes
- 多进程：使用concurrent.futures处理多时间步
- 内存：及时删除不用的大数组

## 文件位置

```
包安装后可通过以下方式定位：
import jorek_postproc
import os
print(os.path.dirname(jorek_postproc.__file__))
```

## 获取帮助

```python
# 查看函数说明
help(read_boundary_file)
help(reshape_to_grid)

# 查看类说明
help(BoundaryQuantitiesData)
help(PlottingConfig)

# 运行示例
python jorek_postproc/example.py
```

## 常见错误排查

| 错误 | 原因 | 解决 |
|------|------|------|
| ModuleNotFoundError | 包未安装 | pip install -e . |
| FileNotFoundError | 文件路径错误 | 检查文件路径 |
| ValueError: Column name mismatch | 列名不存在 | 检查col_names中有无该列 |
| ValueError: No valid data points | 掩膜过于严格 | 放宽掩膜条件 |
| 图显示不清 | 分辨率太低 | 提高dpi参数 |

---

更多信息：见README.md、USAGE_GUIDE.md、PACKAGE_STRUCTURE.md
