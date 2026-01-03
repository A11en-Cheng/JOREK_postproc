# 使用 jorek_postproc 包 - 最终总结

## ✨ 你现在拥有什么

一个完整的、模块化的、可扩展的JOREK后处理包，包含：

### 📦 核心功能
- **读取模块** (io.py)：读取JOREK边界量文件
- **重整化模块** (reshaping.py)：将1D点云转换为2D结构网格
- **处理模块** (processing.py)：时间步处理和数据流程
- **几何模块** (geometry.py)：装置位形定义和管理
- **绘图模块** (plotting.py)：3D散点和表面图
- **配置模块** (config.py)：参数管理和命令行解析
- **CLI模块** (cli.py)：完整的命令行工具

### 📚 文档
- README.md：功能介绍
- USAGE_GUIDE.md：详细使用指南
- QUICK_REFERENCE.md：快速参考
- PACKAGE_STRUCTURE.md：包结构说明
- GETTING_STARTED.md：快速入门

### 🎯 示例和测试
- example.py：5个递进式使用示例
- 可直接运行验证功能

## 🚀 立即开始

### 1. 安装包

```bash
cd /home/ac_desktop/utils/plot_tools_py
pip install -e .
```

验证安装：
```bash
python -c "import jorek_postproc; print('✓ 安装成功')"
```

### 2. 最简单的使用

```python
from jorek_postproc import *

# 三行代码完成：读取 → 处理 → 绘图
col_names, blocks, _ = read_boundary_file('data.dat')
data = reshape_to_grid(blocks['004200'], col_names, ['R','Z','phi','heatF_tot_cd'])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,8)); ax = fig.add_subplot(111, projection='3d')
plot_surface_3d(data, fig, ax, config=PlottingConfig(log_norm=True))
```

### 3. 命令行使用

```bash
# 最简单的使用
python -m jorek_postproc.cli -f data.dat -t 4200 -n heatF_tot_cd

# 完整的使用示例
python -m jorek_postproc.cli \
    -f boundary_quantities_s04200.dat \
    -t 4200 4650 5000 \
    -n heatF_tot_cd \
    --device EXL50U \
    --log-norm \
    --limits 1e5 3e8 \
    -o output_dir
```

### 4. 运行示例

```bash
python -m jorek_postproc.example
```

## 💡 常见使用场景

### 场景1：生成单个热流图

```python
from jorek_postproc import *

col_names, blocks, _ = read_boundary_file('boundary_quantities_s04200.dat')
data = reshape_to_grid(blocks['004200'], col_names, ['R','Z','phi','heatF_tot_cd'], iplane=1080)

fig = plt.figure(figsize=(10, 8), dpi=300)
ax = fig.add_subplot(111, projection='3d')
config = PlottingConfig(log_norm=True, data_limits=[1e5, 3e8])
plot_surface_3d(data, fig, ax, config=config)
plt.savefig('heatflux.png', dpi=300)
```

### 场景2：对比多个位置

```python
device = get_device_geometry('EXL50U', data.R, data.Z)

fig, axes = plt.subplots(2, 2, figsize=(12, 10),
                         subplot_kw={'projection': '3d'})
config = PlottingConfig(log_norm=True, data_limits=[1e5, 3e8])

for ax, (mask_name, mask) in zip(axes.flat, device.masks.items()):
    angle = device.view_angles[mask_name]
    plot_surface_3d(data, fig, ax, config=config, mask=mask, view_angle=angle)
    
plt.tight_layout()
plt.savefig('all_positions.png')
```

### 场景3：处理多个时间步

```python
from jorek_postproc import process_multiple_timesteps

data_dict = process_multiple_timesteps(
    ['4200', '4650', '5000'],
    '/path/to/data',
    col_names,
    ['R', 'Z', 'phi', 'heatF_tot_cd'],
    iplane=1080
)

# 绘制时间演化
fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                         subplot_kw={'projection': '3d'})
for ax, (ts, grid_data) in zip(axes, data_dict.items()):
    plot_surface_3d(grid_data, fig, ax)
plt.savefig('evolution.png')
```

### 场景4：批量生成所有视图

```bash
# 使用CLI一次生成所有位置的图
python -m jorek_postproc.cli \
    -f data.dat \
    -t 4200 \
    -n heatF_tot_cd \
    -o output_dir \
    --log-norm
```

## 📋 快速参考

### 主要函数

| 函数 | 用途 | 文件 |
|------|------|------|
| `read_boundary_file()` | 读取JOREK文件 | io.py |
| `reshape_to_grid()` | 1D→2D网格 | reshaping.py |
| `process_timestep()` | 处理单时间步 | processing.py |
| `process_multiple_timesteps()` | 批量处理 | processing.py |
| `get_device_geometry()` | 获取装置位形 | geometry.py |
| `plot_surface_3d()` | 绘制表面 | plotting.py |
| `plot_scatter_3d()` | 绘制散点 | plotting.py |

### 主要配置

```python
PlottingConfig(
    log_norm=True,              # 对数色图
    cmap='viridis',             # 色图名称
    dpi=300,                    # 分辨率
    data_limits=[1e5, 3e8],    # 数据范围
    find_max=True               # 标记最大值
)
```

### 支持的装置

```python
device = get_device_geometry('EXL50U', R, Z)  # EXL50U
device = get_device_geometry('ITER', R, Z)    # ITER
```

## 📞 获取帮助

```python
# 查看函数文档
help(read_boundary_file)
help(reshape_to_grid)
help(plot_surface_3d)

# 查看类文档
help(BoundaryQuantitiesData)
help(PlottingConfig)

# 查看命令行帮助
python -m jorek_postproc.cli -h

# 运行示例
python -m jorek_postproc.example
```

## 🔧 高级特性

### 处理X点数据（撕裂模）
```python
import numpy as np
xpoints = np.array([[0.75, -0.8], [0.73, 0.877]], dtype=float)
data = reshape_to_grid(block, col_names, names, xpoints=xpoints)
```

### 启用调试模式
```python
data = reshape_to_grid(block, col_names, names, debug=True)
```

### 自定义绘图参数
```python
config = PlottingConfig(
    log_norm=True,
    cmap='plasma',
    dpi=300,
    data_limits=[1e4, 1e9],
    find_max=True
)
plot_surface_3d(data, fig, ax, config=config)
```

## 📈 性能优化

- **大文件处理**：使用 `debug=False` 减少开销
- **多时间步**：使用 `process_multiple_timesteps()` 批量处理
- **多图像**：预先分配Figure和Axes，避免重复创建
- **内存管理**：及时删除不需要的大数组

## 🎓 学习路径

1. **入门** → 运行 `example.py`
2. **基本使用** → 参考 `QUICK_REFERENCE.md`
3. **详细学习** → 阅读 `USAGE_GUIDE.md`
4. **深入理解** → 查看 `PACKAGE_STRUCTURE.md`
5. **扩展功能** → 修改源代码添加新功能

## 🔄 从原始脚本迁移

如果你之前有类似 `plot_bnd_quant_3d_legs.py` 的脚本：

**之前**：
```python
# 分散的函数定义，难以复用
def read_boundary_file(file_path):
    ...
def reshape_to_grid_updated(block, col_names, names):
    ...
def plot_surface_from_scatter_dict(...):
    ...
```

**现在**：
```python
# 模块化的、可复用的包
from jorek_postproc import read_boundary_file, reshape_to_grid, plot_surface_3d
```

## ✅ 验证清单

- [ ] 已安装包：`pip install -e .`
- [ ] 可导入包：`import jorek_postproc`
- [ ] 运行示例：`python -m jorek_postproc.example`
- [ ] CLI工作：`python -m jorek_postproc.cli -h`
- [ ] 文档完整：阅读各markdown文件

## 🎉 你已准备好！

现在你有：

✓ **15个高质量文件**
- 10个Python模块
- 5份详细文档

✓ **完整的功能**
- 文件读取 → 数据处理 → 可视化的完整流程
- 支持多装置、多时间步、灵活配置

✓ **多种使用方式**
- Python代码调用
- 命令行工具
- 示例脚本

✓ **优秀的设计**
- 模块化结构
- 统一数据格式
- 易于扩展

---

## 现在就开始吧！

```bash
# 1. 安装
cd /home/ac_desktop/utils/plot_tools_py
pip install -e .

# 2. 测试
python -m jorek_postproc.example

# 3. 使用
python -m jorek_postproc.cli -f data.dat -t 4200 -n heatF_tot_cd
```

祝你使用愉快！ 🚀

---

**需要帮助？**
- 查看文档：各模块的markdown文件
- 运行示例：`example.py`
- 查看源代码：各Python文件中的docstring
- 使用命令行帮助：`python -m jorek_postproc.cli -h`
