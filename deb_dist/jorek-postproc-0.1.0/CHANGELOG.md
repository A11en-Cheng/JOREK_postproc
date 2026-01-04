# CHANGELOG

所有值得注意的项目更改都应在此文件中记录。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 待添加
- 能量冲击计算模块
- HDF5/VTK数据导出
- 交互式Plotly绘图

### 待改进
- 性能优化用于大数据集
- 多进程支持
- 更多装置定义

## [0.1.0] - 2024-01-04

### 新增

#### 核心功能
- `read_boundary_file()` - 读取JOREK边界量文件
- `reshape_to_grid()` - 将非结构化点云重整化为2D网格
- `process_timestep()` - 单时间步处理
- `process_multiple_timesteps()` - 批量处理多个时间步
- `apply_data_limits()` - 应用数据上下限

#### 几何和掩膜
- `get_device_geometry()` - 获取装置几何信息
- `create_mask_exl50u()` - EXL50-U位形定义
- `create_mask_iter()` - ITER位形定义

#### 可视化
- `plot_surface_3d()` - 3D表面图绘制
- `plot_scatter_3d()` - 3D散点图绘制
- 掩膜支持和自定义视角

#### 数据模型
- `BoundaryQuantitiesData` - 统一数据格式
- `DeviceGeometry` - 装置几何信息
- `PlottingConfig` - 绘图配置

#### 配置和接口
- `ProcessingConfig` - 完整配置类
- `parse_args()` - 命令行参数解析
- 完整的CLI工具

#### 文档
- 详细的使用指南
- API快速参考
- 5个递进式示例脚本
- 包结构说明

### 已知问题
- X点分段排序需要改进
- 大数据集性能有待优化

---

## 版本发布历史

### 如何发布新版本

1. 更新版本号在 `pyproject.toml`
2. 在CHANGELOG.md中记录变更
3. 提交更改并创建标签
4. 构建分布包：`python -m build`
5. 上传到PyPI：`twine upload dist/*`

---

### 版本命名规则

遵循语义化版本 (Semantic Versioning)：
- **MAJOR** 版本：不兼容的API改变
- **MINOR** 版本：向后兼容的功能新增
- **PATCH** 版本：向后兼容的bug修复

示例：
- `0.1.0` - alpha版本，初始发布
- `1.0.0` - 首个稳定版本
- `1.1.0` - 新功能发布
- `1.1.1` - bug修复
