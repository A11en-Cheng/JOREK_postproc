"""
JOREK后处理包 - 边界量可视化工具

一个用于处理和可视化JOREK边界量数据的综合Python包。

主要功能：
  - 读取JOREK边界量文件
  - 将非结构化数据重整化为结构化网格
  - 生成3D散点和表面图
  - 支持多个装置位形 (EXL50U, ITER, 等)
  - 灵活的配置和扩展机制

基本使用:
  >>> from jorek_postproc import BoundaryQuantitiesData, PlottingConfig
  >>> from jorek_postproc import read_boundary_file, reshape_to_grid
  >>> from jorek_postproc import plot_scatter_3d, plot_surface_3d
  
  >>> # 读取文件
  >>> col_names, blocks, t_mapping = read_boundary_file('boundary_quantities_s04200.dat')
  >>> 
  >>> # 重整化数据
  >>> names = ['R', 'Z', 'phi', 'heatF_tot_cd']
  >>> grid_data = reshape_to_grid(blocks['004200'], col_names, names, iplane=1080)
  >>>
  >>> # 绘图
  >>> fig = plt.figure(figsize=(8, 6), dpi=300)
  >>> ax = fig.add_subplot(111, projection='3d')
  >>> config = PlottingConfig(log_norm=True, data_limits=[1e5, 3e8])
  >>> plot_surface_3d(grid_data, fig, ax, config=config)

设备位形:
  支持的设备: EXL50U, ITER
  可通过get_device_geometry()获取位形信息

扩展机制:
  对于新装置，可创建新的掩膜生成函数并通过register_custom_device()注册
"""

__version__ = '0.1.0'
__author__ = 'Your Name'
__all__ = [
    # 数据模型
    'BoundaryQuantitiesData',
    'DeviceGeometry',
    'PlottingConfig',
    
    # IO 模块
    'read_boundary_file',
    
    # 重整化模块
    'reshape_to_grid',
    
    # 处理模块
    'process_timestep',
    'process_multiple_timesteps',
    'apply_data_limits',
    
    # 几何模块
    'get_device_geometry',
    'create_mask_exl50u',
    'create_mask_iter',
    
    # 绘图模块
    'plot_scatter_3d',
    'plot_surface_3d',
    
    # 配置和解析
    'ProcessingConfig',
    'parse_args',
    'create_debug_config',
]

# 导入子模块
from .data_models import BoundaryQuantitiesData, DeviceGeometry, PlottingConfig
from .io import read_boundary_file
from .reshaping import reshape_to_grid
from .processing import (
    process_timestep,
    process_multiple_timesteps,
    apply_data_limits
)
from .geometry import (
    get_device_geometry,
    create_mask_exl50u,
    create_mask_iter,
)
from .plotting import plot_scatter_3d, plot_surface_3d
from .config import ProcessingConfig, parse_args, create_debug_config
