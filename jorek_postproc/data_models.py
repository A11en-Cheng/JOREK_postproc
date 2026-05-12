"""
Data models and format definitions for postprocessing boundary quantities.

This module defines the standard data formats used throughout the package to ensure
consistency and facilitate extension with new devices and processing functions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np


@dataclass
class BoundaryQuantitiesData:
    """
    标准化的边界量数据格式。
    
    该类定义了通过整个处理管道的统一数据格式。
    
    Attributes
    ----------
    R : numpy.ndarray
        主要半径坐标，形状为 (N_phi, N_poloidal) 或 (N_samples,)
    Z : numpy.ndarray
        竖向坐标，形状为 (N_phi, N_poloidal) 或 (N_samples,)
    phi : numpy.ndarray
        环向角，形状为 (N_phi, N_poloidal) 或 (N_samples,)
    data : numpy.ndarray
        物理量数据，形状为 (N_phi, N_poloidal) 或 (N_samples,)
    data_name : str
        物理量的名称 (e.g., 'heatF_tot_cd')
    time : float, optional
        对应的物理时间
    time_step : str, optional
        时间步数标识 (e.g., '004200')
    grid_shape : Tuple[int, int], optional
        网格形状 (N_phi, N_poloidal)，用于3D表面绘图
    """
    R: np.ndarray
    Z: np.ndarray
    phi: np.ndarray
    data: np.ndarray
    data_name: str
    time: Optional[float] = None
    time_step: Optional[str] = None
    grid_shape: Optional[Tuple[int, int]] = None
    theta: Optional[np.ndarray] = None
    arc_length: Optional[np.ndarray] = None
    
    def is_2d_grid(self) -> bool:
        """检查数据是否已重整化为2D网格格式"""
        return len(self.R.shape) == 2
    
    def get_2d_view(self, iplane: int) -> 'BoundaryQuantitiesData':
        """将1D数据转换为2D网格视图"""
        if self.is_2d_grid():
            return self
        
        n_total = len(self.R)
        n_poloidal = n_total // iplane
        
        return BoundaryQuantitiesData(
            R=np.reshape(self.R, (iplane, n_poloidal), order='C'),
            Z=np.reshape(self.Z, (iplane, n_poloidal), order='C'),
            phi=np.reshape(self.phi, (iplane, n_poloidal), order='C'),
            data=np.reshape(self.data, (iplane, n_poloidal), order='C'),
            data_name=self.data_name,
            time=self.time,
            time_step=self.time_step,
            grid_shape=(iplane, n_poloidal),
            theta=np.reshape(self.theta, (iplane, n_poloidal), order='C') if self.theta is not None else None
        )


@dataclass
class DeviceGeometry:
    """
    设备位形定义。
    
    Attributes
    ----------
    name : str
        设备名称 (e.g., 'EXL50U', 'ITER')
    masks : Dict[str, np.ndarray]
        位形掩膜字典，键为位置标识 (e.g., 'mask_UO', 'mask_LI')
    view_angles : Dict[str, Tuple[int, int]]
        对应位置的推荐观看角度 (elevation, azimuth)
    xpoints : Optional[np.ndarray] = None
        X点坐标数组
    """
    name: str
    masks: Dict[str, np.ndarray]
    view_angles: Dict[str, Tuple[int, int]]
    xpoints: Optional[np.ndarray] = None


@dataclass
class PlottingConfig:
    """
    绘图配置参数。
    
    Attributes
    ----------
    log_norm : bool
        是否使用对数归一化
    cmap : str
        色图名称
    dpi : int
        图像分辨率
    data_limits : Optional[List[float]]
        数据显示范围 [min, max]
    find_max : bool
        是否在图上标记最大值
    """
    log_norm: bool = False
    cmap: str = 'viridis'
    dpi: int = 300
    data_limits: Optional[List[float]] = None
    find_max: bool = True
    show_left_plot: bool = True
    show_right_plot: bool = True
    use_arc_length: bool = False
