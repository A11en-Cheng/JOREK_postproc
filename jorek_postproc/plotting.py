"""
Plotting module for 3D visualization of boundary quantities.

Provides functions for scatter and surface plotting in 3D space.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, Tuple

from .data_models import BoundaryQuantitiesData, PlottingConfig


def plot_scatter_3d(
    data: BoundaryQuantitiesData,
    fig: plt.Figure,
    ax: plt.Axes,
    config: Optional[PlottingConfig] = None,
    mask: Optional[np.ndarray] = None,
    view_angle: Tuple[int, int] = (30, 30),
    save_path: Optional[str] = None,
    debug: bool = False
) -> None:
    """
    绘制3D散点图。

    Parameters
    ----------
    data : BoundaryQuantitiesData
        要绘制的数据
    fig : matplotlib.figure.Figure
        图形对象
    ax : matplotlib.axes.Axes
        坐标轴对象
    config : PlottingConfig, optional
        绘图配置，如果为None使用默认值
    mask : numpy.ndarray, optional
        布尔掩膜，用于过滤数据点
    view_angle : tuple of 2 int
        视角 (elevation, azimuth)
    save_path : str, optional
        图像保存路径，如果为None则显示而不保存
    debug : bool, optional
        调试模式标志
    """
    if config is None:
        config = PlottingConfig()
    
    # 准备数据
    if mask is not None:
        R = data.R[mask]
        Z = data.Z[mask]
        phi = data.phi[mask]
        val = data.data[mask]
    else:
        R = data.R.flatten() if len(data.R.shape) == 2 else data.R
        Z = data.Z.flatten() if len(data.Z.shape) == 2 else data.Z
        phi = data.phi.flatten() if len(data.phi.shape) == 2 else data.phi
        val = data.data.flatten() if len(data.data.shape) == 2 else data.data
    
    # 应用数据限制
    if config.data_limits is not None:
        min_val, max_val = config.data_limits[0], config.data_limits[1]
        val = np.clip(val, min_val, max_val)
    
    # 创建归一化和颜色映射
    if config.log_norm:
        norm = LogNorm(vmin=np.nanmin(val), vmax=np.nanmax(val))
    else:
        norm = plt.Normalize(vmin=np.nanmin(val), vmax=np.nanmax(val))
    
    sm = cm.ScalarMappable(norm=norm, cmap=config.cmap)
    sm.set_array(val)
    
    # 绘制散点
    sc = ax.scatter(
        R, phi, Z,
        c=val,
        cmap=config.cmap,
        s=2,
        alpha=0.5,
        norm=norm
    )
    
    # 标记最大值
    if config.find_max:
        plot_max_point(R, Z, phi, val, ax)
    
    # 设置坐标轴
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label(data.data_name, rotation=270, labelpad=15)
    ax.set_aspect('equalxz')
    ax.set_xlabel('R Axis', fontsize=10)
    ax.set_ylabel('Phi Axis', fontsize=10)
    ax.set_zlabel('Z Axis', fontsize=10)
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # 保存或显示
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        if debug:
            print(f"[Plotting] Saved scatter plot to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_surface_3d(
    data: BoundaryQuantitiesData,
    fig: plt.Figure,
    ax: plt.Axes,
    config: Optional[PlottingConfig] = None,
    mask: Optional[np.ndarray] = None,
    view_angle: Tuple[int, int] = (30, 30),
    save_path: Optional[str] = None,
    debug: bool = False
) -> None:
    """
    绘制3D表面图。

    Parameters
    ----------
    data : BoundaryQuantitiesData
        要绘制的数据，应为2D网格格式
    fig : matplotlib.figure.Figure
        图形对象
    ax : matplotlib.axes.Axes
        坐标轴对象
    config : PlottingConfig, optional
        绘图配置，如果为None使用默认值
    mask : numpy.ndarray, optional
        2D布尔掩膜，用于过滤数据点
    view_angle : tuple of 2 int
        视角 (elevation, azimuth)
    save_path : str, optional
        图像保存路径，如果为None则显示而不保存
    debug : bool, optional
        调试模式标志
    """
    if config is None:
        config = PlottingConfig()
    
    if not data.is_2d_grid():
        raise ValueError("Data must be in 2D grid format for surface plotting. "
                        "Use data.get_2d_view(iplane) to convert.")
    
    # 应用掩膜
    if mask is not None:
        R_plot = np.where(mask, data.R, np.nan)
        Z_plot = np.where(mask, data.Z, np.nan)
        phi_plot = np.where(mask, data.phi, np.nan)
        data_plot = np.where(mask, data.data, np.nan)
    else:
        R_plot = data.R
        Z_plot = data.Z
        phi_plot = data.phi
        data_plot = data.data
    
    # 应用数据限制
    if config.data_limits is not None:
        min_val, max_val = config.data_limits[0], config.data_limits[1]
        data_plot = np.clip(data_plot, min_val, max_val)
    
    # 验证有效数据
    valid_data = data_plot[~np.isnan(data_plot)]
    if valid_data.size == 0:
        raise ValueError("No valid data points available for plotting.")
    
    # 创建归一化
    if config.log_norm:
        norm = LogNorm(vmin=np.nanmin(valid_data), vmax=np.nanmax(valid_data))
    else:
        norm = plt.Normalize(vmin=np.nanmin(valid_data), vmax=np.nanmax(valid_data))
    
    sm = cm.ScalarMappable(norm=norm, cmap=config.cmap)
    sm.set_array(valid_data)
    
    # 绘制表面
    n_phi = len(phi_plot[:, 0])
    n_pol = len(R_plot[0, :])
    
    kwargs = {
        'rcount': n_phi,
        'ccount': n_pol,
        'lw': 0,
        'edgecolor': 'none',
        'antialiased': False,
        'shade': False
    }
    if debug:
        kwargs['rcount'] = 32
        kwargs['ccount'] = n_pol
        # kwargs['antialiased'] = True
    sc = ax.plot_surface(
        R_plot, phi_plot, Z_plot,
        facecolors=cm.get_cmap(config.cmap)(norm(data_plot)),
        cmap=config.cmap,
        alpha=1,
        **kwargs
    )
    
    # 标记最大值
    if config.find_max:
        plot_max_point(R_plot, Z_plot, phi_plot, data_plot, ax)
    
    # 设置坐标轴
    ax.set_aspect('equalxz')
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label(data.data_name, rotation=270, labelpad=15)
    ax.set_xlabel('R Axis', fontsize=10)
    ax.set_ylabel('Phi Axis', fontsize=10)
    ax.set_zlabel('Z Axis', fontsize=10)
    ax.view_init(elev=view_angle[0], azim=view_angle[1])
    
    # 保存或显示
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        if debug:
            print(f"[Plotting] Saved surface plot to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_max_point(R, Z, phi, val, ax):
    """
    标记最大值。
    计算phi面上RZ图形的重心，从而找到图形外部/内部的方向，
    并让最大值的R,Z坐标适当向外延伸一些。
    """
    # 展平数据
    R_flat = R.flatten()
    Z_flat = Z.flatten()
    phi_flat = phi.flatten()
    val_flat = val.flatten()

    # 找到最大值索引
    max_idx = np.nanargmax(val_flat)
    max_R = R_flat[max_idx]
    max_Z = Z_flat[max_idx]
    max_phi = phi_flat[max_idx]
    max_value = val_flat[max_idx]

    # 计算重心 (忽略NaN)
    centroid_R = np.nanmean(R_flat)
    centroid_Z = np.nanmean(Z_flat)
    centroid_R = 0.8
    centroid_Z = 0.0
    # 计算从重心指向最大值的向量
    vec_R = max_R - centroid_R
    vec_Z = max_Z - centroid_Z
    
    # 适当向外延伸 (例如1.1倍距离)
    scale = 1.01
    plot_R = centroid_R + vec_R * scale
    plot_Z = centroid_Z + vec_Z * scale

    ax.scatter([plot_R], [max_phi], [plot_Z], color='red', s=40,
               label=f'Max: {max_value:.2e}')
    #ax.legend()