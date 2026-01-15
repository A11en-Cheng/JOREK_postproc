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
    # cbar.set_label(data.data_name, rotation=270, labelpad=15)
    ax.set_aspect('equalxz')
    ax.set_xlabel(fr'$R$ Axis', fontsize=10)
    ax.set_ylabel(fr'$\phi$ Axis', fontsize=10)
    ax.set_zlabel(fr'$Z$ Axis', fontsize=10)
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
    #cbar.set_label(data.data_name, rotation=270, labelpad=15)
    ax.set_xlabel(fr'$R$ Axis', fontsize=10)
    ax.set_ylabel(fr'$\phi$ Axis', fontsize=10)
    ax.set_zlabel(fr'$Z$ Axis', fontsize=10)
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

def plot_heat_flux_analysis(
    data: BoundaryQuantitiesData,
    config: Optional[PlottingConfig] = None,
    save_path: Optional[str] = None,
    debug: bool = False
) -> None:
    """
    绘制完整的热流分析图集：
    1. 选定phi面上的R-Z边界轮廓（左图）
    2. 展开的phi-theta热流分布图（右图）
    并在两图上同步标记出偏滤器等特征位置。

    Parameters
    ----------
    data : BoundaryQuantitiesData
        处理后的边界数据 (必须包含theta信息)
    config : PlottingConfig
    save_path : str
    """
    if config is None:
        config = PlottingConfig()
        
    if data.theta is None:
        if debug: print("[Plotting] Warning: No theta data available, cannot plot phi-theta map.")
        return

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- 左图：R-Z 边界轮廓 (取 phi=0 的截面) ---
    # 假设数据已经是 grid (phi, pol)
    # 取第一圈 phi 对应的 poloidal 数据
    try:
        if data.is_2d_grid():
            R_pol = data.R[0, :]
            Z_pol = data.Z[0, :]
            theta_pol = data.theta[0, :]
        else:
            # 如果是散点，稍微麻烦点，这里先只支持 grid
            if debug: print("Data is not grid, skipping R-Z plot")
            return
            
        ax1.plot(R_pol, Z_pol, 'k-', linewidth=1.5, label='Boundary')
        ax1.set_aspect('equal')
        ax1.set_xlabel('R [m]')
        ax1.set_ylabel('Z [m]')
        ax1.set_title(f'Poloidal Cross-section (phi={data.phi[0,0]:.2f})')
        ax1.grid(True, alpha=0.3)

        # 定义关键位置 (Div targets) 的 theta 范围或点
        # 这里的坐标点参考了用户提供的脚本中的注释
        # inner_upper, outer_upper, outer_lower, inner_lower
        # 对应大概的 theta 值 (需要根据实际平衡确认，这里沿用用户脚本的数值作为示例)
        key_points = [
            {'label': 'IU', 'theta': 1.95, 'color': 'red'},
            {'label': 'OU', 'theta': 1.45, 'color': 'orange'},
            {'label': 'OL', 'theta': 4.86, 'color': 'green'},
            {'label': 'IL', 'theta': 4.33, 'color': 'blue'}
        ]

        for kp in key_points:
            # 在 R-Z 轮廓上找到对应 theta 最近的点
            idx = np.argmin(np.abs(theta_pol - kp['theta']))
            r_pt, z_pt = R_pol[idx], Z_pol[idx]
            
            # 绘制点和标签
            ax1.scatter(r_pt, z_pt, c=kp['color'], s=50, zorder=5)
            ax1.annotate(kp['label'], (r_pt, z_pt), 
                         xytext=(5, 5), textcoords='offset points', 
                         color=kp['color'], fontweight='bold')
    except Exception as e:
        if debug: print(f"Error plotting R-Z contour: {e}")

    # --- 右图：Phi-Theta 热流分布 ---
    # data.phi 形状 (N_phi, N_pol)
    # data.theta 形状 (N_phi, N_pol) -> 用第一行的 theta 做 x 轴即可 (忽略 theta随phi微小的变化)
    # data.data 形状 (N_phi, N_pol)
    
    #构造网格
    # Y轴: Phi (0 ~ 2pi)
    # X轴: Theta (0 ~ 2pi)
    
    # 展平做 pcolormesh
    # 注意：jorek 的 phi 可能是 multiharmonic展开的，这里假设已经是物理空间的 grid
    phi_axis = data.phi[:, 0]
    theta_axis = data.theta[0, :] # 假设所有截面 theta 分布一致
    
    # 数据矩阵转置? 
    # data.data[i_phi, i_pol] -> row 是 phi, col 是 theta
    # pcolormesh(X, Y, C) -> X 是列坐标(theta), Y 是行坐标(phi)
    
    # 简单的归一化
    val = data.data
    if config.data_limits:
        val = np.clip(val, config.data_limits[0], config.data_limits[1])
        
    if config.log_norm:
        norm = LogNorm(vmin=np.nanmin(val[val>0]), vmax=np.nanmax(val))
    else:
        norm = plt.Normalize(vmin=np.nanmin(val), vmax=np.nanmax(val))
        
    # 绘制热图
    # 注意 pcolormesh 需要网格边缘，这里简化直接用中心点坐标会自动居中
    mesh = ax2.pcolormesh(theta_axis, phi_axis, val, 
                          cmap=config.cmap, norm=norm, shading='auto')
    
    # 标记关键位置的竖线
    for kp in key_points:
        ax2.axvline(x=kp['theta'], color=kp['color'], linestyle='--', alpha=0.8)
        ax2.text(kp['theta'], phi_axis.max(), kp['label'], 
                 color=kp['color'], ha='center', va='bottom', fontweight='bold')

    fig.colorbar(mesh, ax=ax2, label=data.data_name)
    ax2.set_xlabel('Theta (Poloidal Angle)')
    ax2.set_ylabel('Phi (Toroidal Angle)')
    ax2.set_title(f'Heat Flux Distribution: {data.data_name}')
    
    # 反转 X 轴? 通常 theta 从 0 到 2pi
    ax2.set_xlim(theta_axis.min(), theta_axis.max())

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=config.dpi)
        if debug: print(f"Saved heat flux analysis to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)