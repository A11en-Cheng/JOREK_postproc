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
from typing import Optional, Tuple, List, Dict

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
    ax.set_xlabel(fr'$R$ Axis', fontsize=16)
    ax.set_ylabel(fr'$\phi$ Axis', fontsize=16)
    ax.set_zlabel(fr'$Z$ Axis', fontsize=16)
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
    regions: Optional[List[Dict]] = None,
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
    regions : list of dict
        关键区域标记 [{'label': 'IU', 'mask': bool_array_2d, 'color': 'red'}, ...]
    """
    if config is None:
        config = PlottingConfig()
        
    if data.theta is None:
        if debug: print("[Plotting] Warning: No theta data available, cannot plot phi-theta map.")
        return

    if debug:
        print("[Plotting] Generating heat flux analysis plots...")
        print(f"  Data shape: R{data.R.shape}, Z{data.Z.shape}, phi{data.phi.shape}, theta{data.theta.shape}, data{data.data.shape}")
        print(f"  Data range: [{data.data.min():.2e}, {data.data.max():.2e}]")
        print(f"  theta range: [{data.theta.min():.2e}, {data.theta.max():.2e}]")
    # 创建画布
    plt.rcParams.update({'font.size': 16}) # 增大字体
    
    # 确定子图布局
    if config.show_left_plot and config.show_right_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), dpi=200)
    elif config.show_left_plot:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 10), dpi=200)
        ax2 = None
    elif config.show_right_plot:
        fig, ax2 = plt.subplots(1, 1, figsize=(10, 8), dpi=200)
        ax1 = None
    else:
        if debug: print("[Plotting] No plots selected, skipping.")
        return

    # --- 左图：R-Z 边界轮廓 (取 phi=0 的截面) ---
    if ax1 is not None:
        try:
            if data.is_2d_grid():
                R_pol = data.R[0, :]
                Z_pol = data.Z[0, :]
                theta_pol = data.theta[0, :]
            else:
                if debug: print("Data is not grid, skipping R-Z plot")
                if ax2 is None: return # nothing to plot
                
            ax1.plot(R_pol, Z_pol, 'gray', linewidth=1.5, alpha=0.9, label='Boundary') # Changed to gray
            ax1.set_aspect('equal')
            ax1.set_xlabel(r'$R$ [m]',fontsize=16)
            ax1.set_ylabel(r'$Z$ [m]',fontsize=16)
            # ax1.set_title(f'Poloidal Cross-section')
            ax1.grid(True, alpha=0.3)

            if regions:
                for reg in regions:
                    # 左图：绘制区域对应的线段 (取第一圈 phi=0 对应的 mask)
                    # mask 形状假设为 (N_phi, N_pol)，取 [0, :]
                    if 'mask' in reg:
                        mask_pol = reg['mask'][0, :]
                        
                        # 更好的方法：分段绘制
                        # 但为了简单，先画点，点够密就是线
                        ax1.plot(R_pol[mask_pol], Z_pol[mask_pol], 
                                color=reg['color'], linewidth=3, label=reg['label'])
                        
                        # 标注文字 (在区域中心)
                        if np.any(mask_pol):
                            c_r = np.mean(R_pol[mask_pol])
                            c_z = np.mean(Z_pol[mask_pol])
                            # Shift label slightly based on position
                            offset_r = 0.05 if c_r > 0.8 else -0.05
                            ax1.text(c_r, c_z, reg['label'], color=reg['color'], 
                                    fontweight='bold', fontsize=12, ha='center', va='center', # Smaller font
                                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        except Exception as e:
            if debug: 
                print(f"Error plotting R-Z contour: {e}")
                import traceback
                traceback.print_exc()

    # --- 右图：Phi-Theta 热流分布 ---
    if ax2 is not None:
        # 转置坐标系：Phi -> X, Theta -> Y
        phi_axis = data.phi[:, 0]
        
        # Determine Y-axis (Theta or Arc Length)
        use_len = config.use_arc_length and data.arc_length is not None
        if use_len:
            if debug: print("[Plotting] Using Arc Length for Y-axis")
            y_axis_raw = data.arc_length[0, :]
            y_label = r'Length [m]'
        else:
            if debug: print("[Plotting] Using Theta for Y-axis")
            y_axis_raw = data.theta[0, :] 
            y_label = r'$\theta$ (Poloidal Angle)'
    
        val = data.data 
        # val shape: (N_phi, N_theta)
        
        # Sort y-axis to ensure monotonicity for pcolormesh
        sort_idx = np.argsort(y_axis_raw)
        y_axis_sorted = y_axis_raw[sort_idx]
        val = val[:, sort_idx]

        if config.data_limits:
            val = np.clip(val, config.data_limits[0], config.data_limits[1])
            
        if config.log_norm:
            norm = LogNorm(vmin=np.nanmin(val[val>0]), vmax=np.nanmax(val))
        else:
            norm = plt.Normalize(vmin=np.nanmin(val), vmax=np.nanmax(val))
        
        # pcolormesh(X, Y, C)
        # X: Phi (N_phi), Y: Theta (N_theta), C: (N_theta, N_phi) -> Transpose val
        mesh = ax2.pcolormesh(phi_axis, y_axis_sorted, val.T, 
                              cmap='viridis', norm=norm, shading='gouraud') 
        
        # 标记区域框
        if regions:
            for reg in regions:
                if 'mask' in reg:
                    # 找出该区域对应的 Y 轴范围
                    mask_pol = reg['mask'][0, :]
                    if np.any(mask_pol):
                        # Use UNSORTED original Y data to match mask indices
                        y_vals_in_region = y_axis_raw[mask_pol]
                        y_min = np.min(y_vals_in_region)
                        y_max = np.max(y_vals_in_region)
                        
                        # 绘制图边粗线段标注范围 (左侧/Minimum Phi)
                        line_x = phi_axis.min()
                        ax2.plot([line_x, line_x], [y_min, y_max], 
                                 color=reg['color'], linewidth=8, solid_capstyle='butt')
                        
                        # 标注文字 (文字在内部，靠左对齐，略微右移)
                        ax2.text(line_x + (phi_axis.max()-phi_axis.min())*0.01, (y_min + y_max)/2, f"{reg['label']}", 
                                 color=reg['color'], va='center', ha='left', fontweight='bold', fontsize=16)

        cbar = fig.colorbar(mesh, ax=ax2, label=data.data_name)
        
        if use_len:
            # 保持画面为正方形 (Square Frame)
            ax2.set_box_aspect(1)
        else:
            # Theta 模式不强制正方形，使用默认纵横比
            # ax2.set_aspect('equal') # Removed
            pass

        ax2.set_xlabel(r'$\phi$ (Toroidal Angle)', fontsize=16)
        ax2.set_ylabel(y_label, fontsize=16)
        
        title_str = f'Heat Flux'
        if data.time:
            title_str += f' (t={data.time*1e3:.4f}ms)'
        ax2.set_title(title_str, fontsize=18)
        
        # 调整显示范围
        ax2.set_xlim(phi_axis.min(), phi_axis.max())
        # ax2.set_ylim(theta_axis.min(), theta_axis.max())

    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=config.dpi)
        if debug: print(f"Saved heat flux analysis to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)

def plot_set(
    data: BoundaryQuantitiesData,
    config: Optional[PlottingConfig] = None,
    save_path: Optional[str] = None,
    regions: Optional[List[Dict]] = None,
    debug: bool = False
) -> None:
    """
    绘制完整的边界分析图集（plot_set）。
    别名函数，调用 plot_heat_flux_analysis。
    
    Parameters
    ----------
    data : BoundaryQuantitiesData
        边界数据（需包含R, Z, phi, theta, data）
    config : PlottingConfig, optional
        绘图配置
    save_path : str, optional
        保存路径
    regions : list of dict, optional
         关键区域标记 [{'label': 'IU', 'mask': bool_array_2d, 'color': 'red'}, ...]
    debug : bool
        调试信息
    """
    if debug:
        print("[Plotting] Executing plot_set...")
    plot_heat_flux_analysis(data, config, save_path, regions, debug)
