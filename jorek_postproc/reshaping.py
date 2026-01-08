"""
Data reshaping and grid generation module.

Transforms unstructured 1D point data into structured 2D grids suitable for 3D surface plotting.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
from .data_models import BoundaryQuantitiesData


def reshape_to_grid(
    block: np.ndarray,
    col_names: List[str],
    names: Tuple[str, str, str, str],
    iplane: Optional[int] = None,
    xpoints: Optional[np.ndarray] = None,
    debug: bool = False
) -> BoundaryQuantitiesData:
    """
    将非结构化的1D点数据重整化为结构化的2D网格 (Toroidal x Poloidal)。

    该函数按环向平面 (Phi) 分组原始数据，在每个切面上进行极向排序，
    并堆叠成适合3D表面绘图的2D网格。

    Parameters
    ----------
    block : numpy.ndarray
        原始输入数据矩阵，形状为 (N_samples, N_features)
    col_names : list of str
        列名列表
    names : tuple of 4 str
        物理量列映射，预期顺序：[R_name, Z_name, phi_name, val_name]
    iplane : int, optional
        环向平面数（用于重整化检查）
    xpoints : numpy.ndarray, optional
        形状为 (2, 2) 的X点坐标数组，用于复杂几何的分段排序
        如果提供，排序会根据上/下X点分段进行
        如果为None，使用标准的重心角度排序
    debug : bool, optional
        调试模式标志，默认False

    Returns
    -------
    BoundaryQuantitiesData
        包含重整化2D网格的数据对象

    Raises
    ------
    ValueError
        如果列名在col_names中找不到
    """
    
    # 1. 解析列名对应关系
    r_col_name, z_col_name, phi_col_name, val_col_name = names
    if debug:
        print(f"[Reshaping] Mapping columns: R='{r_col_name}', Z='{z_col_name}', phi='{phi_col_name}', value='{val_col_name}'")
        print(f"[Reshaping] Input block shape: {block.shape}")
        print(f"[Reshaping] Using iplane={iplane}, xpoints={'provided' if xpoints is not None else 'not provided'}")
    try:
        r_idx = col_names.index(r_col_name)
        z_idx = col_names.index(z_col_name)
        phi_idx = col_names.index(phi_col_name)
        val_idx = col_names.index(val_col_name)
    except ValueError as e:
        raise ValueError(f"Column name mismatch: {e}")

    if debug:
        print(f"[Reshaping] Column indices: R={r_idx}, Z={z_idx}, phi={phi_idx}, value={val_idx}")
    # 提取原始一维数据
    R_raw = block[:, r_idx]
    Z_raw = block[:, z_idx]
    phi_raw = block[:, phi_idx]
    val_raw = block[:, val_idx]

    # 2. 识别唯一的 Phi 切面 (Toroidal Planes)
    unique_phi = np.unique(np.round(phi_raw, 5))
    unique_phi.sort()
    # unique_phi = unique_phi - unique_phi[0]  # 使第一个切面为0
    
    n_phi = len(unique_phi)
    if debug:
        print(f"[Reshaping] Detected {n_phi} toroidal planes (phi slices).")

    # 容器：用于存放整理后的每一圈的数据
    R_slices = []
    Z_slices = []
    Phi_slices = []
    Val_slices = []
    points_per_slice = []

    # 3. 遍历每个 Phi 切面进行 R-Z 排序
    for current_phi in unique_phi:
        # 3.1 提取当前切面的所有点
        mask = np.abs(phi_raw - current_phi) < 1e-4
        
        if debug:
            print(f"[Reshaping] Processing phi={current_phi:.5f} with {np.sum(mask)} points.")
        
        if xpoints is not None:
            # X点分段排序（用于双X点撕裂模）
            c_rup, c_zup = xpoints[1, :]
            c_rdn, c_zdn = xpoints[0, :]
            mask_xpt_up = Z_raw[mask] >= 0
            mask_xpt_dn = Z_raw[mask] < 0
            
            r_slice_up = R_raw[mask][mask_xpt_up]
            z_slice_up = Z_raw[mask][mask_xpt_up]
            r_slice_dn = R_raw[mask][mask_xpt_dn]
            z_slice_dn = Z_raw[mask][mask_xpt_dn]
            
            
            angles_up = np.arctan2(z_slice_up - c_zup, r_slice_up - c_rup)
            angles_dn = np.arctan2(z_slice_dn - c_zdn, r_slice_dn - c_rdn)
            
            angle_up_0 = np.arctan2(-c_zup, np.max(r_slice_up) - c_rup)
            angle_down_0 = np.arctan2(-c_zdn, np.min(r_slice_dn) - c_rdn)
            
            idx_angle_up_0 = np.argmin(np.abs(angles_up - angle_up_0))
            idx_angle_down_0 = np.argmin(np.abs(angles_dn - angle_down_0))
            sort_org_up = np.argsort(angles_up)
            sort_org_dn = np.argsort(angles_dn)
            
            sort_idx_up = np.roll(sort_org_up, -np.where(sort_org_up == idx_angle_up_0)[0][0])
            sort_idx_dn = np.roll(sort_org_dn, -np.where(sort_org_dn == idx_angle_down_0)[0][0])
            
            R_slices.append(np.concatenate((r_slice_up[sort_idx_up], r_slice_dn[sort_idx_dn])))
            Z_slices.append(np.concatenate((z_slice_up[sort_idx_up], z_slice_dn[sort_idx_dn])))
            Phi_slices.append(np.full_like(
                np.concatenate((r_slice_up[sort_idx_up], r_slice_dn[sort_idx_dn])), 
                current_phi
            ))
            Val_slices.append(np.concatenate((
                val_raw[mask][mask_xpt_up][sort_idx_up],
                val_raw[mask][mask_xpt_dn][sort_idx_dn]
            )))
            
            points_per_slice.append(len(r_slice_up) + len(r_slice_dn))
        
        else:
            # 标准重心角度排序
            r_slice = R_raw[mask]
            z_slice = Z_raw[mask]
            v_slice = val_raw[mask]
            
            if len(r_slice) == 0:
                continue
            
            # 计算重心
            c_r = np.mean(r_slice)
            c_z = np.mean(z_slice)
        
            if debug:
                print(f"[Reshaping] Phi={current_phi:.5f}: Centroid at (R={c_r:.3f}, Z={c_z:.3f}), Points={len(r_slice)}")
        
            # 计算每个点相对于重心的角度
            angles = np.arctan2(z_slice - c_z, r_slice - c_r)
            sort_idx = np.argsort(angles)
            
            R_slices.append(r_slice[sort_idx])
            Z_slices.append(z_slice[sort_idx])
            Phi_slices.append(np.full_like(r_slice, current_phi))
            Val_slices.append(v_slice[sort_idx])
            
            points_per_slice.append(len(r_slice))

    # 4. 数据对齐检查
    if len(set(points_per_slice)) > 1:
        if debug:
            print(f"[Reshaping] Warning: Points per slice vary: {set(points_per_slice)}")
        min_points = min(points_per_slice)
        # 截断到最小公共大小
        for i in range(len(R_slices)):
            R_slices[i] = R_slices[i][:min_points]
            Z_slices[i] = Z_slices[i][:min_points]
            Phi_slices[i] = Phi_slices[i][:min_points]
            Val_slices[i] = Val_slices[i][:min_points]
    
    # 5. 堆叠成 2D 矩阵 (N_phi x N_poloidal)
    R_grid = np.vstack(R_slices)
    Z_grid = np.vstack(Z_slices)
    Phi_grid = np.vstack(Phi_slices)
    Val_grid = np.vstack(Val_slices)
    
    if debug:
        print(f"[Reshaping] Reshaped grid size: {R_grid.shape}")
    
    return BoundaryQuantitiesData(
        R=R_grid,
        Z=Z_grid,
        phi=Phi_grid,
        data=Val_grid,
        data_name=val_col_name,
        grid_shape=R_grid.shape
    )
