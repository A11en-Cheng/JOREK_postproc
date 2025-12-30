#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import matplotlib.tri as mtri
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def read_boundary_file(file_path, DEBUG=0):
    """优化后的文件读取函数，使用numpy直接加载数据"""
    blocks = {}
    current_block = None
    data_rows = []
    col_names = []
    t_now = np.nan
    t_nowall = {}
    
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('# '):
                if line.startswith('# time step'):
                    t_now = float(line.split()[-1])
                    step_now = line.split()[3][1:].strip(',')
                    t_nowall[step_now] = t_now
                    if current_block and data_rows:
                        blocks[current_block] = np.array(data_rows, dtype=np.float32)
                        data_rows = []
                    block_name = line.split()[3][1:-1]
                    current_block = block_name
                else:
                    col_names = line.split()[1:]
            else:
                data_rows.append(list(map(float, line.split())))

        if current_block and data_rows:
            blocks[current_block] = np.array(data_rows, dtype=np.float32)

    if DEBUG:
        print(f"Columns: {col_names}, Blocks: {list(blocks.keys())}")
    
    return col_names, blocks, t_nowall


def reshape_to_grid_updated(block, col_names, names, iplane=None,xpoints=None):
    """
    Reshapes unstructured 1D point data into structured 2D grids (Toroidal x Poloidal).

    This function groups raw data by unique toroidal planes (Phi), performs poloidal sorting 
    on the (R, Z) coordinates within each slice, and stacks them into 2D meshes suitable 
    for surface plotting.

    Parameters
    ----------
    block : numpy.ndarray
        The raw input data matrix of shape (N_samples, N_features).
    col_names : list of str
        The list of column headers corresponding to the columns in `block`.
    names : list or tuple
        A list of 4 strings mapping specific physics quantities to column names.
        Expected order: [R_name, Z_name, phi_name, val_name].
    iplane : int, optional
        Index of the plane (currently unused/reserved).
    xpoints : numpy.ndarray, optional
        Array of shape (2, 2) defining X-point coordinates for complex geometries.
        If provided, sorting is split into upper/lower segments based on these points.
        If None, a standard centroid-based angular sorting is applied.

    Returns
    -------
    dict
        A dictionary containing the reshaped 2D grids keyed by the original column names:
        - R (N_phi, N_poloidal)
        - Z (N_phi, N_poloidal)
        - Phi (N_phi, N_poloidal)
        - Value (N_phi, N_poloidal)

    Raises
    ------
    ValueError
        If the specified column names in `names` are not found in `col_names`.
    """
    
    # 1. 解析列名对应关系
    # names input expected: [R_name, Z_name, phi_name, val_name]
    r_col_name, z_col_name, phi_col_name, val_col_name = names
    
    try:
        r_idx = col_names.index(r_col_name)
        z_idx = col_names.index(z_col_name)
        phi_idx = col_names.index(phi_col_name)
        val_idx = col_names.index(val_col_name)
    except ValueError as e:
        raise ValueError(f"Column name mismatch: {e}")

    # 提取原始一维数据
    R_raw = block[:, r_idx]
    Z_raw = block[:, z_idx]
    phi_raw = block[:, phi_idx]
    val_raw = block[:, val_idx]

    # 2. 识别唯一的 Phi 切面 (Toroidal Planes)
    # 由于浮点误差，需要保留几位小数后取唯一值
    # 假设 phi 是主要的切分维度
    unique_phi = np.unique(np.round(phi_raw, 5))
    unique_phi.sort() # 确保 phi 从小到大排列
    
    n_phi = len(unique_phi)
    print(f"Detected {n_phi} toroidal planes (phi slices).")

    # 容器：用于存放整理后的每一圈的数据
    R_slices = []
    Z_slices = []
    Phi_slices = [] # 虽然每层一样，但为了生成网格，还是存一下
    Val_slices = []
    
    points_per_slice = []

    # 3. 遍历每个 Phi 切面进行 R-Z 排序
    for current_phi in unique_phi:
        # 3.1 提取当前切面的所有点
        # 使用近似匹配防止浮点数精度问题
        mask = np.abs(phi_raw - current_phi) < 1e-4
        
        if xpoints is not None: # todo: this is currently for dxpoint; need to be improved for general use
            c_rup,c_zup = xpoints[1,:]
            c_rdn,c_zdn = xpoints[0,:]
            mask_xpt_up = Z_raw[mask]>=0
            mask_xpt_dn = Z_raw[mask]<0
            
            
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
            
            
            R_slices.append(np.concatenate((r_slice_up[sort_idx_up],r_slice_dn[sort_idx_dn])))
            Z_slices.append(np.concatenate((z_slice_up[sort_idx_up],z_slice_dn[sort_idx_dn])))
            Phi_slices.append(np.full_like(np.concatenate((r_slice_up[sort_idx_up],r_slice_dn[sort_idx_dn])), current_phi))
            Val_slices.append(np.concatenate((val_raw[mask][mask_xpt_up][sort_idx_up],val_raw[mask][mask_xpt_dn][sort_idx_dn])))
            
            points_per_slice.append(len(r_slice_up)+len(r_slice_dn))
        
        
        else:
            r_slice = R_raw[mask]
            z_slice = Z_raw[mask]
            v_slice = val_raw[mask]
            
            if len(r_slice) == 0: continue
            
            # 3.2 核心步骤：在 (R, Z) 平面上对点进行几何排序
            # 针对闭合磁面或环形结构，使用“重心角度法”排序最稳健
            # 计算重心 (Centroid)
            c_r = np.mean(r_slice)
            c_z = np.mean(z_slice)
        
        
            print(f"Phi={current_phi:.5f}: Centroid at (R={c_r:.3f}, Z={c_z:.3f}), Points={len(r_slice)}")
        
            # 计算每个点相对于重心的角度 (-pi 到 pi)
            # arctan2(y, x) -> arctan2(z - cz, r - cr)
            angles = np.arctan2(z_slice - c_z, r_slice - c_r)
            
            # 获取排序后的索引
            sort_idx = np.argsort(angles)
            
            # 存入排序后的切片
            R_slices.append(r_slice[sort_idx])
            Z_slices.append(z_slice[sort_idx])
            # Phi 这一层全是同一个值，但也需要展开成数组以便后续堆叠
            Phi_slices.append(np.full_like(r_slice, current_phi))
            Val_slices.append(v_slice[sort_idx])
            
            points_per_slice.append(len(r_slice))

    # 4. 数据对齐检查与网格化
    # plot_surface 要求矩阵每一行的列数必须相同。
    # 如果不同 phi 切面上的点数不一样（例如网格有缺失），这里需要处理。
    
    # 检查是否所有切面的点数一致
    if len(set(points_per_slice)) > 1:
        print(f"Warning: Points per slice vary: {set(points_per_slice)}")
        print("Truncating to minimum common size (or you might want to interpolate).")
        min_points = min(points_per_slice)
        # 简单截断策略：只取前 min_points 个点（前提是已经按角度排好序了，截断只会少一小段）
        # 更优策略是插值，但为了代码简洁先做截断/重采样
        for i in range(len(R_slices)):
            # 这里为了保证闭合性，最好是重采样而不是直接截断。
            # 这里暂时使用简单的切片，实际物理数据通常点数是一致的。
            R_slices[i] = R_slices[i][:min_points]
            Z_slices[i] = Z_slices[i][:min_points]
            Phi_slices[i] = Phi_slices[i][:min_points]
            Val_slices[i] = Val_slices[i][:min_points]
    
    # 5. 堆叠成 2D 矩阵 (N_phi x N_poloidal)
    R_grid = np.vstack(R_slices)
    Z_grid = np.vstack(Z_slices)
    Phi_grid = np.vstack(Phi_slices)
    Val_grid = np.vstack(Val_slices)
    
    print(f"Reshaped grid size: {R_grid.shape}")
    
    # 6. 返回结果
    # 注意：为了配合 plot_surface，这里返回完整的网格矩阵
    sorted_df = {
        r_col_name: R_grid,
        z_col_name: Z_grid,
        phi_col_name: Phi_grid,
        val_col_name: Val_grid
    }
    
    print('Sorted and Reshaped Successfully.')
    return sorted_df

def process_timestep(args,filename = None):
    """处理单个时间步的任务函数"""
    ts, file_addr, iplane, names, xpoints = args
    if filename == None:
        file_name = f'boundary_quantities_s0{ts}.dat'
    else:
        file_name = filename
    file_path = os.path.join(file_addr, file_name)
    
    try:
        col_names, blocks, t_now = read_boundary_file(file_path)
        
        block_data = blocks.get(ts)
        if block_data is None:
            return ts, None
        
        print(f"Processing timestep {ts} with {block_data.shape[0]} points.")
        grid_data = reshape_to_grid_updated(block_data, col_names, names, iplane,xpoints=xpoints)

        return t_now, grid_data
    except Exception as e:
        print(f"Error processing timestep {ts}: {e}")
        return ts, None


def plot_scatter_from_scatter_dict(fig, ax, data, log, names, time_phys, cmap='viridis', fig_destiny='figure_3d_scatter', angs=(30,30), mask_name='',find_max=False,DEBUG=False,test_flag=False):
    R = np.array(data['R'])
    Z = np.array(data['Z'])
    phi = np.array(data['phi'])
    val = np.array(data['val'])
   
    if log:
        norm = LogNorm(val.min(), val.max())
    else:
        norm = plt.Normalize(val.min(), val.max())
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(val)
    
    sc = ax.scatter(
        R, phi, Z,
        c=val, 
        cmap=cmap,
        s=2,          # Point size
        alpha=0.5
    )
    
    if find_max:
        max_idx = np.argmax(val)
        max_R = R[max_idx]
        max_Z = Z[max_idx]
        max_phi = phi[max_idx]
        max_value = val[max_idx]
        ax.scatter([max_R], [max_phi], [max_Z], color='red', s=40, label=f'Max: {max_value:.2e}')

    cbar = fig.colorbar(sm, pad=0.1)
    cbar.set_label('Value Intensity', rotation=270, labelpad=15)
    ax.set_xlabel('R Axis', fontsize=10)
    ax.set_ylabel('phi Axis', fontsize=10)
    ax.set_zlabel('Z Axis', fontsize=10)
    ax.set_aspect('equalxz')
    #ax.set_title(f'{names[-1]} at t={time_phys}ms')
    ax.view_init(*angs)  # 解包元组，确保传入两个数值
    if DEBUG or test_flag:
        plt.show()
    else:
        if mask_name != '':
            plt.savefig(os.path.join(fig_destiny, f'3d_plot_{time_phys}ms_{names[-1]}_scatter_{mask_name}.png'), dpi=300)
        else:
            plt.savefig(os.path.join(fig_destiny, f'3d_plot_{time_phys}ms_{names[-1]}_scatter_overall.png'), dpi=300)
    plt.close(fig)

def plot_surface_from_scatter_dict(fig, ax, data, log, names, time_phys, mask, iplane, cmap='viridis', fig_destiny='figure_3d_surface', angs=(30,30), mask_name='',find_max=False,DEBUG=False,test_flag=False):

    # 保持二维形状：在 mask 为 False 的位置填充 NaN（plot_surface 会忽略 NaN 区域）
    R_set_new = np.where(mask, data['R'], np.nan)
    Z_set_new = np.where(mask, data['Z'], np.nan)
    phi_set_new = np.where(mask, data['phi'], np.nan)
    data_new = np.where(mask, data['val'], np.nan)

    # 确保 data_new 的最小值和最大值有效
    valid_data = data_new[~np.isnan(data_new)]
    if valid_data.size == 0:
        raise ValueError("No valid data points available for plotting.")

    if log:
        norm = LogNorm(valid_data.min(), valid_data.max())
    else:
        norm = plt.Normalize(valid_data.min(), valid_data.max())
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(valid_data)
    kwargs = {'rcount': len(phi_set_new[:,0]), 'ccount': len(R_set_new[0,:]), 'lw': 0, 'edgecolor': 'none', 'antialiased': False, 'shade': False}
    if DEBUG or test_flag:
        kwargs['rcount'] = 100
        kwargs['ccount'] = 100
        kwargs['antialiased'] = True
    sc = ax.plot_surface(
        R_set_new, phi_set_new, Z_set_new,
        facecolors=cm.viridis(norm(data_new)),
        cmap=cmap,
        alpha=1,
        **kwargs
    )
    if find_max:
        max_idx = np.nanargmax(data_new)
        max_R = R_set_new.flatten()[max_idx]*1.01
        max_Z = Z_set_new.flatten()[max_idx]*1.01
        max_phi = phi_set_new.flatten()[max_idx]*1.01
        max_value = data_new.flatten()[max_idx]
        ax.scatter([max_R], [max_phi], [max_Z], color='red', s=40, label=f'Max: {max_value:.2e}')

    cbar = plt.colorbar(sm, pad=0.1)
    cbar.set_label('Value Intensity', rotation=270, labelpad=15)
    ax.set_xlabel('R Axis', fontsize=10)
    ax.set_ylabel('phi Axis', fontsize=10)
    ax.set_zlabel('Z Axis', fontsize=10)
    ax.set_aspect('equalxz')
    #ax.set_title(f'{names[-1]} at t={time_phys}ms')
    ax.view_init(*angs)  # 解包元组，确保传入两个数值
    if DEBUG or test_flag:
        plt.show()
    else:
        
        plt.savefig(os.path.join(fig_destiny, f'3d_plot_{str(time_phys)}ms_{names[-1]}_surface_{mask_name}.png'), dpi=300)
        #plt.show()
    plt.close(fig)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process and plot boundary quantities.")
    parser.add_argument("-f", "--file", required=True, help="Path to the boundary quantities file.",default='/home/ac_desktop/syncfiles/postproc_145/boundary_quantities_s04200.dat')
    parser.add_argument("-t", "--time", required=True, type=str, help="Time step to process.", default='4200')
    parser.add_argument("--iplane", required=True, type=int, help="Number of planes.", default=1080)
    parser.add_argument("--name", required=True, help="Name of the data column to process.", default='heatF_tot_cd')
    parser.add_argument("-lim","--limit", nargs='+', type=float, default=[1e5], help="Data limits for plotting.")
    parser.add_argument("-nf", "--norm_factor", type=float, default=4.1006E-07, help="Normalization factor for data.")
    parser.add_argument("-sf", "--plot_surface", action="store_true", default=True, help="Enable surface plotting from scatter data.")
    parser.add_argument("-o", "--overall", action="store_true",default=False, help="Enable overall plotting.")
    parser.add_argument("-debug","--DEBUG", action="store_true", default=False, help="Enable debug mode.")
    parser.add_argument("-test", "--test_flag", action="store_true", default=False, help="Enable test flag for new feature.")
    parser.add_argument("-log", "--log_norm", action="store_true", default=False, help="Enable logarithmic normalization for color mapping.")
    parser.add_argument("-xpt", "--xpoints", nargs='+',type=float, default=None, help="Xpoints positions for slicing. If the surface run into folding, please provide two Xpoints positions as four float numbers: x1 z1 x2 z2")
    parser.add_argument("-m", "--find_max", action="store_true", default=True,help="Plot maximum value location.")
    return parser.parse_args()

def debug_parse_arguments():
    class Args:
        def __init__(self):
            self.file = '/home/ac_desktop/syncfiles/postproc_145/boundary_quantities_s04200.dat'
            self.time = '4200'
            self.iplane = 1080
            self.name = 'heatF_tot_cd'
            self.limit = [1e5]
            self.norm_factor = 4.1006E-07
            self.plot_surface = True
            self.overall = True
            self.DEBUG = True
            self.test_flag = True
            self.log_norm = True
            self.xpoints = [[0.73, 0.877], [0.75, -0.8]]
            self.find_max = False
    return Args()

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def mask_choose(R_set_org,Z_set_org,mask):
    if mask == 'EXL50U':
        masks = {
            'mask_UO': (R_set_org >= 0.6) & (R_set_org <= 1.1) & (Z_set_org >= 1.0) & (Z_set_org <= 1.6),
            'mask_LO': (R_set_org >= 0.6) & (R_set_org <= 1.1) & (Z_set_org <= -1.0) & (Z_set_org >= -1.6),
            'mask_UI': (R_set_org >= 0.3) & (R_set_org <= 0.6) & (Z_set_org <= 1.2) & (Z_set_org >= 0.75),
            'mask_LI': (R_set_org >= 0.3) & (R_set_org <= 0.6) & (Z_set_org <= -0.75) & (Z_set_org >= -1.2),
        }
        angs = {
            'mask_UO': (36, 27),
            'mask_LO': (-36, -27),
            'mask_UI': (23, 163),
            'mask_LI': (-23, -163),
        }
    elif mask == 'ITER':
        masks = {
            'mask_UO'   :  (R_set_org >= 3  ) & (R_set_org <= 3.5) & (Z_set_org >= 2.5 ) & (Z_set_org <= 3   ),
            'mask_LO'   :  (R_set_org >= 3  ) & (R_set_org <= 3.5) & (Z_set_org <= -2.4) & (Z_set_org >= -3  ),
            'mask_UI'   :  (R_set_org >= 2.2) & (R_set_org <= 2.5) & (Z_set_org <= 2.8 ) & (Z_set_org >= 2.3 ),
            'mask_LI'   :  (R_set_org >= 2.2) & (R_set_org <= 2.5) & (Z_set_org <= -2.0) & (Z_set_org >= -2.8),
        }
        angs = {
            'mask_UO'    :  (40,45),
            'mask_LO'    :  (-40,-45),
            'mask_UI'    :  (20,150),
            'mask_LI'    :  (-20,-150),
        }
    else:
        raise ValueError(f"Unknown mask type: {mask}")
    return masks, angs

def main():
    global DEBUG
    global test_flag
    
    args = parse_arguments()
    DEBUG = False
    test_flag = False
    if args.DEBUG:
        DEBUG = True
        args = debug_parse_arguments()
        
    if args.test_flag:

        test_flag = True
    xpoints = None
    if args.xpoints is not None:
        print(type(args.xpoints))
        xpoints = np.sort(np.array(args.xpoints, dtype=float).reshape(2, -1),axis=0)
        print(xpoints)
    file_addr = os.path.dirname(args.file)
    
    if args.plot_surface:
        fig_destiny = os.path.join(os.getcwd(), f"figure_3d_surface_{args.time}")
    else:
        fig_destiny = os.path.join(os.getcwd(), f"figure_3d_scatter_{args.time}")
    ensure_directory_exists(fig_destiny)


    tss = [args.time]
    iplane = args.iplane
    names = ['R', 'Z', 'phi', args.name]
    norm_factor = 4.1006E-07  # jorekU to s conversion factor
    lim = [1e5]

    data_set = {}
    for ts in tss:
        ts = str(ts).zfill(6)
        t_now, q_data = process_timestep((ts, file_addr, iplane, names, xpoints), filename=os.path.basename(args.file))
        t_phys = t_now[ts] * norm_factor
        data_set[t_now[ts]] = q_data
        if args.DEBUG:
            print(f"Processed time {t_now[ts]:.2e} with shape {q_data['R'].shape}, {q_data['Z'].shape}, {q_data['phi'].shape}, t_phys={t_phys:.2e}")


    if args.overall: # plot overall
        for key in data_set.keys():
            time_phys = round(key * norm_factor * 1e3, 4)  # ms
            R_set_org = data_set[key]['R']
            Z_set_org = data_set[key]['Z']
            phi_set_org = data_set[key]['phi']
            data_org = np.where(data_set[key][names[-1]] >= lim[0], data_set[key][names[-1]], lim[0])
            try:
                if lim[1] <= lim[0]:
                    raise ValueError(f"Upper limit {lim[1]} must be greater than lower limit {lim[0]}.")
            except IndexError:
                lim = [lim[0], data_org.max()]

            data_org = np.where(data_org <= lim[1], data_org, lim[1])

            mask_overall = (data_org > 0)
            R_set = R_set_org[mask_overall]
            Z_set = Z_set_org[mask_overall]
            phi_set = phi_set_org[mask_overall]
            data = data_org[mask_overall]

            
            mask_name = ['overall', 'overall_back']
            angs = [(30,30), (30,210)]
            if args.plot_surface:
                R_grid = np.reshape(R_set_org, (iplane, -1), order='C')
                Z_grid = np.reshape(Z_set_org, (iplane, -1), order='C')
                phi_grid = np.reshape(phi_set_org, (iplane, -1), order='C')
                data_grid = np.reshape(data_org, (iplane, -1), order='C')
                # 将 mask 重塑为二维，与网格对齐
                mask_grid = np.reshape(mask_overall, (iplane, -1), order='C')
                for mask, ang in zip(mask_name, angs):
                    fig = plt.figure(figsize=(8, 6),dpi=300)
                    ax = fig.add_subplot(111, projection='3d')
                    cmap = plt.get_cmap('viridis')
                    plot_surface_from_scatter_dict(fig, ax, {'R': R_grid, 'Z': Z_grid, 'phi': phi_grid, 'val': data_grid}, log=args.log_norm, cmap=cmap, names=names, time_phys=time_phys, mask=mask_grid, iplane=iplane, fig_destiny=fig_destiny, angs=ang, find_max=args.find_max, mask_name=mask, DEBUG=DEBUG, test_flag=test_flag)
            else:
                for mask, ang in zip(mask_name, angs):
                    fig = plt.figure(figsize=(8, 6),dpi=300)
                    ax = fig.add_subplot(111, projection='3d')
                    cmap = plt.get_cmap('viridis')
                    plot_scatter_from_scatter_dict(fig, ax, {'R': R_set, 'Z': Z_set, 'phi': phi_set, 'val': data}, log=args.log_norm, cmap=cmap, names=names, time_phys=time_phys, fig_destiny=fig_destiny, angs=ang, find_max=args.find_max, mask_name=mask, DEBUG=DEBUG, test_flag=test_flag)

    else: # plot leg strike points
        for key in data_set.keys():
            time_phys = round(key * norm_factor * 1e3, 4) # ms

            R_set_org = data_set[key]['R']
            Z_set_org = data_set[key]['Z']
            phi_set_org = data_set[key]['phi']
            data_org = np.where(data_set[key][names[-1]] >= lim[0], data_set[key][names[-1]], lim[0])
            try:
                if lim[1] <= lim[0]:
                    raise ValueError(f"Upper limit {lim[1]} must be greater than lower limit {lim[0]}.")
            except IndexError:
                lim = [lim[0], data_org.max()]

            data_org = np.where(data_org <= lim[1], data_org, lim[1])

            masks, angs = mask_choose(R_set_org, Z_set_org, 'EXL50U')

            for mask_name, mask in masks.items():
                
                
                fig = plt.figure(figsize=(8, 6),dpi=300)
                ax = fig.add_subplot(111, projection='3d')
                
                if args.plot_surface:
                    
                    ang = angs[mask_name]

                    plot_surface_from_scatter_dict(fig, ax, {'R': R_set_org, 'Z': Z_set_org, 'phi': phi_set_org, 'val': data_org}, cmap='viridis', log=args.log_norm, names=names, time_phys=time_phys, mask=mask, iplane=iplane, fig_destiny=fig_destiny, angs=angs[mask_name], mask_name=mask_name, find_max=args.find_max, DEBUG=DEBUG, test_flag=test_flag)

                else:
                    ang = angs[mask_name]
                    R_set = R_set_org[mask]
                    Z_set = Z_set_org[mask]
                    phi_set = phi_set_org[mask]
                    data = data_org[mask]

                    plot_scatter_from_scatter_dict(fig, ax, {'R': R_set, 'Z': Z_set, 'phi': phi_set, 'val': data}, cmap='viridis', log=args.log_norm, names=names, time_phys=time_phys, fig_destiny=fig_destiny, angs=ang, mask_name=mask_name, find_max=args.find_max, DEBUG=DEBUG, test_flag=test_flag)


if __name__ == '__main__':
    main()
