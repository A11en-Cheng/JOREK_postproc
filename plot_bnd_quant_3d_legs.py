#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
DEBUG = False
test_flag = False

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

def reshape_to_grid(block,col_names,names,iplane):
    '''
    normalize data to 3x3 matrix
    input names: [x_name, y_name, z_name, data_name] should be two column names and one data name
    output sorted_df: {x_name: x_grid, y_name: y_grid, data_name: data_grid}
    '''
    dataidx = names[-1] 
    if dataidx not in col_names:
        raise ValueError(f"Data index '{dataidx}' not found in column names.")
    df = {}
    
    for idx_col, name in enumerate(col_names):
        df[name] = block[:, idx_col]
    
    x = df[names[0]]  # R
    y = df[names[2]]  # phi
    z = df[names[1]]  # Z

    data = df[dataidx]

    dtype = [('x', float), ('y', float), ('z', float), ('data', float)]
    points = np.zeros(len(x), dtype=dtype)
    points['x'] = x
    points['y'] = y
    points['z'] = z
    points['data'] = data
    
    sorted_indices = np.lexsort((points['z'], points['y'], points['x']))
    sorted_points = points[sorted_indices]
    
    x_sorted = sorted_points['x']
    y_sorted = sorted_points['y']
    z_sorted = sorted_points['z']
    data_sorted = sorted_points['data']

    
    #lenth = 128*888  # iplane * 888
    lenth = 1080*1080
    row = int(iplane)
    col = int(lenth//iplane)
    print(row, col)
    
    x_grid = np.reshape(points['x'], (row, col))
    y_grid = np.reshape(points['y'], (row, col))
    z_grid = np.reshape(points['z'], (row, col))
    data_grid = np.reshape(data_sorted, (row, col))
    
    sorted_df = {
        names[0]: x_grid[0,:],
        names[1]: z_grid[0,:],
        names[2]: y_grid[:,0],
        dataidx : data_grid
    }
    sorted_df = {
        names[0]: x,
        names[1]: z,
        names[2]: y,
        dataidx : data
    }
    print('Sorted.')
    return sorted_df

def find_max(df,names,time):
    x_grid = df[names[0]]
    y_grid = df[names[1]]
    data = df[names[2]]
    
    max_val = np.max(data)
    if max_val == 0:
        print(f"Warning: Maximum value is zero at time {time}.")
        return None, None, max_val
    rows, cols = np.where(data == max_val)
    
    
    return rows,cols,max_val

def process_timestep(args,filename = None):
    """处理单个时间步的任务函数"""
    ts, file_addr, iplane, names = args
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
        
        grid_data = reshape_to_grid(block_data, col_names, names, iplane)
        return t_now, grid_data
    except Exception as e:
        print(f"Error processing timestep {ts}: {e}")
        return ts, None

def plot_scatter_from_scatter_dict(fig, ax, data, norm, names, time_phys, cmap='viridis', fig_destiny='figure_3d_scatter', angs=(30,30)):
    R = np.array(data['R'])
    Z = np.array(data['Z'])
    phi = np.array(data['phi'])
    val = np.array(data['val'])
    
    sc = ax.scatter(
        R, phi, Z,
        c=val, 
        cmap=cmap,
        s=1,          # Point size
        alpha=0.5
    )
    cbar = fig.colorbar(sc, pad=0.1)
    cbar.set_label('Value Intensity', rotation=270, labelpad=15)
    ax.set_xlabel('R Axis', fontsize=10)
    ax.set_ylabel('phi Axis', fontsize=10)
    ax.set_zlabel('Z Axis', fontsize=10)
    #ax.set_title(f'{names[-1]} at t={time_phys}ms')
    ax.view_init(*angs)  # 解包元组，确保传入两个数值
    if DEBUG or test_flag:
        plt.show()
    else:
        plt.savefig(os.path.join(fig_destiny, f'3d_plot_{time_phys}ms_{names[-1]}_overall.png'), dpi=300)
    plt.close(fig)

def plot_surface_from_scatter_dict(fig, ax, data, norm, names, time_phys, mask, iplane, cmap='viridis', fig_destiny='figure_3d_surface', angs=(30,30)):

    # 保持二维形状：在 mask 为 False 的位置填充 NaN（plot_surface 会忽略 NaN 区域）
    R_set_new = np.where(mask, data['R'], np.nan)
    Z_set_new = np.where(mask, data['Z'], np.nan)
    phi_set_new = np.where(mask, data['phi'], np.nan)
    data_new = np.where(mask, data['val'], np.nan)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(data_new)
    
    sc = ax.plot_surface(
        R_set_new, phi_set_new, Z_set_new,
        facecolors=cm.viridis(norm(data_new)),
        rcount=1080,
        ccount=1080,
        cmap=cmap,
        alpha=1,
        lw=0.,
    )
    cbar = plt.colorbar(sm, pad=0.1)
    cbar.set_label('Value Intensity', rotation=270, labelpad=15)
    ax.set_xlabel('R Axis', fontsize=10)
    ax.set_ylabel('phi Axis', fontsize=10)
    ax.set_zlabel('Z Axis', fontsize=10)
    #ax.set_title(f'{names[-1]} at t={time_phys}ms')
    ax.view_init(*angs)  # 解包元组，确保传入两个数值
    if DEBUG or test_flag:
        plt.show()
    else:
        #print(time_phys,names[-1],
              #type(time_phys),type(names[-1]))
        plt.savefig(os.path.join(fig_destiny, f'3d_plot_{str(time_phys)}ms_{names[-1]}_overall.png'), dpi=120)
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
    parser.add_argument("-s", "--plot_surface", action="store_true", default=True, help="Enable surface plotting from scatter data.")
    parser.add_argument("-st", "--strike_points", default=True, action="store_true", help="Enable strike points plotting.")
    parser.add_argument("--overall", action="store_true",default=False, help="Enable overall plotting.")
    parser.add_argument("--debug", action="store_true", default=False, help="Enable debug mode.")
    parser.add_argument("--test_flag", action="store_true", default=False, help="Enable test flag.")
    return parser.parse_args()

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
            'mask_UI': (30, 150),
            'mask_LI': (-12, -158),
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
    args = parse_arguments()
    if args.debug:
        global DEBUG
        DEBUG = True
    if args.test_flag:
        global test_flag
        test_flag = True
    if args.strike_points and args.overall:
        raise ValueError("Flags --strike_points and --overall cannot be enabled simultaneously.")
    if args.overall == True:
        args.strike_points = False
    if args.strike_points == False:
        args.overall = True
    
    file_addr = os.path.dirname(args.file)
    if args.plot_surface and args.strike_points:
        fig_destiny = os.path.join(os.getcwd(), f"figure_3d_surface_leg_{args.time}")
    elif args.overall and args.plot_surface:
        fig_destiny = os.path.join(os.getcwd(), f"figure_3d_surface_overall_{args.time}")
    elif args.overall and not args.plot_surface:
        fig_destiny = os.path.join(os.getcwd(), f"figure_3d_scatter_overall_{args.time}")
    elif args.strike_points and not args.plot_surface:
        fig_destiny = os.path.join(os.getcwd(), f"figure_3d_scatter_leg_{args.time}")
    
    ensure_directory_exists(fig_destiny)

    tss = [args.time]
    iplane = args.iplane
    names = ['R', 'Z', 'phi', args.name]
    norm_factor = 4.1006E-07  # jorekU to s conversion factor
    lim = [1e5]

    data_set = {}
    for ts in tss:
        ts = str(ts).zfill(6)
        t_now, q_data = process_timestep((ts, file_addr, iplane, names), filename=os.path.basename(args.file))
        t_phys = t_now[ts] * norm_factor
        data_set[t_now[ts]] = q_data
        if args.debug:
            print(f"Processed time {t_now[ts]:.2e} with shape {q_data['R'].shape}, {q_data['Z'].shape}, {q_data['phi'].shape}, t_phys={t_phys:.2e}")

    if args.overall:
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

            mask_overall = (R_set_org > 0)
            R_set = R_set_org[mask_overall]
            Z_set = Z_set_org[mask_overall]
            phi_set = phi_set_org[mask_overall]
            data = data_org[mask_overall]

            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            cmap = plt.get_cmap('viridis')
            norm = plt.Normalize(1e5, lim[1])
            if args.plot_surface:
                plot_surface_from_scatter_dict(fig, ax, {'R': R_set, 'Z': Z_set, 'phi': phi_set, 'val': data}, cmap=cmap, norm=norm, names=names, time_phys=time_phys, mask=mask_overall, iplane=iplane, fig_destiny=fig_destiny, angs=(30,30))
            else:
                plot_scatter_from_scatter_dict(fig, ax, {'R': R_set, 'Z': Z_set, 'phi': phi_set, 'val': data}, cmap=cmap, norm=norm, names=names, time_phys=time_phys, fig_destiny=fig_destiny, angs=(30,30))

    if args.strike_points:
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
                
                
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                cmap = plt.get_cmap('viridis')
                norm = plt.Normalize(1e5, lim[1])
                    
                if args.plot_surface:
                    R_grid = np.reshape(R_set_org, (iplane, -1), order='C')
                    Z_grid = np.reshape(Z_set_org, (iplane, -1), order='C')
                    phi_grid = np.reshape(phi_set_org, (iplane, -1), order='C')
                    data_grid = np.reshape(data_org, (iplane, -1), order='C')
                    # 将 mask 重塑为二维，与网格对齐
                    mask_grid = np.reshape(mask, (iplane, -1), order='C')
                    
                    ang = angs[mask_name]
                    
                    plot_surface_from_scatter_dict(fig, ax, {'R': R_grid, 'Z': Z_grid, 'phi': phi_grid, 'val': data_grid}, cmap='viridis', norm=norm, names=names, time_phys=time_phys, mask=mask_grid, iplane=iplane, fig_destiny=fig_destiny, angs=angs[mask_name])
                    
                else:
                    ang = angs[mask_name]
                    R_set = R_set_org[mask]
                    Z_set = Z_set_org[mask]
                    phi_set = phi_set_org[mask]
                    data = data_org[mask]

                    plot_scatter_from_scatter_dict(fig, ax, {'R': R_set, 'Z': Z_set, 'phi': phi_set, 'val': data}, cmap='viridis', norm=norm, names=names, time_phys=time_phys, fig_destiny=fig_destiny, angs=ang)


if __name__ == '__main__':
    main()
