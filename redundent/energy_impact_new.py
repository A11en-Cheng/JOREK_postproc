#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import time
import gc

# 常量定义
MAX_WORKERS = os.cpu_count() or 8
CHUNK_SIZE = 108  # 任务分块大小
INTEGRATION_POINTS = 50  # 积分点数
INTERP_POINTS = 500  # 插值点数

def read_boundary_file(file_path, DEBUG=0):
    """优化后的文件读取函数，使用numpy直接加载数据"""
    blocks = {}
    current_block = None
    data_rows = []
    col_names = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('# '):
                if line.startswith('# time step'):
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
    
    return col_names, blocks

def reshape_to_grid(data, col_names, names, iplane):
    """向量化数据网格化处理"""
    x_idx = col_names.index(names[0])
    y_idx = col_names.index(names[1])
    data_idx = col_names.index(names[2])
    
    x = data[:, x_idx]
    y = data[:, y_idx]
    y = np.arange(0,2*np.pi,2*np.pi/1080)
    values = data[:, data_idx]
    
    # 获取唯一坐标点
    unique_x = np.unique(x)
    unique_y = np.unique(y)
    
    # 创建网格
    x_grid, y_grid = np.meshgrid(unique_x, unique_y, indexing='ij')
    
    # 创建值网格
    value_grid = np.full_like(x_grid, np.nan, dtype=np.float32)
    
    # 向量化映射
    coord_map = {}
    for i, (xi, yi) in enumerate(zip(x, y)):
        coord_map[(xi, yi)] = values[i]
    
    for i, xi in enumerate(unique_x):
        for j, yj in enumerate(unique_y):
            value_grid[i, j] = coord_map.get((xi, yj), np.nan)
    
    return {
        names[0]: x_grid,
        names[1]: y_grid,
        names[2]: value_grid
    }

def compute_delta_q(t_raw, q_raw, t_eval, n_points=INTEGRATION_POINTS):
    """向量化计算delta_q"""
    if t_raw.size == 0 or q_raw.size == 0:
        return np.zeros_like(t_eval)
    
    # 创建插值函数（使用线性插值外推）
    interp_func = interp1d(
        t_raw, q_raw, kind='linear',
        bounds_error=False, fill_value=(q_raw[0], q_raw[-1]))
    
    # 预计算u_max数组
    u_max = np.sqrt(np.maximum(t_eval - t_raw[0], 0))
    valid_mask = u_max > 1e-10
    
    # 初始化结果数组
    delta_q = np.zeros_like(t_eval)
    
    # 处理有效点
    for i, valid in enumerate(valid_mask):
        if not valid:
            continue
            
        # 生成积分点
        u_array = np.linspace(0, u_max[i], n_points)
        tau_array = t_eval[i] - u_array**2
        
        # 向量化插值
        q_array = interp_func(tau_array)
        
        # 使用梯形法则积分（比辛普森更快）
        integral = np.trapz(q_array, u_array)
        delta_q[i] = 2 * integral  # 2倍因子
        
    return delta_q

def process_timestep(args):
    """处理单个时间步的任务函数"""
    ts, file_addr, iplane, names = args
    file_name = f'boundary_quantities_s0{ts}.dat'
    file_path = os.path.join(file_addr, file_name)
    
    try:
        col_names, blocks = read_boundary_file(file_path)
        block_data = blocks.get(f'00{ts}')
        if block_data is None:
            return ts, None
        
        grid_data = reshape_to_grid(block_data, col_names, names, iplane)
        return ts, grid_data[names[2]]
    except Exception as e:
        print(f"Error processing timestep {ts}: {e}")
        return ts, None

def main():
    # 配置参数
    file_addr = '/home/ac_desktop/syncfiles/postproc_145'
    fig_destiny = '/home/ac_desktop/syncfiles/bdy_flux/145'
    tss = [2200, 3600, 4200, 4650, 5720]
    iplane = 1080
    names = ['phi', 'theta', 'heatF_tot_cd']
    
    # 步骤1: 并行读取所有时间步数据
    data_set = {}
    tasks = [(ts, file_addr, iplane, names) for ts in tss]
    
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tss))) as executor:
        futures = [executor.submit(process_timestep, task) for task in tasks]
        for future in as_completed(futures):
            ts, q_data = future.result()
            if q_data is not None:
                t_phys = ts * 4.1006e-4
                data_set[t_phys] = q_data.astype(np.float32)
                print(f"Processed timestep {ts} with shape {q_data.shape}")
    
    # 步骤2: 时间插值
    t_raw = np.sort(np.array(list(data_set.keys())))
    t_eval = np.linspace(t_raw.min(), t_raw.max(), INTERP_POINTS)
    
    # 预分配结果数组
    n_rows, n_cols = 1080, 1080
    DQ = np.zeros((n_rows, n_cols, len(t_eval)), dtype=np.float32)
    
    # 步骤3: 并行计算delta_q
    tasks = []
    for i in range(n_rows):
        for j in range(n_cols):
            q_raw = np.array([data_set[t][i, j] for t in t_raw])
            tasks.append((i, j, t_raw, q_raw, t_eval))
    
    # 分块处理减少内存压力
    for chunk_idx in range(0, len(tasks), CHUNK_SIZE):
        chunk = tasks[chunk_idx:chunk_idx + CHUNK_SIZE]
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(compute_delta_q, t, q, te): (i, j) 
                      for i, j, t, q, te in chunk}
            
            for future in as_completed(futures):
                i, j = futures[future]
                try:
                    dq = future.result()
                    DQ[i, j, :] = dq
                except Exception as e:
                    print(f"Error at ({i},{j}): {e}")
        
        # 手动清理内存
        del chunk, futures
        gc.collect()
        print(f"Processed chunk {chunk_idx//CHUNK_SIZE + 1}/{(len(tasks)+CHUNK_SIZE-1)//CHUNK_SIZE}")
    
    # 步骤4: 结果可视化
    try:
        while True:
            t_input = float(input("Enter time value: "))* 4.1006e-4
            t_idx = np.abs(t_eval - t_input).argmin()
            
            plt.figure(figsize=(10, 8))
            plt.imshow(DQ[:, :, t_idx].T, 
                      origin='lower',
                      norm=LogNorm(vmin=1e-3, vmax=np.nanmax(DQ[:, :, t_idx])),
                      cmap='viridis')
            
            plt.colorbar(label='Energy Impact (J/m²)')
            plt.title(f'Energy Impact at t = {t_eval[t_idx]:.2f} ms')
            plt.xlabel('Phi Index')
            plt.ylabel('Theta Index')
            
            save_path = os.path.join(fig_destiny, f'energy_impact_{t_input:.2f}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved visualization to {save_path}")
            
    except KeyboardInterrupt:
        print("Processing completed")

if __name__ == '__main__':
    main()