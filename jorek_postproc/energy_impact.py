#!/usr/bin/env python3

"""
Energy impact calculation module.

Handles parallel processing of multiple time-step files for energy impact analysis.
"""

import os
import multiprocessing
import numpy as np
from typing import List, Dict, Any, Tuple, Callable, Optional
import gc
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.colors import LogNorm
from tqdm import tqdm


from . import read_boundary_file, reshape_to_grid, plot_surface_3d, plot_scatter_3d, PlottingConfig

# 常量定义
MAX_WORKERS = os.cpu_count() or 8
CHUNK_SIZE = 108
INTEGRATION_POINTS = 100
INTERP_POINTS = 2000

def _worker_wrapper(args: Tuple[str, Callable, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Internal wrapper to unpack arguments and catch exceptions in worker processes.
    """
    filename, func, kwargs = args
    try:
        return func(filename, **kwargs)
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return {'filename': filename, 'error': str(e), 'time': np.nan}

def compute_delta_q(t_raw, q_raw, t_eval, n_points=INTEGRATION_POINTS):
    """向量化计算delta_q (能量冲击卷积)"""
    if t_raw.size == 0 or q_raw.size == 0:
        return np.zeros_like(t_eval)
    
    # 创建插值函数
    interp_func = interp1d(
        t_raw, q_raw, kind='linear',
        bounds_error=False, fill_value=(q_raw[0], q_raw[-1]))
    
    # 预计算u_max数组
    u_max = np.sqrt(np.maximum(t_eval - t_raw[0], 0))
    valid_mask = u_max > 1e-10
    
    delta_q = np.zeros_like(t_eval)
    
    for i, valid in enumerate(valid_mask):
        if not valid:
            continue
        u_array = np.linspace(0, u_max[i], n_points)
        tau_array = t_eval[i] - u_array**2
        q_array = interp_func(tau_array)
        integral = np.trapz(q_array, u_array)
        delta_q[i] = 2 * integral
        
    return delta_q

def load_timestep_data(args):
    """单个时间步数据加载任务"""
    ts, base_dir, iplane, data_name, xpoints, debug = args
    
    # 尝试构建文件名，兼容 s04200 和 s004200 等格式
    candidates = [
        f"boundary_quantities_s0{str(ts)}.dat",
        f"boundary_quantities_s{str(ts).zfill(6)}.dat",
        f"boundary_quantities_s0{str(ts).zfill(4)}.dat"
    ]
    
    file_path = None
    for fname in candidates:
        p = os.path.join(base_dir, fname)
        if os.path.exists(p):
            file_path = p
            break
            
    if not file_path:
        if debug: print(f"  [Worker] 未找到时间步 {ts} 的文件")
        return ts, None

    try:
        # 使用包内的读取函数
        col_names, blocks, _ = read_boundary_file(file_path, debug=False)
        
        # 查找对应的数据块
        # 尝试匹配 ts 字符串
        block_key = None
        for key in blocks.keys():
            if str(ts) in key or key in str(ts):
                block_key = key
                break
        
        if not block_key:
            # 如果只有一个块，就用那个
            if len(blocks) == 1:
                block_key = list(blocks.keys())[0]
            else:
                return ts, None

        block_data = blocks[block_key]
        
        # 重整化
        names = ['R', 'Z', 'phi', data_name]
        if data_name not in col_names:
            return ts, None
            
        grid_data = reshape_to_grid(
            block_data, col_names, names,
            iplane=iplane, xpoints=xpoints, debug=False
        )
        
        return ts, grid_data # 返回完整对象以获取坐标信息
        
    except Exception as e:
        if debug: print(f"  [Worker] 处理时间步 {ts} 出错: {e}")
        return ts, None

def run_energy_impact_analysis(conf):
    """
    主控制流程
    替代了之前的 main_energy_impact 和旧的 run_energy_impact_analysis
    """
    base_dir = os.path.dirname(conf.file_path)
    # 初始输出目录，稍后会根据时间范围细化
    base_output_dir = conf.output_dir or os.path.join(base_dir, f"energy_impact_{conf.device}")
    
    print(f"[EnergyImpact] 并行加载 {len(conf.timesteps)} 个时间步的数据...\n 时间步：{conf.timesteps}")
    
    # 1. 并行加载数据
    data_set = {}
    tasks = []
    xpoints = None
    
    if conf.xpoints is not None:
        xpoints = np.array(conf.xpoints, dtype=float).reshape(-1, 2)
        xpoints.sort(axis=0)
    print(f"  ✓ 使用xpoints：\n{xpoints}" if xpoints is not None else "  ✓ 未使用xpoints")
    
    ref_grid_data = None # 用于存储网格信息(R, Z)
    tss = np.array(conf.timesteps,dtype=int)
    
    for ts in tss:
        tasks.append((ts, base_dir, conf.iplane, conf.data_name, xpoints, conf.debug))
        
        
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as executor:
        futures = [executor.submit(load_timestep_data, task) for task in tasks]
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Loading Data"):
            ts, g_data = future.result()
            if g_data is not None:
                # 保存第一个成功的grid_data作为参考（包含R, Z坐标）
                if ref_grid_data is None:
                    ref_grid_data = g_data
                
                # 假设时间单位转换，如果config里没有定义，默认使用 4.1006e-4
                time_factor = getattr(conf, 'time_factor', 4.1006e-4)
                key = ts*time_factor
                data_set[key] = g_data.data.astype(np.float32)
    
    if not data_set:
        print("❌ 未加载到任何数据，终止。")
        return
    print(data_set.keys(), len(data_set))
    # 2. 准备时间插值
    t_raw = np.sort(np.array(list(data_set.keys())))
    t_eval = np.linspace(t_raw.min(), t_raw.max(), INTERP_POINTS)
    
    # 更新输出目录，包含时间范围
    time_range_folder = f"{t_eval[0]:.5f}_{t_eval[-1]:.5f}"
    output_dir = os.path.join(base_output_dir, time_range_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    sample_shape = next(iter(data_set.values())).shape
    n_rows, n_cols = sample_shape
    print(f"[EnergyImpact] 网格大小: {n_rows}x{n_cols}, 插值点数: {INTERP_POINTS}")
    
    DQ = np.zeros((n_rows, n_cols, len(t_eval)), dtype=np.float32)
    
    # 3. 并行计算卷积
    print(f"[EnergyImpact] 开始计算卷积积分 (Workers: {MAX_WORKERS})...")
    calc_tasks = []
    for i in range(n_rows):
        for j in range(n_cols):
            q_raw = np.array([data_set[t][i, j] for t in t_raw])
            calc_tasks.append((i, j, t_raw, q_raw, t_eval))
            
    total_chunks = (len(calc_tasks) + CHUNK_SIZE - 1) // CHUNK_SIZE
    
    # 分块处理以节省内存
    chunk_iter = range(0, len(calc_tasks), CHUNK_SIZE)
    for chunk_idx in tqdm(chunk_iter, total=total_chunks, desc="Calculating Convolution"):
        chunk = calc_tasks[chunk_idx:chunk_idx + CHUNK_SIZE]
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(compute_delta_q, t, q, te): (i, j) 
                      for i, j, t, q, te in chunk}
            
            for future in as_completed(futures):
                i, j = futures[future]
                try:
                    dq = future.result()
                    DQ[i, j, :] = dq
                except Exception as e:
                    pass
        
        gc.collect()

    # 4. 保存结果 (可选)
    if getattr(conf, 'save_convolution', False):
        save_path = os.path.join(output_dir, 'convolution_result.npz')
        np.savez_compressed(save_path, DQ=DQ, t_eval=t_eval, t_raw=t_raw)
        print(f"[EnergyImpact] 原始数据已保存至 {save_path}")

    # 5. 生成最大值分布图
    print(f"[EnergyImpact] 生成3D图像...")
    # 计算每个点在所有时间上的最大值
    max_impact = np.max(DQ, axis=2)
    
    if ref_grid_data is not None:
        # 更新数据为最大能量冲击
        ref_grid_data.data = max_impact
        print(max_impact.min(), max_impact.max())
        # 配置绘图参数
        plotting_config = PlottingConfig(
            log_norm=conf.log_norm,
            cmap='inferno', # 能量图通常用热图
            dpi=300,
            data_limits=[max(np.nanmin(max_impact), 1e-5), np.nanmax(max_impact)],
            find_max=conf.find_max
        )
        
        try:
            fig = plt.figure(figsize=(12, 10), dpi=150)
            ax = fig.add_subplot(111, projection='3d')
            
            save_path = os.path.join(output_dir, 'max_energy_impact_3d.png')
            
            plot_surface_3d(
                ref_grid_data, fig, ax, 
                config=plotting_config,
                view_angle=(30, 45),
                save_path=save_path,
                debug=conf.debug
            )
            plt.close(fig)
            print(f"  ✓ 3D表面图已保存: {save_path}")
            
        except Exception as e:
            print(f"  ✗ 3D绘图失败: {e}")
            if conf.debug:
                import traceback
                traceback.print_exc()
    
    # 保留2D投影图作为参考
    plt.figure(figsize=(10, 8))
    plt.imshow(max_impact.T, origin='lower', cmap='inferno', aspect='auto')
    plt.colorbar(label='Max Energy Impact')
    plt.title(f'Maximum Energy Impact over {t_eval[0]:.4f}-{t_eval[-1]:.4f}')
    plt.savefig(os.path.join(output_dir, 'max_energy_impact_2d.png'), dpi=300)
    plt.close()
    
    print(f"✓ 处理完成，结果保存在: {output_dir}")

