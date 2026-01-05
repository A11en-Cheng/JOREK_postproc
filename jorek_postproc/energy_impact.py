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

# Numpy 2.0 compatibility
if hasattr(np, 'trapz'):
    trapz = np.trapz
else:
    trapz = np.trapezoid


from . import (
    read_boundary_file, 
    reshape_to_grid, 
    plot_surface_3d, 
    plot_scatter_3d, 
    PlottingConfig,
    get_device_geometry
)

# 常量定义
MAX_WORKERS = os.cpu_count() or 8
CHUNK_SIZE = 108
INTEGRATION_POINTS = 100
INTERP_POINTS = 2000
BATCH_SIZE = 5000 

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

def compute_delta_q_batch(t_raw, q_raw_batch, t_eval, n_points=INTEGRATION_POINTS):
    """
    向量化计算一批像素的delta_q
    q_raw_batch: (n_pixels, n_raw_time)
    Returns: (n_pixels, n_eval_time)
    """
    n_pixels = q_raw_batch.shape[0]
    n_eval = len(t_eval)
    
    if t_raw.size == 0 or q_raw_batch.size == 0:
        return np.zeros((n_pixels, n_eval), dtype=np.float32)
    
    # 为整批数据创建一个插值函数
    # axis=-1 表示沿着最后一个维度（时间）插值
    interp_func = interp1d(
        t_raw, q_raw_batch, kind='linear', axis=-1,
        bounds_error=False, fill_value=(q_raw_batch[:, 0], q_raw_batch[:, -1]),
        assume_sorted=True
    )
    
    delta_q_batch = np.zeros((n_pixels, n_eval), dtype=np.float32)
    
    # 预计算所有时刻的积分上限 u_max
    u_max_all = np.sqrt(np.maximum(t_eval - t_raw[0], 0))
    
    # 遍历评估时间点 (2000次循环)
    # 内部操作是对 n_pixels (如5000个) 的向量化操作，效率很高
    for k in range(n_eval):
        u_m = u_max_all[k]
        if u_m <= 1e-10:
            continue
            
        u_array = np.linspace(0, u_m, n_points) # (n_points,)
        tau_array = t_eval[k] - u_array**2      # (n_points,)
        
        # 插值: 结果形状 (n_pixels, n_points)
        q_vals = interp_func(tau_array)
        
        # 积分: 沿着 n_points 维度积分 -> (n_pixels,)
        integral = trapz(q_vals, u_array, axis=-1)
        
        delta_q_batch[:, k] = 2 * integral
        
    return delta_q_batch

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
    tss.sort()
    
    for ts in tss:
        tasks.append((ts, base_dir, conf.iplane, conf.data_name, xpoints, conf.debug))
        
    # 假设时间单位转换，如果config里没有定义，默认使用 4.1006e-4
    time_factor = getattr(conf, 'time_factor', 4.1006e-4)
        
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as executor:
        futures = [executor.submit(load_timestep_data, task) for task in tasks]
        
        # 使用tqdm显示进度
        for future in tqdm(as_completed(futures), total=len(tasks), desc="Loading Data"):
            ts, g_data = future.result()
            if g_data is not None:
                # 保存第一个成功的grid_data作为参考（包含R, Z坐标）
                if ref_grid_data is None:
                    ref_grid_data = g_data
                
                key = ts*time_factor
                data_set[key] = g_data.data.astype(np.float32)
    
    if not data_set:
        print("❌ 未加载到任何数据，终止。")
        return
    
    # 确保t=0时刻有数据 (参考energy_impact_new_copy.py)
    # 如果最小时间远大于0，插入0时刻的全0数据，保证积分从0开始
    min_t = min(data_set.keys())
    if min_t > 1e-6:
        print(f"  ⚠ 检测到起始时间 {min_t:.4e} > 0，自动补充 t=0 时刻的零数据以修正积分下限。")
        sample_shape = next(iter(data_set.values())).shape
        data_set[0.0] = np.zeros(sample_shape, dtype=np.float32)

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
    
    # 优化：构建3D数组 (Time, R, Z) -> (R*Z, Time) 以进行批处理
    n_timesteps = len(t_raw)
    
    # 1. 构建 (Time, R, Z)
    all_data_3d = np.zeros((n_timesteps, n_rows, n_cols), dtype=np.float32)
    for k, t in enumerate(t_raw):
        all_data_3d[k] = np.maximum(data_set[t], 0.0)
    
    # 2. 转换为 (N_pixels, Time)
    # transpose to (R, Z, Time) then reshape
    all_data_flat = all_data_3d.transpose(1, 2, 0).reshape(-1, n_timesteps)
    del all_data_3d # 释放内存
    
    n_total_pixels = all_data_flat.shape[0]
    
    # 3. 创建任务分块
    # 增加块大小以充分利用向量化优势，减少进程间通信开销
    # 假设每个像素2000个点，每个点4字节，10000个像素约80MB数据
    
    
    pixel_indices = np.arange(n_total_pixels)
    chunks = []
    for i in range(0, n_total_pixels, BATCH_SIZE):
        end_idx = min(i + BATCH_SIZE, n_total_pixels)
        # 提取该批次的数据
        batch_data = all_data_flat[i:end_idx, :]
        chunks.append((i, end_idx, t_raw, batch_data, t_eval))
            
    total_chunks = len(chunks)
    print(f"  总像素数: {n_total_pixels}, 分块数: {total_chunks}, 每块大小: {BATCH_SIZE}")
    
    # 4. 并行执行
    # DQ 形状 (R, Z, Time) -> Flattened (R*Z, Time)
    DQ_flat = np.zeros((n_total_pixels, len(t_eval)), dtype=np.float32)
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {
            executor.submit(compute_delta_q_batch, t_raw, batch_data, t_eval): (start, end)
            for start, end, t_raw, batch_data, t_eval in chunks
        }
        
        for future in tqdm(as_completed(futures), total=total_chunks, desc="Calculating Convolution"):
            start, end = futures[future]
            try:
                batch_result = future.result()
                DQ_flat[start:end, :] = batch_result
            except Exception as e:
                print(f"Chunk {start}-{end} failed: {e}")
                
    # 5. 重塑回 (R, Z, Time)
    DQ = DQ_flat.reshape(n_rows, n_cols, len(t_eval))
    del DQ_flat, all_data_flat # 释放内存
    gc.collect()

    # 4. 保存结果 (强制保存完整DQ，参考energy_impact_new_copy.py)
    # 即使config没开，为了诊断也建议保存，或者遵循config但默认True
    if getattr(conf, 'save_convolution', True):
        save_path = os.path.join(output_dir, 'DQ_full_data.npz')
        np.savez_compressed(save_path, DQ=DQ, t_eval=t_eval, t_raw=t_raw)
        print(f"[EnergyImpact] 完整DQ数据已保存至 {save_path}")

    # 5. 生成3D图像 (针对每个输入的时间步)
    print(f"[EnergyImpact] 生成3D图像 (共 {len(conf.timesteps)} 帧)...")
    
    if ref_grid_data is not None:
        # 获取装置几何
        print(f"[EnergyImpact] 获取装置位形 ({conf.device})...")
        try:
            device = get_device_geometry(conf.device, ref_grid_data.R, ref_grid_data.Z, xpoints=xpoints, debug=conf.debug)
            print(f"  ✓ 装置：{device.name}")
            print(f"  ✓ 位置：{list(device.masks.keys())}")
        except Exception as e:
            print(f"  ✗ 获取位形失败：{e}")
            return

        # 计算全局极值用于统一色标
        global_max = np.nanmax(DQ)
        global_min = np.nanmin(DQ)
        
        # 确定绘图范围
        if conf.data_limits:
            vmin, vmax = conf.data_limits
        else:
            vmin = global_min
            vmax = global_max
            # 对数坐标下避免0
            if conf.log_norm and vmin <= 0:
                vmin = max(1e-3, vmax * 1e-4)
        
        print(f"  绘图范围: [{vmin:.2e}, {vmax:.2e}]")

        plotting_config = PlottingConfig(
            log_norm=conf.log_norm,
            cmap='viridis', # 能量图通常用热图
            dpi=300,
            data_limits=[vmin, vmax],
            find_max=conf.find_max
        )
        
        # 对输入的时间步进行排序处理
        sorted_timesteps = sorted([float(ts) for ts in conf.timesteps])
        
        for ts_val in tqdm(sorted_timesteps, desc="Plotting 3D"):
            t_phys = ts_val * time_factor
            
            # 在t_eval中寻找最近的时间点索引
            idx = np.abs(t_eval - t_phys).argmin()
            
            # 获取对应时刻的DQ切片
            dq_slice = DQ[:, :, idx]
            
            # 更新数据对象
            ref_grid_data.data = dq_slice
            
            # 1. 绘制整体视图 (Front/Back)
            for view_name, angle in [('front', (30, 30)), ('back', (30, 210))]:
                fname = f'energy_impact_{t_phys:.5f}s_ts{int(ts_val)}_overall_{view_name}.png'
                save_path = os.path.join(output_dir, fname)
                
                try:
                    fig = plt.figure(figsize=(10, 8), dpi=150)
                    ax = fig.add_subplot(111, projection='3d')
                    
                    plot_surface_3d(
                        ref_grid_data, fig, ax, 
                        config=plotting_config,
                        view_angle=angle,
                        save_path=save_path,
                        debug=conf.debug
                    )
                    plt.close(fig)
                except Exception as e:
                    print(f"  ✗ 绘图失败 {fname}: {e}")

            # 2. 绘制部件视图 (Masked: UO, UI, LO, LI etc.)
            for mask_name, mask in device.masks.items():
                angle = device.view_angles.get(mask_name, (30, 45))
                fname = f'energy_impact_{t_phys:.5f}s_ts{int(ts_val)}_{mask_name}.png'
                save_path = os.path.join(output_dir, fname)
                
                try:
                    fig = plt.figure(figsize=(10, 8), dpi=150)
                    ax = fig.add_subplot(111, projection='3d')
                    
                    plot_surface_3d(
                        ref_grid_data, fig, ax, 
                        config=plotting_config,
                        mask=mask,
                        view_angle=angle,
                        save_path=save_path,
                        debug=conf.debug
                    )
                    plt.close(fig)
                except Exception as e:
                    print(f"  ✗ 绘图失败 {fname}: {e}")
    
    print(f"✓ 处理完成，结果保存在: {output_dir}")
