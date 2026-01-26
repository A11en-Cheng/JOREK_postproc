#!/usr/bin/env python3

"""
Energy impact calculation module.

Handles parallel processing of multiple time-step files for energy impact analysis.
"""

from datetime import time
from types import SimpleNamespace
import os
import multiprocessing

from jorek_postproc import plotting
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
from .plotting import plot_heat_flux_analysis

# 常量定义
MAX_WORKERS = os.cpu_count() or 8
CHUNK_SIZE = 108
INTEGRATION_POINTS = 4000
INTERP_POINTS = 4000
BATCH_SIZE =108
test_dataset = False

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
    ts, primary_file, base_dir, data_name, xpoints, debug = args
    
    # 候选文件列表：优先尝试主文件，然后是标准命名格式文件
    candidates = []
    if primary_file and os.path.exists(primary_file):
        candidates.append(primary_file)

    # 尝试构建文件名，兼容 s04200 和 s004200 等格式
    standard_names = [
        f"boundary_quantities_s0{str(ts)}.dat",
        f"boundary_quantities_s{str(ts).zfill(6)}.dat",
        f"boundary_quantities_s0{str(ts).zfill(4)}.dat"
    ]
    for fname in standard_names:
        p = os.path.join(base_dir, fname)
        # 避免与主文件重复
        if p not in candidates and os.path.exists(p):
            candidates.append(p)

    time_now = np.nan
    
    # 遍历候选文件查找对应的 block
    for file_path in candidates:
        if debug: print(f"Checking file for step {ts}: {file_path}")
        
        try:
            # 使用包内的读取函数
            col_names, blocks, time_mapping = read_boundary_file(file_path, debug=False) # 减少详细输出
            
            # 查找对应的数据块
            # 尝试匹配 ts 字符串
            block_key = None
            for key in blocks.keys():
                # 精确匹配或子串匹配 (视 blocks keys 格式而定，通常包含步骤号)
                # 增强匹配逻辑：检查 key 中的数字部分是否匹配 ts
                if str(ts) in key:
                    block_key = key
                    time_now = time_mapping.get(key, np.nan)
                    break
            
            if not block_key:
                # 只有当文件只包含一个块且文件名就是为该块设计时，才回退到单块逻辑
                # 但这里我们在多个候选文件中搜索，如果主文件有多个块但不包含我们要的，不能直接取第一个
                # 如果只有一个块，且我们明确这是针对该ts的文件（文件名匹配），则可以使用
                if len(blocks) == 1 and str(ts) in os.path.basename(file_path):
                     block_key = list(blocks.keys())[0]
                     time_now = time_mapping.get(block_key, np.nan)

            if not block_key:
                # 在当前文件中未找到匹配的块，尝试下一个候选文件
                continue

            # 找到了 block
            if debug: print(f"  [Worker] Found step {ts} in {file_path}, block: {block_key}")
            block_data = blocks[block_key]
            
            # 过滤掉原本文件中整行为0的坏数据
            non_zero_mask = ~np.all(np.isclose(block_data, 0.0), axis=1)
            if np.sum(~non_zero_mask) > 0:
                if debug: print(f"  [Worker] 丢弃 {np.sum(~non_zero_mask)} 行全零数据")
                block_data = block_data[non_zero_mask]
            
            # 重整化
            names = ['R', 'Z', 'phi', data_name]
            if data_name not in col_names:
                print(f"  [Worker] 列 {data_name} 不存在于 {file_path}")
                return time_now, None

            grid_data = reshape_to_grid(
                block_data, col_names, names,
                xpoints=xpoints, debug=debug
            )
            
            return time_now, grid_data 
            
        except Exception as e:
            if debug: print(f"  [Worker] 读取/处理文件 {file_path} 出错: {e}")
            continue

    if debug: print(f"  [Worker] 未找到时间步 {ts} 的数据 (检查了 {len(candidates)} 个文件)")
    return time_now, None

def make_test_dataset():
    
    """
    创建测试用的数据集。
    """
    # 模拟读取数据  
    normalize_factor = 4.1006e-7
    names = ['R', 'Z', 'phi', 'heatF_tot_cd']
    data_set = {}
    data_set[0.e0] = np.zeros((300, 300), dtype=np.float32)  # 添加初始时间步
    data_set[normalize_factor] = np.zeros((300, 300), dtype=np.float32)
    #data_set[2.e1*normalize_factor] = np.zeros((300, 300), dtype=np.float32)
    for j in range(0,3):
        for i in range(0,3):
            data_set[normalize_factor][100*i:100*(i+1),100*j:100*(j+1)] = np.float32(10**(i+j+1))
    data_set[2.e0*normalize_factor] = np.zeros((300, 300), dtype=np.float32)
    data_set[2.e3*normalize_factor] = np.zeros((300, 300), dtype=np.float32)
    
    return data_set
    
    
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
        tasks.append((ts, conf.file_path, base_dir, conf.data_name, xpoints, conf.debug))
        
    # 假设时间单位转换，如果config里没有定义，默认使用 4.1006e-7
    time_factor = getattr(conf, 'time_factor', 4.1006e-7)
    
    if not test_dataset:
        
        with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as executor:
            futures = [executor.submit(load_timestep_data, task) for task in tasks]
            
            # 使用tqdm显示进度
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Loading Data"):
                time_now, g_data = future.result()
                if g_data is not None:
                    # 保存第一个成功的grid_data作为参考（包含R, Z坐标）
                    if ref_grid_data is None:
                        ref_grid_data = g_data
                    
                    key = time_now * time_factor
                    data_set[key] = g_data.data.astype(np.float32)
        
        if not data_set:
            print("❌ 未加载到任何数据，终止。")
            return
    else:
        print("[EnergyImpact] 使用测试数据集。")
        max_set = []
        data_set = make_test_dataset()
        # 创建参考网格数据
        sample_shape = next(iter(data_set.values())).shape
        
        # 映射到 R-Z-Phi 表面 (椭圆截面 torus)
        # 假设 axis 0 是 Phi (toroidal), axis 1 是 Theta (poloidal)
        n_phi, n_theta = sample_shape
        
        # 椭圆参数
        R0, Z0 = 3.0, 0.0
        a, b = 1.0, 1.5
        
        phi_1d = np.linspace(0, 2*np.pi, n_phi)
        theta_1d = np.linspace(0, 2*np.pi, n_theta)
        
        # indexing='ij' -> (n_phi, n_theta)
        Phi_grid, Theta_grid = np.meshgrid(phi_1d, theta_1d, indexing='ij')
        
        # 椭圆截面 R, Z (轴对称)
        R_grid = R0 + a * np.cos(Theta_grid)
        Z_grid = Z0 + b * np.sin(Theta_grid)
        
        ref_grid_data = SimpleNamespace(
            R=R_grid, Z=Z_grid, phi=Phi_grid, 
            data=next(iter(data_set.values())),
            is_2d_grid=lambda: True,
            data_name=conf.data_name
        )

    # 确保t=0时刻有数据 (参考energy_impact_new_copy.py)
    # 如果最小时间远大于0，插入0时刻的全0数据，保证积分从0开始
    min_t = min(data_set.keys())
    if min_t > 1e-9:
        print(f"  ⚠ 检测到起始时间 {min_t:.4e} > 0，自动补充 t=0 时刻的零数据以修正积分下限。")
        sample_shape = next(iter(data_set.values())).shape
        data_set[0.0] = np.zeros(sample_shape, dtype=np.float32)

    # 2. 准备时间插值
    t_raw = np.sort(np.array(list(data_set.keys())))
    t_eval = np.linspace(t_raw.min(), t_raw.max(), INTERP_POINTS)
    
    # 更新输出目录，包含时间范围
    time_range_folder = f"{t_raw[0]:.5f}_{t_raw[-1]:.5f}"
    output_dir = os.path.join(base_output_dir, time_range_folder)
    os.makedirs(output_dir, exist_ok=True)
    
    sample_shape = next(iter(data_set.values())).shape
    n_rows, n_cols = sample_shape
    print(f"[EnergyImpact] 网格大小: {n_rows}x{n_cols}, 插值点数: {INTERP_POINTS}")
    print(f"[EnergyImpact] time_raw: {t_raw}, time_eval: {t_eval.shape}")
    
    # DQ = np.zeros((n_rows, n_cols, len(t_eval)), dtype=np.float32) # Removed to save memory
    
    # 3. 并行计算卷积
    print(f"[EnergyImpact] 开始计算卷积积分 (Workers: {MAX_WORKERS})...")
    
    # 优化1：直接构建 (N_pixels, Time) 数组，避免 transpose/reshape 的内存复制
    n_timesteps = len(t_raw)
    n_total_pixels = n_rows * n_cols
    
    print(f"  构建输入数据矩阵 ({n_total_pixels} x {n_timesteps})...")
    all_data_flat = np.zeros((n_total_pixels, n_timesteps), dtype=np.float32)
    
    for k, t in enumerate(t_raw):
        # 直接展平并赋值，避免中间3D数组
        all_data_flat[:, k] = np.maximum(data_set[t], 0.0).ravel()
    
    # 释放原始字典以节省内存
    del data_set
    gc.collect()
    
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
    # 优化2：使用 memmap 存储巨大的结果矩阵，避免内存溢出
    # DQ 形状 (R, Z, Time) -> Flattened (R*Z, Time)
    
    dq_memmap_path = os.path.join(output_dir, 'dq_temp.dat')
    print(f"  使用磁盘映射文件存储结果: {dq_memmap_path}")
    
    DQ_flat = np.memmap(dq_memmap_path, dtype='float32', mode='w+', shape=(n_total_pixels, len(t_eval)))
    
    try:
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
                    # 定期flush
                    if (start // BATCH_SIZE) % 10 == 0:
                        DQ_flat.flush()
                except Exception as e:
                    print(f"Chunk {start}-{end} failed: {e}")
             
        # 5. 重塑回 (R, Z, Time)
        DQ = DQ_flat.reshape(n_rows, n_cols, len(t_eval))
        
        # 释放输入数据内存
        del all_data_flat
        del DQ_flat
        gc.collect()
        
    except Exception as e:
        print(f"❌ 卷积计算失败: {e}")
        return   
    # 4. 保存结果 (强制保存完整DQ，参考energy_impact_new_copy.py)
    # 即使config没开，为了诊断也建议保存，或者遵循config但默认True
    if getattr(conf, 'save_convolution', False):
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
            find_max=conf.find_max,
            show_left_plot=conf.show_left_plot,
            show_right_plot=conf.show_right_plot,
            use_arc_length=conf.use_arc_length
        )
        
        # 准备文件名后缀
        suffix = ""
        if conf.use_arc_length: suffix += "_arc"
        if conf.log_norm: suffix += "_log"
        
        # 对输入的时间步进行排序处理
        sorted_timesteps = sorted([float(ts) for ts in t_raw])
        if test_dataset:
            sorted_timesteps = sorted([float(ts) for ts in t_eval])
        
        for ts_val in tqdm(sorted_timesteps, desc="Plotting 3D"):
            t_phys = ts_val
            
            # 在t_eval中寻找最近的时间点索引
            idx = np.abs(t_eval - t_phys).argmin()
            print(idx)
            
            # 获取对应时刻的DQ切片
            dq_slice = DQ[:, :, idx]
            if test_dataset:
                max_set.append(dq_slice.max())
               
                
                continue
            # 更新数据对象
            ref_grid_data.data = dq_slice
            
            # 1. 绘制整体视图 (Front/Back)
            # 2D模式：总是绘制整体展开图 (包含所有区域标注)
            if conf.dim == '2d':
                 fname = f'energy_impact_2d_{t_phys*1.e3:.5f}ms_ts{int(idx)}_overall{suffix}.png'
                 save_path = os.path.join(output_dir, fname)
                 
                 # 准备所有区域标记
                 all_regions = []
                 if device.masks:
                     region_style = {
                        'mask_UI': {'label': 'IU', 'color': 'red'},
                        'mask_UO': {'label': 'OU', 'color': 'cyan'},
                        'mask_LI': {'label': 'IL', 'color': 'green'},
                        'mask_LO': {'label': 'OL', 'color': 'orange'}
                     }
                     for mname, mmask in device.masks.items():
                         if mname in region_style:
                             st = region_style[mname]
                             all_regions.append({'label': st['label'], 'mask': mmask, 'color': st['color']})
                 
                 try:
                    plot_heat_flux_analysis(
                        ref_grid_data, 
                        config=plotting_config,
                        save_path=save_path,
                        regions=all_regions,
                        debug=conf.debug
                    )
                 except Exception as e:
                    print(f"  ✗ 2D Overall绘图失败: {e}")
            
            # 3D模式：仅当启用 plot_overall 时绘制整体3D视图
            elif conf.dim != '2d' and conf.plot_overall:
                for view_name, angle in [('front', (30, 30)), ('back', (30, 210))]:
                    
                    if conf.plot_surface:
                        fname = f'energy_impact_surf_{t_phys*1.e3:.5f}ms_ts{int(idx)}_overall_{view_name}{suffix}.png'
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
                    else:
                        fname = f'energy_impact_scat_{t_phys*1.e3:.5f}ms_ts{int(idx)}_overall_{view_name}{suffix}.png'
                        save_path = os.path.join(output_dir, fname)
                        try:
                            plotting_config.cmap='inferno'  # 散点图用不同配色
                            fig = plt.figure(figsize=(10, 8), dpi=150)
                            ax = fig.add_subplot(111, projection='3d')
                        
                            plot_scatter_3d(
                                ref_grid_data, fig, ax, 
                                config=plotting_config,
                                view_angle=angle,
                                save_path=save_path,
                                debug=conf.debug
                            )
                            plt.close(fig)
                        except Exception as e:
                            print(f"  ✗ 散点图绘图失败 {fname}: {e}")

            # 2. 绘制部件视图 (Masked: UO, UI, LO, LI etc.)
            for mask_name, mask in device.masks.items():
                angle = device.view_angles.get(mask_name, (30, 45))
                
                if conf.dim == '2d':
                    # 2D 展开图 (部件高亮)
                    fname = f'energy_impact_2d_{t_phys*1.e3:.5f}ms_ts{int(idx)}_{mask_name}{suffix}.png'
                    save_path = os.path.join(output_dir, fname)
                    
                    current_regions = [{
                        'label': mask_name.replace('mask_', ''),
                        'mask': mask,
                        'color': 'red'
                    }]
                    
                    try:
                        plot_heat_flux_analysis(
                            ref_grid_data, 
                            config=plotting_config,
                            save_path=save_path,
                            regions=current_regions,
                            debug=conf.debug
                        )
                        if conf.debug: print(f"  ✓ 绘制部件2D展开图: {mask_name}")
                    except Exception as e:
                        print(f"  ✗ 2D绘图失败 {fname}: {e}")

                elif conf.plot_surface:
                    print(f"  ✓ 绘制部件视图: {mask_name}")
                    fname = f'energy_impact_surf_{t_phys*1.e3:.5f}ms_ts{int(idx)}_{mask_name}{suffix}.png'
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
                else:
                    print(f"  ✓ 绘制部件散点图视图: {mask_name}")
                    fname = f'energy_impact_scat_{t_phys*1.e3:.5f}ms_ts{int(idx)}_{mask_name}{suffix}.png'
                    save_path = os.path.join(output_dir, fname)
                    try:
                        plotting_config.cmap='inferno'  # 散点图用不同配色
                        fig = plt.figure(figsize=(10, 8), dpi=150)
                        ax = fig.add_subplot(111, projection='3d')
                        
                        plot_scatter_3d(
                            ref_grid_data, fig, ax, 
                            config=plotting_config,
                            mask=mask,
                            view_angle=angle,
                            save_path=save_path,
                            debug=conf.debug
                        )
                        plt.close(fig)
                    except Exception as e:
                        print(f"  ✗ 散点图绘图失败 {fname}: {e}")
    
    # 清理 memmap
    print("[EnergyImpact] 清理临时文件...")
    if test_dataset:

        np.savetxt('max_set.txt', np.array(max_set))
    try:
        # 确保删除引用以关闭文件句柄
        if 'DQ' in locals(): del DQ
        if 'DQ_flat' in locals(): del DQ_flat
        gc.collect()
        
        if os.path.exists(dq_memmap_path):
            os.remove(dq_memmap_path)
    except Exception as e:
        print(f"  ⚠ 清理临时文件失败: {e}")

    print(f"✓ 处理完成，结果保存在: {output_dir}")
    print(f"✓ 处理完成，结果保存在: {output_dir}")
