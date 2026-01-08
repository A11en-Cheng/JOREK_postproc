"""
命令行接口 - 允许直接从命令行使用包

用法示例：
  python -m jorek_postproc.cli -f boundary_quantities_s04200.dat -t 4200 \\
      --iplane 1080 -n heatF_tot_cd --log-norm
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import config as cfg
from . import energy_impact  # 新增导入
from . import (
    read_boundary_file,
    reshape_to_grid,
    get_device_geometry,
    plot_scatter_3d,
    plot_surface_3d,
    PlottingConfig,
)


def process_single_timestep(conf: cfg.ProcessingConfig):
    """
    处理单个时间步的完整流程。
    """
    print("\n" + "="*70)
    print(f"处理文件：{conf.file_path}")
    print(f"设备：{conf.device}")
    print(f"物理量：{conf.data_name}")
    print("="*70)
    
    # 检查文件存在
    file_dir = os.path.dirname(conf.file_path)
    if not os.path.exists(conf.file_path):
        raise FileNotFoundError(f"文件不存在：{conf.file_path}")
    
    # 1. 读取文件
    print("\n[1/4] 读取文件...")
    try:
        col_names, blocks, t_mapping = read_boundary_file(conf.file_path, debug=conf.debug)
    except Exception as e:
        print(f"  ✗ 读取失败：{e}")
        return
    
    print(f"  ✓ 成功读取，列数：{len(col_names)}，块数：{len(blocks)}")
    non_zero_mask = ~np.all(np.isclose(blocks, 0.0), axis=1)
    if np.sum(~non_zero_mask) > 0:
        if conf.debug: print(f"  [Worker] 丢弃 {np.sum(~non_zero_mask)} 行全零数据")
        blocks = blocks[non_zero_mask]
    # 2. 处理每个时间步
    for ts in conf.timesteps:
        ts_str = str(ts).zfill(6)
        print(f"\n[2/4] 处理时间步 {ts_str}...")
        
        # 获取数据块
        if ts_str not in blocks:
            print(f"  ✗ 时间步 {ts_str} 不在文件中")
            continue
        
        block_data = blocks[ts_str]
        print(f"  ✓ 数据块大小：{block_data.shape}")
        
        # 重整化数据
        print(f"\n[3/4] 重整化数据...")
        try:
            names = ['R', 'Z', 'phi', conf.data_name]
            
            # 检查列名是否存在
            if conf.data_name not in col_names:
                raise ValueError(f"列 '{conf.data_name}' 不在文件中。"
                               f"可用列：{col_names}")
            
            # 处理xpoints
            xpoints = None
            if conf.xpoints is not None:
                xpoints = np.array(conf.xpoints, dtype=float).reshape(-1, 2)
                xpoints.sort(axis=0)
            print(f"  ✓ 使用xpoints：\n{xpoints}" if xpoints is not None else "  ✓ 未使用xpoints")
            
            grid_data = reshape_to_grid(
                block_data, col_names, names,
                iplane=conf.iplane,
                xpoints=xpoints,
                debug=conf.debug
            )
            print(f"  ✓ 网格大小：{grid_data.grid_shape}")
            print(f"  ✓ 数据范围：[{grid_data.data.min():.2e}, {grid_data.data.max():.2e}]")
        except Exception as e:
            print(f"  ✗ 重整化失败：{e}")
            if conf.debug:
                import traceback
                traceback.print_exc()
            continue
        
        # 获取装置几何
        print(f"\n[4/4] 获取装置位形...")
        try:
            device = get_device_geometry(conf.device, grid_data.R, grid_data.Z, xpoints=xpoints, debug=conf.debug)
            print(f"  ✓ 装置：{device.name}")
            print(f"  ✓ 位置：{list(device.masks.keys())}")
        except Exception as e:
            print(f"  ✗ 获取位形失败：{e}")
            continue
        
        # 创建输出目录
        if conf.output_dir is None:
            output_dir = f"output_{conf.device}_{ts_str}"
        else:
            output_dir = conf.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"  ✓ 输出目录：{output_dir}")
        
        # 绘图配置
        plotting_config = PlottingConfig(
            log_norm=conf.log_norm,
            cmap='viridis',
            dpi=300,
            data_limits=conf.data_limits,
            find_max=conf.find_max
        )
        
        # 绘制图像
        print(f"\n绘制图像...")
        
        if conf.plot_overall:
            # 绘制整体视图
            print(f"  绘制整体视图...")
            for view_name, angle in [('front', (30, 30)), ('back', (30, 210))]:
                try:
                    fig = plt.figure(figsize=(10, 8), dpi=150)
                    ax = fig.add_subplot(111, projection='3d')
                    
                    save_path = os.path.join(output_dir, 
                                           f'overall_{view_name}_{ts_str}.png')
                    
                    if conf.plot_surface:
                        plot_surface_3d(grid_data, fig, ax, config=plotting_config,
                                      view_angle=angle, save_path=save_path, debug=conf.debug)
                    else:
                        plot_scatter_3d(grid_data, fig, ax, config=plotting_config,
                                      view_angle=angle, save_path=save_path, debug=conf.debug)
                    
                    print(f"    ✓ {view_name}视图已保存")
                except Exception as e:
                    print(f"    ✗ {view_name}视图失败：{e}")
        else:
            # 绘制位置特定的视图
            print(f"  绘制位置特定视图...")
            for mask_name, mask in device.masks.items():
                angle = device.view_angles[mask_name]
                try:
                    fig = plt.figure(figsize=(10, 8), dpi=150)
                    ax = fig.add_subplot(111, projection='3d')
                    
                    save_path = os.path.join(output_dir,
                                           f'{mask_name}_{ts_str}.png')
                    
                    if conf.plot_surface:
                        plot_surface_3d(grid_data, fig, ax, config=plotting_config,
                                      mask=mask, view_angle=angle, save_path=save_path,
                                      debug=conf.debug)
                    else:
                        plot_scatter_3d(grid_data, fig, ax, config=plotting_config,
                                      mask=mask, view_angle=angle, save_path=save_path,
                                      debug=conf.debug)
                    
                    print(f"    ✓ {mask_name} 已保存")
                except Exception as e:
                    print(f"    ✗ {mask_name} 失败：{e}")
                    if conf.debug:
                        import traceback
                        traceback.print_exc()
        
        print(f"\n✓ 时间步 {ts_str} 处理完成")
    
    print("\n" + "="*70)
    print("✓ 处理完成！")
    print("="*70)

def process_energy_impact(conf: cfg.ProcessingConfig):
    """
    处理能量冲击计算的完整流程。
    """
    print("\n" + "="*70)
    print(f"启动能量冲击分析 (Energy Impact Analysis)")
    print(f"基准文件路径：{conf.file_path}")
    print(f"包含时间步数：{len(conf.timesteps)}")
    print(f"物理量：{conf.data_name}")
    print("="*70)

    try:
        # 调用新模块的处理函数
        energy_impact.run_energy_impact_analysis(conf)
    except Exception as e:
        print(f"\n✗ 能量冲击计算失败：{e}")
        
        import traceback
        traceback.print_exc()


def main():
    """
    主函数
    """
    try:
        INTERACTIVE_DEBUG = False 
        
        if INTERACTIVE_DEBUG:
            print("[DEBUG] 使用调试配置")
            conf = cfg.create_debug_config()
        else:
        # 解析命令行参数
            conf = cfg.parse_args()
            
            if conf.debug:
                print(f"[DEBUG] 配置信息：")
                print(f"  文件：{conf.file_path}")
                print(f"  时间步：{conf.timesteps}")
                print(f"  iplane：{conf.iplane}")
                print(f"  数据：{conf.data_name}")
                print(f"  设备：{conf.device}")
                print(f"  绘图模式：{'表面图' if conf.plot_surface else '散点图'}")
        
        # 处理数据
        if not conf.energy_impact:
            process_single_timestep(conf)
        else:
            process_energy_impact(conf)
        
    except KeyboardInterrupt:
        print("\n\n用户中断处理")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ 错误：{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
