"""
示例脚本 - 演示如何使用jorek_postproc包

此脚本展示了包的基本使用方法，包括：
1. 读取边界量文件
2. 重整化数据为网格
3. 应用位形掩膜
4. 绘制3D图像
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from jorek_postproc import (
    read_boundary_file,
    reshape_to_grid,
    get_device_geometry,
    plot_scatter_3d,
    plot_surface_3d,
    PlottingConfig,
)


def example_basic_usage():
    """
    示例1：基本使用流程
    """
    print("\n=== 示例1：基本使用流程 ===")
    
    # 配置参数
    file_path = '/home/ac_desktop/syncfiles/postproc_145/boundary_quantities_s04200.dat'
    iplane = 1080
    names = ['R', 'Z', 'phi', 'heatF_tot_cd']
    
    # 1. 读取文件
    print("[Step 1] 读取边界量文件...")
    col_names, blocks, t_mapping = read_boundary_file(file_path, debug=True)
    
    # 2. 重整化数据
    print("\n[Step 2] 重整化数据...")
    block_data = blocks['004200']
    grid_data = reshape_to_grid(block_data, col_names, names, iplane=iplane, debug=True)
    
    print(f"Grid shape: {grid_data.grid_shape}")
    print(f"Data range: [{grid_data.data.min():.2e}, {grid_data.data.max():.2e}]")
    
    return grid_data, col_names


def example_with_device_geometry(grid_data):
    """
    示例2：使用装置位形掩膜
    """
    print("\n=== 示例2：装置位形掩膜 ===")
    
    # 获取设备几何信息
    device = get_device_geometry('EXL50U', grid_data.R, grid_data.Z, debug=True)
    
    print(f"Device: {device.name}")
    print(f"Available masks: {list(device.masks.keys())}")
    
    # 绘制单个位置
    mask_name = 'mask_UO'  # Upper Outer
    mask = device.masks[mask_name]
    angle = device.view_angles[mask_name]
    
    print(f"\nPlotting {mask_name} with viewing angle {angle}")
    
    return device, mask, angle


def example_scatter_plot(grid_data, mask, angle, output_dir=None):
    """
    示例3：绘制散点图
    """
    print("\n=== 示例3：3D散点图 ===")
    
    config = PlottingConfig(
        log_norm=True,
        cmap='viridis',
        dpi=150,
        data_limits=[1e5, 3e8],
        find_max=True
    )
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置保存路径
    if output_dir is not None:
        save_path = os.path.join(output_dir, 'scatter_plot.png')
    else:
        save_path = None
    
    plot_scatter_3d(
        grid_data, fig, ax,
        config=config,
        mask=mask,
        view_angle=angle,
        save_path=save_path,
        debug=True
    )
    
    print("散点图已保存" if save_path else "散点图已显示")


def example_surface_plot(grid_data, mask, angle, output_dir=None):
    """
    示例4：绘制表面图
    """
    print("\n=== 示例4：3D表面图 ===")
    
    config = PlottingConfig(
        log_norm=True,
        cmap='viridis',
        dpi=150,
        data_limits=[1e5, 3e8],
        find_max=True
    )
    
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置保存路径
    if output_dir is not None:
        save_path = os.path.join(output_dir, 'surface_plot.png')
    else:
        save_path = None
    
    plot_surface_3d(
        grid_data, fig, ax,
        config=config,
        mask=mask,
        view_angle=angle,
        save_path=save_path,
        debug=True
    )
    
    print("表面图已保存" if save_path else "表面图已显示")


def example_multiple_views(grid_data, device, output_dir=None):
    """
    示例5：绘制多个位置的视图
    """
    print("\n=== 示例5：多位置视图 ===")
    
    config = PlottingConfig(
        log_norm=True,
        cmap='viridis',
        dpi=150,
        data_limits=[1e5, 3e8],
        find_max=False
    )
    
    # 对每个位置进行绘图
    for mask_name, mask in device.masks.items():
        angle = device.view_angles[mask_name]
        
        print(f"  绘制 {mask_name}...")
        
        fig = plt.figure(figsize=(10, 8), dpi=150)
        ax = fig.add_subplot(111, projection='3d')
        
        if output_dir is not None:
            save_path = os.path.join(output_dir, f'surface_{mask_name}.png')
        else:
            save_path = None
        
        plot_surface_3d(
            grid_data, fig, ax,
            config=config,
            mask=mask,
            view_angle=angle,
            save_path=save_path,
            debug=False
        )
    
    print(f"已生成 {len(device.masks)} 个位置的视图")


def main():
    """
    主函数 - 运行所有示例
    """
    print("="*60)
    print("JOREK后处理包 - 使用示例")
    print("="*60)
    
    # 创建输出目录
    output_dir = './example_output'
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 示例1：基本使用
        grid_data, col_names = example_basic_usage()
        
        # 示例2：装置位形
        device, mask, angle = example_with_device_geometry(grid_data)
        
        # 示例3和4：单个位置的散点和表面图
        # example_scatter_plot(grid_data, mask, angle, output_dir)
        # example_surface_plot(grid_data, mask, angle, output_dir)
        
        # 示例5：多位置视图
        example_multiple_views(grid_data, device, output_dir)
        
        print("\n" + "="*60)
        print(f"示例运行完成！输出已保存到：{output_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"\n错误：{e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
