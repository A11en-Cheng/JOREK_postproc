"""
Processing module for timestep handling and data pipeline.
"""

import os
import numpy as np
from typing import Tuple, Optional, Dict
from .data_models import BoundaryQuantitiesData
from .io import read_boundary_file
from .reshaping import reshape_to_grid


def process_timestep(
    timestep: str,
    file_path: str,
    column_names: list,
    names: Tuple[str, str, str, str],
    xpoints: Optional[np.ndarray] = None,
    debug: bool = False
) -> Tuple[Optional[float], Optional[BoundaryQuantitiesData]]:
    """
    处理单个时间步的完整流程。

    包括文件读取、数据提取、重整化等步骤。

    Parameters
    ----------
    timestep : str
        时间步标识 (e.g., '004200')
    file_path : str
        边界量文件路径
    column_names : list
        列名列表
    names : tuple of 4 str
        物理量列映射 [R_name, Z_name, phi_name, val_name]
    xpoints : numpy.ndarray, optional
        X点坐标
    debug : bool, optional
        调试模式标志

    Returns
    -------
    time : float or None
        对应的物理时间，如果处理失败返回None
    grid_data : BoundaryQuantitiesData or None
        重整化后的网格数据，如果处理失败返回None
    """
    try:
        # 从二维数据块中提取
        block_data = None
        for col_idx, col_name in enumerate(column_names):
            if 'data' in col_name.lower():  # 简单启发式方法
                break
        
        if block_data is None or block_data.shape[0] == 0:
            if debug:
                print(f"[Processing] No data found for timestep {timestep}")
            return None, None
        
        if debug:
            print(f"[Processing] Processing timestep {timestep} with {len(block_data)} points.")
        
        # 重整化数据
        grid_data = reshape_to_grid(
            block_data,
            column_names,
            names,
            xpoints=xpoints,
            debug=debug
        )
        
        # 设置时间步标识
        grid_data.time_step = timestep
        
        return None, grid_data
        
    except Exception as e:
        if debug:
            print(f"[Processing] Error processing timestep {timestep}: {e}")
        return None, None


def process_multiple_timesteps(
    timesteps: list,
    file_addr: str,
    column_names: list,
    names: Tuple[str, str, str, str],
    xpoints: Optional[np.ndarray] = None,
    debug: bool = False,
    filename: Optional[str] = None
) -> Dict[str, BoundaryQuantitiesData]:
    """
    批量处理多个时间步。

    Parameters
    ----------
    timesteps : list of str
        时间步标识列表
    file_addr : str
        文件所在目录
    column_names : list
        列名列表
    names : tuple of 4 str
        物理量列映射
    iplane : int, optional
        环向平面数
    xpoints : numpy.ndarray, optional
        X点坐标
    debug : bool, optional
        调试模式标志
    filename : str, optional
        具体文件名，如果为None则根据timestep自动生成

    Returns
    -------
    data_dict : dict
        timestep标识映射到处理好的数据的字典
    """
    data_dict = {}
    
    for ts in timesteps:
        ts_str = str(ts).zfill(6)
        
        # 构造文件路径
        if filename is None:
            file_name = f'boundary_quantities_s0{ts_str}.dat'
        else:
            file_name = filename
        
        file_path = os.path.join(file_addr, file_name)
        
        if not os.path.exists(file_path):
            if debug:
                print(f"[Processing] File not found: {file_path}")
            continue
        
        try:
            # 读取文件
            col_names, blocks, t_mapping = read_boundary_file(file_path, debug=debug)
            
            # 获取对应的数据块
            block_data = blocks.get(ts_str)
            if block_data is None:
                if debug:
                    print(f"[Processing] Block {ts_str} not found in file")
                continue
            
            # 重整化数据
            grid_data = reshape_to_grid(
                block_data,
                col_names,
                names,
                xpoints=xpoints,
                debug=debug
            )
            
            # 设置时间信息
            grid_data.time_step = ts_str
            if ts_str in t_mapping:
                grid_data.time = t_mapping[ts_str]
            
            data_dict[ts_str] = grid_data
            
            if debug:
                print(f"[Processing] Successfully processed {ts_str}")
        
        except Exception as e:
            if debug:
                print(f"[Processing] Error processing timestep {ts_str}: {e}")
            continue
    
    return data_dict


def apply_data_limits(
    data: BoundaryQuantitiesData,
    limits: Optional[list] = None,
    debug: bool = False
) -> BoundaryQuantitiesData:
    """
    对数据应用上下限约束。

    Parameters
    ----------
    data : BoundaryQuantitiesData
        输入数据
    limits : list of 2 float, optional
        [min, max] 数据限制
    debug : bool, optional
        调试模式标志

    Returns
    -------
    data_clipped : BoundaryQuantitiesData
        应用限制后的数据副本
    """
    if limits is None or len(limits) != 2:
        return data
    
    min_val, max_val = limits[0], limits[1]
    
    # 创建新的数据副本
    data_clipped = BoundaryQuantitiesData(
        R=data.R.copy(),
        Z=data.Z.copy(),
        phi=data.phi.copy(),
        data=np.clip(data.data.copy(), min_val, max_val),
        data_name=data.data_name,
        time=data.time,
        time_step=data.time_step,
        grid_shape=data.grid_shape
    )
    
    if debug:
        print(f"[Processing] Applied data limits: [{min_val}, {max_val}]")
    
    return data_clipped
