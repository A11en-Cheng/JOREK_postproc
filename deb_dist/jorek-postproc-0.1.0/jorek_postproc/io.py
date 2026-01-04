"""
IO module for reading boundary quantities files from JOREK output.
"""

import os
import numpy as np
from typing import Tuple, Dict, List


def read_boundary_file(file_path: str, debug: bool = False) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, float]]:
    """
    读取JOREK边界量文件。
    
    Parameters
    ----------
    file_path : str
        文件路径
    debug : bool, optional
        调试模式，默认False。若为True，打印详细信息
        
    Returns
    -------
    col_names : list of str
        列名列表
    blocks : dict
        数据块字典，键为时间步标识 (e.g., '004200')，值为对应的numpy数组
    t_mapping : dict
        时间步标识到物理时间的映射字典
        
    Examples
    --------
    >>> col_names, blocks, t_mapping = read_boundary_file('boundary_quantities_s04200.dat')
    >>> print(col_names)
    ['R', 'Z', 'phi', 'theta', 'heatF_tot_cd', ...]
    >>> print(blocks['004200'].shape)
    (1224000,)
    """
    blocks = {}
    current_block = None
    data_rows = []
    col_names = []
    t_now = np.nan
    t_mapping = {}
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('# '):
                if line.startswith('# time step'):
                    # 提取时间信息
                    t_now = float(line.split()[-1])
                    step_now = line.split()[3][1:].strip(',')
                    t_mapping[step_now] = t_now
                    
                    # 保存前一个数据块
                    if current_block and data_rows:
                        blocks[current_block] = np.array(data_rows, dtype=np.float32)
                        data_rows = []
                    
                    # 提取新的块名称
                    block_name = line.split()[3][1:-1]
                    current_block = block_name
                else:
                    # 列名信息
                    col_names = line.split()[1:]
            else:
                # 数据行
                data_rows.append(list(map(float, line.split())))

        # 保存最后一个数据块
        if current_block and data_rows:
            blocks[current_block] = np.array(data_rows, dtype=np.float32)

    if debug:
        print(f"[IO] Columns: {col_names}")
        print(f"[IO] Blocks: {list(blocks.keys())}")
        print(f"[IO] Time mapping: {t_mapping}")
    
    return col_names, blocks, t_mapping
