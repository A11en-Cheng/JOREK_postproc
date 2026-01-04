#!/usr/bin/env python3

#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def file_read(file_name,file_addr,DEBUG=0):
    '''
    input file_name, file_adderss, debug option
    output colum names, blocks dictionary blocks:{tstep:datablock}
    '''
    file = file_addr + '/' + file_name
    
    blocks = {}
    current_block = None
    data_rows = []

    with open(file, 'r') as f:
        #read header
        
        for line in f:
            
            line = line.strip()
            if not line:
                continue
        
            if line.startswith('# '):
                #blocks
                if line.startswith('# time step'):
                    if current_block is not None and data_rows:
                        blocks[current_block] = np.array(data_rows, dtype=float)
                        print(blocks[current_block].shape())
                        data_rows = []
                        
                    block_name = list(filter(None, line.split(' ')))[3][1:-1]  # Extract the block name
                    current_block = block_name  # Set the current block name
                #header
                else:
                    header_line1 = list(filter(None, line.split(' '))) # R  Z  phi  theta  heatF_tot_cd   heatF_tot_cv  heatF_prp_cd  heatF_par_cd ...
                    col_names = header_line1[1:]
                
            else:
                data_rows.append(list(map(float, line.split())))
            
            #last block
        if current_block is not None and data_rows:
            blocks[current_block] = np.array(data_rows, dtype=float)
    if DEBUG == 1:
        print(col_names,len(blocks))

    return col_names, blocks #colum names, blocks:{tstep:datablock}
    
def data_norm(col_names,block,iplane,names):
    '''
    normalize data to 2x2 matrix
    input names: [x_name, y_name, data_name] should be two column names and one data name
    output sorted_df: {x_name: x_grid, y_name: y_grid, data_name: data_grid}
    '''
    dataidx = names[2]  # Assuming the third name is the data to be normalized
    if dataidx not in col_names:
        raise ValueError(f"Data index '{dataidx}' not found in column names.")
    df = {}
    
    for idx_col, name in enumerate(col_names):
        df[name] = block[:, idx_col]
    
    x = df[names[0]]  
    y = df[names[1]]  

    data = df[dataidx]
    
    dtype = [('x', float), ('y', float), ('data', float)]
    points = np.zeros(len(x), dtype=dtype)
    points['x'] = x
    points['y'] = y
    points['data'] = data
    
    sorted_indices = np.lexsort((points['y'], points['x']))
    sorted_points = points[sorted_indices]
    
    x_sorted = sorted_points['x']
    y_sorted = sorted_points['y']
    data_sorted = sorted_points['data']

    
    lenth = df['R'].shape[0]
    row = int(iplane)
    col = int(lenth//iplane)
    print(row,col)
    
    unique_x = np.unique(x_sorted)
    unique_y = np.unique(y_sorted)
    nx = len(unique_x)
    ny = len(unique_y)
    
    x_grid = np.reshape(x_sorted, (col, row))
    y_grid = np.reshape(y_sorted, (col, row))
    data_grid = np.reshape(data_sorted, (col, row))
    
    sorted_df = {
        names[0]: x_grid,
        names[1]: y_grid,
        dataidx : data_grid
    }
    print('sorted')
    return sorted_df

def plot_RZcontour(df, names):
    """
    使用排序后的数据绘图
    """
    # 直接从排序后的网格获取数据
    x_grid = df[names[0]]
    y_grid = df[names[1]]
    data = df[names[2]]
    
    # 应用数据阈值
    #data_max = np.max(np.abs(data))
    #if data_max > 0:
    #    data_normalized = data / data_max
    #    # 设置绝对值小于阈值的区域为0
    #    threshold = 1e-6 / data_max
    #    data_normalized[np.abs(data) < 1e-6] = 0
    #else:
    #    data_normalized = np.zeros_like(data)
    
    # 创建绘图
    plt.figure(figsize=(12, 8))
    
    #使用pcolormesh绘制排序后的网格
    mesh = plt.pcolormesh(x_grid, y_grid, data)
    
    # 添加等高线
    #CS = plt.contour(x_grid, y_grid, data_normalized,
    #                levels=np.linspace(-1, 1, 11),
    #                colors='k',
    #                linewidths=0.5,
    #                alpha=0.7)
    #plt.clabel(CS, inline=True, fontsize=9, fmt='%0.1f')
    
    # 添加颜色条和信息
    cbar = plt.colorbar(mesh)
    cbar.set_label(names[2])
    plt.title('Contour Plot of ' + names[2])
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    
    # 添加网格信息
    #plt.grid(True, linestyle='--', alpha=0.3)
    
    # 标记低值区域
    #low_value_mask = np.abs(data) < 1e-6
    #if np.any(low_value_mask):
    #    plt.contourf(x_grid, y_grid, low_value_mask, 
    #                levels=[0.5, 1.5], 
    #                colors='none', 
    #                hatches=['//'],
    #                alpha=0)
    
    plt.tight_layout()


if __name__ == '__main__':
    file_addr = '/home/ac_desktop/syncfiles/postproc_152'
    file_name = 'boundary_quantities_s05600.dat'
    tstep = str('005600')
    iplane = 1080
    
    col_names, blocks = file_read(file_name,file_addr,DEBUG=1)
    
    name = [col_names[2],col_names[3],col_names[7]]
    
    data_all = blocks[tstep]
    df = data_norm(col_names,data_all,iplane,name)
    plot_RZcontour(df,names=name)
    
    plt.show()
    
    


        