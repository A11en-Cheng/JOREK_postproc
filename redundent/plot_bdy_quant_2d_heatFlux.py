#!/usr/bin/env python3

#import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

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

    
    lenth = df['phi'].shape[0]
    row = int(iplane)
    col = int(lenth//iplane)
    print(row,col)
    
    unique_x = np.unique(x_sorted)
    unique_y = np.unique(y_sorted)
    nx = len(unique_x)
    ny = len(unique_y)
    
    x_grid = np.reshape(x_sorted, (row,col))
    y_grid = np.reshape(y_sorted, (row,col))
    data_grid = np.reshape(data_sorted, (row,col))
    sorted_df = {
        names[0]: x_grid,
        names[1]: y_grid,
        dataidx : data_grid
    }
    print('Sorted.')
    return sorted_df

def plot_RZcontour(df, names, save_path,lim = [],log=0):
    """
    使用排序后的数据绘图
    """
    plt.figure(figsize=(8, 6))
    # 直接从排序后的网格获取数据
    x_grid = df[names[0]]
    y_grid = df[names[1]]
    data = df[names[2]]
    norm = plt.Normalize(vmin=np.nanmin(data),vmax=np.nanmax(data))
    if lim != []:
        data = np.where(data>lim[0],data,lim[0])
        data = np.where(data<lim[1],data,lim[1])
        norm = plt.Normalize(vmin=lim[0],vmax=lim[1])
        
    if log ==1:
        norm=LogNorm(vmin=lim[0], vmax=lim[1])
    #使用pcolormesh绘制排序后的网格
    mesh = plt.pcolormesh(x_grid, y_grid, data,norm=norm)

    x_point = [1,1,1,1]
    y_point = [1.95,1.45,4.86,4.33]
    labels = ['inner_upper','outer_upper','outer_lower','inner_lower']
    for i, point in enumerate(x_point):
        plt.annotate(
            f'{labels[i]} (theta={y_point[i]:.2f})',  # 文本内容
            xy=(x_point[i], y_point[i]),           # 箭头指向的点
            xytext=(x_point[i] + 0.3, y_point[i] + 0.3 * (-1)**i),  # 文本位置
            arrowprops=dict(                  # 箭头属性
                arrowstyle='->',              # 箭头样式
                color='red',                  # 箭头颜色
                linewidth=1.5,                # 线宽
                shrinkA=5,                    # 箭头起点离点的距离
                shrinkB=5,                    # 箭头终点离点的距离
                connectionstyle='arc3,rad=0.3', # 箭头弯曲度
                alpha=0.8
            ),
            bbox=dict(                       # 文本框属性
                boxstyle='round,pad=0.3',     # 圆角框
                fc='white',                  # 填充色
                alpha=0.4                     # 透明度
            )
        )
    
        # 在点上添加标记
        plt.scatter(x_point[i], y_point[i], s=1, c='red', marker='o')

    #plt.ylim(0.5,2.5)
    # 添加颜色条和信息
    cbar = plt.colorbar(mesh)
    cbar.set_label(names[2])
    plt.title('Contour Plot of ' + names[2])
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    
    plt.tight_layout()
    #plt.savefig(save_path,dpi=600)
    plt.show()
    plt.close()
    
def addedRZ(df,names):
    '''
    names 
    '''

def find_max(df,names,timestep):
    x_grid = df[names[0]]
    y_grid = df[names[1]]
    data = df[names[2]]
    
    max_pos = np.where(data == np.max(data))
    
    
    return x_grid[max_pos[0],max_pos[1]],y_grid[max_pos[0],max_pos[1]]

    


if __name__ == '__main__':
    file_addr = '/home/ac_desktop/syncfiles/postproc_145'
    fig_destiny = '/home/ac_desktop/syncfiles/postproc_145/figures'
    tss = [2200,3600,4200,4650,5720,6000]
    #tss = [3600]
    MX,MY = [],[]
    for ts in tss:
        ts = str(ts)
        file_name = 'boundary_quantities_s0'+ts+'.dat'
        tstep = '00'+ts
        iplane = 1080

        col_names, blocks = file_read(file_name,file_addr,DEBUG=1)


        data_all = blocks[tstep]
        
        name = ['phi','theta','heatF_tot_cd']  # Assuming these are the column names you want to use
        df = data_norm(col_names,data_all,iplane,name)
        m_x,m_y = find_max(df,name,tstep)
        MX.append(m_x)
        MY.append(m_y)
        plot_RZcontour(df, name, os.path.join(fig_destiny, f'boundary_quantities_s0{ts}.png'), lim=[1e5,3e8],log=1)
    
        

            
            
