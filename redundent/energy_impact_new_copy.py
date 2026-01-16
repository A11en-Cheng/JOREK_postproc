#!/usr/bin/env python3

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

from tqdm import tqdm
import gc

# 常量定义
MAX_WORKERS = 30
CHUNK_SIZE = 540  # 任务分块大小
INTEGRATION_POINTS = 600 #5720  # 积分点数
INTERP_POINTS =600 #5720  # 插值点数
DEBUG = 0
use_DQ = 0
test_flag = 1



def read_boundary_file(file_path, DEBUG=0):
    """优化后的文件读取函数，使用numpy直接加载数据"""
    blocks = {}
    current_block = None
    data_rows = []
    col_names = []
    t_now = np.nan
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('# '):
                if line.startswith('# time step'):
                    t_now = float(line.split()[-1])
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
    
    return col_names, blocks, t_now

def reshape_to_grid(block,col_names,names,iplane):
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
    
    x = df[names[0]]  #phi
    y = df[names[1]]  #theta

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

    
    lenth = 540*540
    row = int(iplane)
    col = int(lenth//iplane)
    print(row, col)
    
    x_grid = np.reshape(x_sorted, (row, col))
    y_grid = np.reshape(y_sorted, (row, col))
    data_grid = np.reshape(data_sorted, (row, col))
    
    sorted_df = {
        names[0]: x_grid,
        names[1]: y_grid,
        dataidx : data_grid
    }
    print('Sorted.')
    return sorted_df

def compute_delta_q(t_raw, q_raw, t_eval, n_points=INTEGRATION_POINTS, test_flag=0):
    """向量化计算delta_q"""
    if t_raw.size == 0 or q_raw.size == 0:
        return np.zeros_like(t_eval)
    
    # 创建插值函数（使用线性插值外推）
    
    interp_func = interp1d(
            t_raw, q_raw, kind='linear',
            bounds_error=False, 
            fill_value=(q_raw[0], q_raw[-1] ),
            assume_sorted=True
        )
    
    # 预计算u_max数组
    u_max = np.sqrt(np.maximum(t_eval - t_raw[0], 0))
    valid_mask = u_max > 0
    
    # 初始化结果数组
    delta_q = np.zeros_like(t_eval)
    
    # 处理有效点
    for i, valid in enumerate(valid_mask):
        if not valid:
            continue
        
        scale_factor =  i / len(u_max)
        current_n_points = max(0, int(n_points * scale_factor))
        # 生成积分点
        u_array = np.linspace(0, u_max[i], current_n_points+1)

        tau_array = t_eval[i] - u_array**2
        
        # 向量化插值
        q_array = interp_func(tau_array)
        
        # 使用梯形法则积分（比辛普森更快）
        integral = np.trapz(q_array, u_array)
        delta_q[i] = integral
        
    return delta_q

def process_timestep(args):
    """处理单个时间步的任务函数"""
    ts, file_addr, iplane, names = args
    file_name = f'boundary_quantities_s0{ts}.dat'
    file_path = os.path.join(file_addr, file_name)
    
    try:
        col_names, blocks, t_now = read_boundary_file(file_path)
        
        block_data = blocks.get(f'00{ts}')
        if block_data is None:
            return ts, None
        
        grid_data = reshape_to_grid(block_data, col_names, names, iplane)
        return t_now, grid_data[names[2]]
    except Exception as e:
        print(f"Error processing timestep {ts}: {e}")
        return ts, None

def find_max(df,names,time):
    x_grid = df[names[0]]
    y_grid = df[names[1]]
    data = df[names[2]]
    
    max_val = np.max(data)
    if max_val == 0:
        print(f"Warning: Maximum value is zero at time {time}.")
        return None, None, max_val
    rows, cols = np.where(data == max_val)
    
    
    return rows,cols,max_val

def plot_RZcontour(df, names, save_path,t_now = np.nan,lim = [],log=0,max_point = []):
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
        if max_point != []:
            print(len(max_point))
            norm = plt.Normalize(vmin=lim[0], vmax=max_point[2])
    if log ==1:
        norm=LogNorm(vmin=lim[0], vmax=lim[1])
        if max_point != []:
            print(len(max_point))
            norm = LogNorm(vmin=lim[0], vmax=max_point[2])

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
        
    if max_point != []:
        plt.scatter((max_point[0]-540)*2*np.pi/540, (max_point[1])*2*np.pi/540, s=1, c='red', marker='o',label=max_point[2])



    #plt.ylim(0.5,2.5)
    # 添加颜色条和信息
    cbar = plt.colorbar(mesh)
    cbar.set_label(names[2])
    if not np.isnan(t_now):
        plt.title(f'Contour Plot of {names[2]} at t={t_now:.2e} s')
    else:
        plt.title('Contour Plot of ' + names[2])
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    
    plt.tight_layout()
    plt.savefig(save_path,dpi=600)
    #plt.show()
    plt.close()

def plot_RZcontour_test(df, names, save_path,t_now = np.nan,lim = [],log=0):
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

    '''x_point = [1,1,1,1]
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
        plt.scatter(x_point[i], y_point[i], s=1, c='red', marker='o')'''
        
    x_point = [180,540,900,180,540,900,180,540,900]
    y_point = [180,540,900,540,900,180,900,180,540]
    for i, point in enumerate(x_point):
        plt.annotate(
            f'(val={data[x_point[i],y_point[i]]:.2e})',  # 文本内容
            xy=(x_grid[x_point[i]], y_grid[y_point[i]]),           # 箭头指向的点
            xytext=(x_grid[x_point[i]], y_grid[y_point[i]]),  # 文本位置
            
            bbox=dict(                       # 文本框属性
                boxstyle='round,pad=0.3',     # 圆角框
                fc='white',                  # 填充色
                alpha=0.4                     # 透明度
            )
        )


    #plt.ylim(0.5,2.5)
    # 添加颜色条和信息
    cbar = plt.colorbar(mesh)
    cbar.set_label(names[2])
    if not np.isnan(t_now):
        plt.title(f'Contour Plot of {names[2]} at t={t_now:.2e} s')
    else:
        plt.title('Contour Plot of ' + names[2])
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    
    plt.tight_layout()
    plt.savefig(save_path,dpi=600)
    #plt.show()
    plt.close()

def main():
    # 配置参数
    file_addr = './'
    fig_destiny = './fig'
    tss = [ 2550, 5000, 6000]
    iplane = 540
    R,Z = 540,540
    names = ['phi', 'theta', 'heatF_total']
    CHUNK_SIZE = 540
    normalize_factor = 4.1006e-7 #时间归一化因子
    save_name = 'test_DQ_data.pkl'
    if test_flag == 0:
        if use_DQ == 0:

            # read boundary file and reshape data
            data_set = {}
            tasks = [(ts, file_addr, iplane, names) for ts in tss]
            
            
            if DEBUG == 1:
                t_now ,q_data = process_timestep(tasks[0])  # 测试单个任务

            with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tss))) as executor:
                futures = [executor.submit(process_timestep, task) for task in tasks]
                for future in as_completed(futures):
                    t_now, q_data = future.result()
                    if q_data is not None:
                        t_phys = t_now * normalize_factor
                        data_set[t_phys] = q_data.astype(np.float32)
                        print(f"Processed time {t_now:.2e} with shape {q_data.shape}, t_phys={t_phys:.2e}")
            
            data_set[0.e1] = np.zeros((R, Z), dtype=np.float32)  # 添加初始时间步
            
            for t, arr in data_set.items():
                nan_count = np.isnan(arr).sum()
                if nan_count > 0:
                    print(f"Timestep {t}: Found {nan_count} NaN values")
            
            # 步骤2: 时间插值
            t_raw = np.sort(np.array(list(data_set.keys())))
            t_eval = np.linspace(t_raw.min(), t_raw.max(), INTERP_POINTS)
            
            # 预分配结果数组
            n_rows, n_cols = R,Z
            DQ = np.zeros((n_rows, n_cols, len(t_eval)), dtype=np.float32)
            
            # 步骤3: 并行计算delta_q
            tasks = []
            for i in range(n_rows):
                for j in range(n_cols):
                    q_raw = np.array([data_set[t][i, j] for t in t_raw])
                    tasks.append((i, j, t_raw, q_raw, t_eval))

            # 分块处理减少内存压力
            for chunk_idx in tqdm(range(0, len(tasks), CHUNK_SIZE)):
                
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
                #print(chunk_idx )
                if DEBUG == 1 and chunk_idx // CHUNK_SIZE + 1 == 10:
                    break
            
            #保存数据
            with open(os.path.join(file_addr, 'DQ_data_heatF_total.pkl'), 'wb') as f:
                pickle.dump(DQ, f)
                print(f"Saved DQ data to {os.path.join(file_addr, 'DQ_data_heatF_total.pkl')}")
        elif use_DQ == 1:#to be tweaked
            DQ = np.load(os.path.join(file_addr, 'DQ_data_heatF_total.pkl'), allow_pickle=True)
            print(f"Loaded DQ data from {os.path.join(file_addr, 'DQ_data_heatF_total.pkl')}")
            
            n_rows, n_cols = 540, 540
            t_raw = np.array([0.        , 0.00289461, 0.0041455 , 0.00443213, 0.00458365, 0.0047298 , 0.00482247, 0.00489931, 0.00492446])
            t_eval = np.linspace(t_raw.min(), t_raw.max(), INTERP_POINTS)
    
    elif test_flag == 1:
        if use_DQ == 0:
            data_set = {}
            CHUNK_SIZE = 1000
            data_set[0.e1] = np.zeros((300, 300), dtype=np.float32)  # 添加初始时间步
            data_set[2.e3*normalize_factor] = np.zeros((300, 300), dtype=np.float32)
            for j in range(0,3):
                for i in range(0,3):
                    data_set[2.e3*normalize_factor][100*i:100*(i+1),100*j:100*(j+1)] = np.float32(10**(i+j+1))
            #data_set[2000*normalize_factor] = np.zeros((300,300))  # 添加一个时间步   
            #decay_rate = 1e7  # 衰减常数
            #for t in range(0, 2000):
                #if t > 0:
                    #data_set[t*normalize_factor] = np.zeros((300,300)) # data_set[0.e1] * np.exp(-decay_rate * t*normalize_factor)
            t_raw = np.sort(np.array(list(data_set.keys())))
            t_eval = np.linspace(t_raw.min(), t_raw.max(), INTERP_POINTS)
            
            # 预分配结果数组
            n_rows, n_cols = 300, 300
            DQ = np.zeros((n_rows, n_cols, len(t_eval)), dtype=np.float32)

            # 步骤3: 并行计算delta_q
            tasks = []
            for i in range(n_rows):
                for j in range(n_cols):
                    q_raw = np.array([data_set[t][i, j] for t in t_raw])
                    tasks.append((i, j, t_raw, q_raw, t_eval))
            
            # 分块处理减少内存压力
            for chunk_idx in tqdm(range(0, len(tasks), CHUNK_SIZE)):
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
                del chunk, futures
                gc.collect()
                #print(f"Processed chunk {chunk_idx//CHUNK_SIZE + 1}/{(len(tasks)+CHUNK_SIZE-1)//CHUNK_SIZE}")
            
            with open(os.path.join(file_addr, 'test_DQ_data_2000.pkl'), 'wb') as f:
                pickle.dump(DQ, f)
                print(f"Saved DQ data to {os.path.join(file_addr, 'test_DQ_data.pkl')}")
        
        elif use_DQ == 1:
            DQ = np.load(os.path.join(file_addr, 'test_DQ_data.pkl'), allow_pickle=True)
            print(f"Loaded DQ data from {os.path.join(file_addr, 'test_DQ_data.pkl')}")
            t_input = 2000 * normalize_factor
            t_eval = np.linspace(0, 2000* normalize_factor, INTERP_POINTS)
        
        n_rows, n_cols = 300, 300
        with open(os.path.join(file_addr, 'impact_t+val_2000'), 'w') as f:
            pass
        
        for t in np.linspace(0, 2000, 2000):
            t_input = t * normalize_factor
            t_idx = np.abs(t_eval - t_input).argmin()
            with open(os.path.join(file_addr, 'impact_t+val_2000'), 'a') as f:
                f.write(f"{t_input:.6e}\t{DQ[150, 150, t_idx]:.6e}\n")
        
        
                
        
        
    # 步骤4: 结果可视化
    
    '''for t in np.linspace(0, 5720, 5720):
        t_input = t * normalize_factor
        t_idx = np.abs(t_eval - t_input).argmin()
        
        plt.figure(figsize=(10, 8))
        plot_RZcontour({
            names[0]: np.linspace(-np.pi, np.pi, n_cols),
            names[1]: np.linspace(0, 2*np.pi, n_rows),
            names[2]: DQ[:, :, t_idx].T
        }, names, os.path.join(fig_destiny, f'energy_impact_{t_input:.4e}.png'),
                        lim=[1.e5,3.e8], log=1)
        
        save_path = os.path.join(fig_destiny, f'energy_impact_{t_input:.4e}.png')
        print(f"Saved visualization to {save_path}")'''
        
    '''with open(os.path.join(file_addr, 'impact_t+val_2000'), 'w') as f:
        pass'''
         
    for t in t_raw:
        t_input = t 
        t_idx = np.abs(t_eval - t_input).argmin()
        df = {
            names[0]: np.linspace(-np.pi, np.pi, n_cols),
            names[1]: np.linspace(0, 2*np.pi, n_rows),
            names[2]: DQ[:, :, t_idx].T
            }
        max_x,max_y,max_val = find_max(df,names,t_idx)
        mp = []
        if max_val == 0.:
            plot_RZcontour(df, names, os.path.join(fig_destiny, f'152_heatF_total_{t:.2e}.png'), t_now=t_input,lim=[1.e3,1.e8], log=1,max_point = [])
        else:
            plot_RZcontour(df, names, os.path.join(fig_destiny, f'152_heatF_total_{t:.2e}.png'), t_now=t_input,lim=[1.e3,1.e8], log=1,max_point = [max_y,max_x,max_val])
        
        
        save_path = os.path.join(fig_destiny, f't_152_NEW_EI_{t:.2e}.png')
        print(f"Saved visualization to {save_path}")
        

    '''try:
        while True:
            t_input = float(input("Enter time value: "))* normalize_factor
            t_idx = np.abs(t_eval - t_input).argmin()
            
            plt.figure(figsize=(10, 8))
            plot_RZcontour({
                names[0]: np.linspace(-np.pi, np.pi, n_cols),
                names[1]: np.linspace(0, 2*np.pi, n_rows),
                names[2]: DQ[:, :, t_idx].T
            }, names, os.path.join(fig_destiny, f'energy_impact_{t_input:.4e}.png'),
                           lim=[1.e5, 3.e8], log=1)
            
            save_path = os.path.join(fig_destiny, f'energy_impact_{t_input:.4e}.png')
            print(f"Saved visualization to {save_path}")
            
    except KeyboardInterrupt:
        print("Processing completed")
    '''
    
if __name__ == '__main__':
    main()
