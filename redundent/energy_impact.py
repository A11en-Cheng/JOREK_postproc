#!/usr/bin/env python3

#import pandas as pd
from encodings.punycode import T
import data
from scipy.interpolate import interp1d
from scipy.integrate import simps
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import time

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
    plt.savefig(save_path,dpi=600)
    #plt.show()
    plt.close()
    
def interpolate_data(df):
    time = list(df.keys())
    t1 = 0
    t_data = []
    df_new = {}
    for i in range(0,len(time)):
        if i == len(time)-1:
            break
        t = time[i]
        t1 = time[i+1]
        t_new = np.linspace(t, t1, 100)
        t_data = np.append(t_data,t_new)
        
        # 在每个时间点上进行插值
        for m, row in enumerate(df[t]):
            for n, ele in enumerate(row):
                if (m, n) not in df_new:
                    df_new[(m, n)] = np.array([])
                ele_1 = df[t1][m][n]
                data_interp = np.interp(t_new, [t,t1], [ele, ele_1])
                df_new[(m,n)] = np.append(df_new[(m, n)],data_interp) 

        print(f"Interpolated data for time step {t} to {t1}")
    
    print("Interpolation complete.")
    if not os.path.exists('interpolate_data.dat'):
        open('interpolate_data.dat', 'w').write(str(df_new))# Save the interpolated data for debugging
        open('t_data.dat','w').write(str(t_data))
    
    return df_new,t_data
    
    

def compute_delta_q(t_data, q_data, t_eval=None, n_points=200, kind='linear'):
    """
    计算 ΔQ(t) = 1/2 ∫_{t0}^{t} [q_s(t') / √(t-t')] dt' 的离散形式
    
    参数:
        t_data: 已知时间点 (array, 升序)
        q_data: 对应 q_s 值 (array)
        t_eval: 需要计算 ΔQ 的时间点 (array, 默认为 t_data)
        n_points: 数值积分点数
        kind: 插值方法 ('linear', 'cubic', 等)
        
    返回:
        delta_q: ΔQ(t) 在 t_eval 上的值
    """
    t0 = t_data[0]  # 积分下限
    
    # 创建插值函数 (允许外推)
    interp_func = interp1d(
        t_data, q_data, kind=kind, 
        bounds_error=False, fill_value="extrapolate"
    )
    
    if t_eval is None:
        t_eval = t_data
        
    delta_q = np.zeros_like(t_eval)
    
    for i, t in enumerate(t_eval):
        if t <= t0:
            delta_q[i] = 0.0
            continue
            
        # 1. 计算 u_max = √(t - t0)
        u_max = np.sqrt(t - t0)
        
        # 2. 在 [0, u_max] 均匀取点
        u_array = np.linspace(0, u_max, n_points)
        
        # 3. 计算 τ = t - u²
        tau_array = t - u_array**2
        
        # 4. 通过插值获取 q_s(τ)
        q_array = interp_func(tau_array)
        
        # 5. 计算积分 ∫q_s(τ)du
        integral = simps(q_array, u_array)  # 使用辛普森法
        
        # 6. ΔQ(t) = 积分结果 (无需额外系数)
        delta_q[i] = integral
        
    return delta_q

def compute_single_delta(params):
    """处理单个任务的函数，减少内存占用"""
    m, n, q_mn = params
    # 避免在子进程中创建大数组
    t_eval = t_data * 4.1006e-4
    return (m, n), compute_delta_q(t_data, q_mn, t_eval=t_eval, n_points=200, kind='linear')

def test_func(x):
    return x * x

def task_generator():
    m, n = 0, 0
    for ele in data_new:
        q_mn = data_new[(m, n)]
        yield (m, n, q_mn)
        n += 1
        if n == 1080:
            n = 0
            m += 1




if __name__ == '__main__':
    
    try:
        pickle.dumps(compute_delta_q)
        print("函数可序列化")
    except Exception as e:
        print("序列化错误:", e)
    
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(test_func, range(10)))
        print(results)  # 应输出 [0, 1, 4, 9, ...]
    
    if not os.path.exists('interpolate_data.dat'):
        file_addr = '/home/allencheng/ZYX/syncfiles/postproc_145'
        fig_destiny = '/home/allencheng/ZYX/syncfiles/bdy_flux/145'
        tss = [2200,3600,4200,4650,5720]
        #tss = [1600]

        q_data = np.zeros([1080,1080])
        t_data = 0.
        data_set = {}
        data_set[t_data] = q_data
        
        for ts in tss:
            ts = str(ts)
            file_name = 'boundary_quantities_s0'+ts+'.dat'
            tstep = '00'+ts
            iplane = 1080

            col_names, blocks = file_read(file_name,file_addr,DEBUG=1)

            data_all = blocks[tstep]
            
            name = ['phi','theta','heatF_tot_cd']   
            df = data_norm(col_names,data_all,iplane,name)
            t_data     = int(ts)* 4.1006e-4 # Convert to ms
            q_data     = df[name[2]]

            data_set[t_data] = q_data
        df,col_names,blocks,data_all = None, None, None, None  # Clear memory
        data_new,t_data = interpolate_data(data_set)
        # Interpolate data for each time step
        print("Interpolation done.")
        data_set = None  # Clear memory after interpolation
        
    
    else:
        with open('interpolate_data.dat', 'r') as f:
            data_new = eval(f.read())
        
        with open('t_data.dat', 'r') as f:
            t_data = eval(f.read())
            
        print("Loaded existing interpolated data.")

        print("Data loaded.")
        
        


    # parallel
    DQ = np.zeros((1080, 1080, 500))  # 预分配结果数组
    
    # 调整进程数和批处理大小
    max_workers = min(4, os.cpu_count())  # 减少并行进程数
    batch_size = 500  # 小批量处理
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = task_generator()
        futures = {}
        completed = 0
        total_tasks = 1080 * 1080
        
        # 使用进度条
        
        while completed < total_tasks:
            # 提交一批任务
            for _ in range(batch_size):
                try:
                    task = next(tasks)
                    print(task)
                    future = executor.submit(compute_single_delta, task)
                    futures[future] = task
                    print(f"Submitted task for ({task[0]}, {task[1]})")
                except StopIteration:
                    break
            
            # 处理完成的任务
            for future in as_completed(futures):
                try:
                    (m, n), result = future.result()
                    DQ[m, n] = result
                    completed += 1
                    del futures[future]
                    print('completed ({},{})'.format(m, n))
                except Exception as e:
                    print(f"Error processing task: {e}")
                
                # 更新进度后跳出内层循环，继续提交新任务
                break
                
            # 控制内存使用
            if len(futures) > max_workers * 2:
                time.sleep(5)
                print('waiting')# 减缓任务提交速度
                
            
    
    '''DQ = np.zeros((1080,1080,len(t_data)))
    t_data = t_data[0:2200]
    m,n = 0,0
    for ele in data_new:
        q_mn = data_new[(m, n)]
        DQ[m,n] = compute_delta_q(t_data, q_mn, t_eval=t_data*4.1006e-4, n_points=200, kind='linear')
        n+=1
        if n == 1080:
            n = 0
            m += 1
        print(f"Processed ({m}, {n})")    
        
    print("DQ done.")'''
    try: 
        while True: # 从用户获取输入 
            t = input("input time(within [2200,3600,4200,4650,5720]): \n")* 4.1006e-4 # 接收用户输入 
            plot_df = {}
            plot_df['phi'] = df['phi']
            plot_df['theta'] = df['theta']
            plot_df['energy_impact'] = DQ[:,:,t]
            
            name = ['phi','theta','energy_impact']
            
            plt.figure(figsize=(8, 6))
            plot_RZcontour(plot_df, name, os.path.join(fig_destiny, f'energy_impact_{t}.png'), lim=[0, 1e6], log=1)

            plt.show()

    except KeyboardInterrupt: 
        print("结束交互") 

            
            
