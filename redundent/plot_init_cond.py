# -*- coding: utf-8 -*-
    
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
import seaborn as sns
import numpy as np

def plot_with_dual_axis(file_pattern, file_extension='txt'):
    # 1. 查找并排序文件
    if os.path.isdir(file_pattern):
        files = glob.glob(os.path.join(file_pattern, f"*.{file_extension}"))
    else:
        files = glob.glob(file_pattern)
    
    if not files:
        print(f"未找到匹配的文件: {file_pattern}")
        return

    files.sort() # 确保顺序一致，最后一个文件即为列表最后一个
    num_files = len(files)
    print(f"共找到 {num_files} 个文件。")
    print(f"前 {num_files - 2} 个文件将绘制在左轴，最后一个文件 '{os.path.basename(files[-1]),os.path.basename(files[-2])}' 将绘制在右轴。")

    # 2. 风格设置
    sns.set_context("notebook", font_scale=1.2)
    # 注意：使用 twinx 时，sns.set_style("ticks") 可能会有一些冲突，这里手动设置风格更稳健
    plt.rcParams['axes.grid'] = True 
    plt.rcParams['grid.alpha'] = 0.5
    plt.rcParams['grid.linestyle'] = '--'

    # 3. 创建画布和主坐标轴
    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=120)
    
    # 创建次坐标轴 (共享X轴)
    ax2 = ax1.twinx()

    # 4. 生成配色
    # 使用 'husl' 调色板，保证颜色区分度
    colors = sns.color_palette("husl", num_files)

    # 5. 循环绘图
    lines = []  # 收集图例句柄
    labels = [] # 收集图例标签

    for i, file in enumerate(files):
        file_name = os.path.basename(file)
        is_last_file = False
        if i >= num_files - 1:
            is_last_file = True
        
        try:
            # 读取数据
            
            df = pd.read_csv(file, header=None, sep=r'\s+', engine='python')
            x = df.iloc[:, 0]
            y = df.iloc[:, 1]
            if y[0] < 0:
                y = -y
                file_name += " (Inverted)"
                
            file_name = file_name.replace('jorek_', ' ')
            curr_color = colors[i]
            
            
            # 核心逻辑：区分左右轴
            if not is_last_file:
                # 普通文件 -> 左轴 ax1
                line = ax1.plot(x, y, label=file_name, color=curr_color, linewidth=2, alpha=0.7)
                lines.append(line[0])
                labels.append(file_name)
                
            else:
                # 最后一个文件 -> 右轴 ax2
                line = ax2.plot(x, y, label=file_name, color=curr_color, linewidth=2, alpha=1, linestyle='--')
                lines.append(line[0])
                labels.append(file_name + " (Right Axis)")
                
                # --- 设置右轴范围 ---
                # 假定上限是数据的最大值，下限为0
                max_val = y.max()
                #ax2.set_ylim(0, max_val)
                
                # --- 右轴视觉美化 ---


                ax2.tick_params(axis='y')
                ax2.set_ylabel(f"{file_name}", fontsize=12)

        except Exception as e:
            print(f"读取文件 {file_name} 时出错: {e}")

    # 6. 设置通用属性
    ax1.set_xlabel("$Psi_{norm}$", fontsize=12)
    ax1.set_ylabel("Data (Left Axis)", fontsize=12)
    ax1.set_title("Initial Conditions", fontsize=16, pad=20)

    # 这里的 grid 只显示 ax1 的，避免双重网格显得杂乱
    ax1.grid(True)
    ax2.grid(False) # 关闭右轴网格

    # 7. 合并图例 (Legend)
    # 因为分了两个轴，如果直接调 legend 会分开显示，这里我们手动合并
    # 放在左上角或最佳位置
    ax1.legend(lines, labels, loc='best', frameon=True, framealpha=0.8, fancybox=True)

    plt.tight_layout()
    plt.show()

# ==========================================
# 测试数据生成器 (为了展示不同量级的数据效果)
# ==========================================
def create_dual_axis_dummy_data():
    if not os.path.exists("test_data_dual"):
        os.makedirs("test_data_dual")
    x = np.linspace(0, 10, 100)
    
    # 1. 生成 3 个普通量级的数据 (0~1 之间)
    for i in range(3):
        y = np.sin(x + i*0.5) * 0.5 + 0.5  # 范围大约在 0-1
        np.savetxt(f"test_data_dual/data_{i+1:02d}.txt", np.column_stack([x, y]), fmt='%.4f')
    
    # 2. 生成 1 个量级完全不同的数据 (例如 0~1000) 作为最后一个文件
    # 这样能明显看出双轴的作用
    y_large = (np.sin(x) + 1.5) * 400 # 范围大约在 200-1000
    np.savetxt(f"test_data_dual/z_last_data.txt", np.column_stack([x, y_large]), fmt='%.4f')
    
    print("已生成测试数据：前3个文件数值小，最后一个文件数值大。\n")

if __name__ == "__main__":
    # 生成测试数据
    #create_dual_axis_dummy_data()
    
    # 运行绘图
    plot_with_dual_axis("/home/ac_desktop/XL50-U/XL50-U_1.5.2/init_cond/*")