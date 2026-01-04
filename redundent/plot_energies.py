#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import re

def plot_dat_file(file_path, log_y=False, x_limit=None, si_index=None, Xpt_loc=[], Right_legend=False):
    """
    读取带表头的 .dat 文件并绘制所有数据列对比第一列(Time)。
    
    参数:
    file_path (str): .dat 文件的路径
    log_y (bool): 是否使用对数纵坐标 (Log Scale)
    x_limit (tuple): 可选，设置X轴范围，例如 (0, 0.5)
    """
    
    if not os.path.exists(file_path):
        print(f"文件未找到: {file_path}")
        return
    
    filename = os.path.basename(file_path)[:-4]
    # step 1: 智能解析表头
    columns = []
    with open(file_path, 'r', encoding='utf-8') as f:
        header_line = f.readline()
        # 使用正则表达式查找所有双引号中间的内容
        # r'"(.*?)"' 的意思是：找到一对双引号，并提取中间的任意字符
        columns = re.findall(r'"(.*?)"', header_line)
    
    if not columns:
        print("错误：无法在第一行解析到带引号的表头。请检查文件格式。")
        return

    print(f"成功解析表头 ({len(columns)}列): {columns}")

    # step 2: 读取数据体
    # skiprows=1: 跳过已经被我们解析过的第一行
    # header=None: 告诉 pandas 不要自己去猜表头，我们后面手动安上去
    # sep=r'\s+': 数据部分依然是用空格分隔的
    try:
        df = pd.read_csv(file_path, sep=r'\s+', skiprows=1, header=None, engine='python')
    except Exception as e:
        print(f"数据读取失败: {e}")
        return

    # step 3: 校验并赋值列名
    if len(df.columns) == len(columns):
        df.columns = columns
    else:
        print(f"警告：表头列数 ({len(columns)}) 与 数据列数 ({len(df.columns)}) 不匹配！")
        print("尝试强制对齐，可能会导致列名错位...")
        # 截断或补齐
        df.columns = columns[:len(df.columns)]
    

    x_col = df.columns[0]
    y_cols = df.columns[1:]
    
    # 3. 绘图风格设置 (保持之前的清晰风格)
    sns.set_context("notebook", font_scale=1.2)
    sns.set_style("ticks")
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)

    # 4. 自动生成配色
    # 使用 'husl' 调色板，根据由多少列数据自动分配颜色
    colors = sns.color_palette(None, len(y_cols))

    print(f"正在绘制 {len(y_cols)} 条数据曲线...")

    # 5. 循环绘制每一列
    for i, col_name in enumerate(y_cols):
        # 提取数据
        x_data = df[x_col]
        x_col_label = x_col
        if si_index is not None and si_index>0:
            x_data = x_data * si_index *1e3  # 转换为毫秒单位
            x_col_label = f"{x_col} (ms)"
        y_data = df[col_name]
        
        # 绘图
        ax.plot(x_data, y_data, 
                label=col_name, 
                color=colors[i], 
                linewidth=3, 
                alpha=0.8) # 保持通透感

    if Xpt_loc is not []:
        for xpt in Xpt_loc:
            if type(xpt) is list:
                ax.axvline(x=xpt[0], color='red', linestyle='--', linewidth=2)
                if log_y:
                    ax.text(xpt[0]*1.001, xpt[1], f't={xpt[0]:.2f}', rotation=90, verticalalignment='top', color='red')
                else:
                    ax.text(xpt[0]*1.001, xpt[1], f't={xpt[0]:.2f}', rotation=90, verticalalignment='top', color='red')
            else:
                ax.axvline(x=xpt, color='red', linestyle='--', linewidth=2)
                if log_y:
                    ax.text(xpt*1.001, ax.get_ylim()[1]*1e-13, f't={xpt:.2f}', rotation=90, verticalalignment='top', color='red')
                else:
                    ax.text(xpt*1.001, ax.get_ylim()[1], f't={xpt:.2f}', rotation=90, verticalalignment='top', color='red')
    # 6. 设置坐标轴 (LogY 开关)
    if log_y:
        ax.set_yscale('log')
        ax.set_ylabel(f"Energies (Log Scale)", fontsize=14)
        title_suffix = " (Log-Y)"
    else:
        ax.set_ylabel(f"Energies (Linear)", fontsize=14)
        title_suffix = ""
        
    if x_limit:
        ax.set_xlim(x_limit)

    # 7. 美化细节
    ax.set_title(f"{filename}{title_suffix}", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(x_col_label, fontsize=16)
    
    # 网格线
    ax.grid(True, which="major", linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    if log_y:
        # 如果是对数坐标，增加次级网格线会有助于读数
        ax.grid(True, which="minor", linestyle=':', linewidth=0.3, color='gray', alpha=0.3)

    # 去除多余边框
    sns.despine()

    # 图例设置 (如果列数太多，把图例放到外侧，否则放自动位置)
    if len(y_cols) > 6 or Right_legend==True:
        # 放在图外右侧
        ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
        plt.tight_layout() # 自动调整布局防止切断图例
    else:
        # 放在图内最佳位置
        ax.legend(frameon=True, framealpha=0.9, loc='best', fontsize=12)
        plt.tight_layout()

    plt.show()

# ==========================================
# 辅助功能：生成符合你格式的测试文件
# ==========================================
def create_dat_file():
    content = """%"time" "E_{mag,00}" "E_{mag,01}" "E_{mag,02}" "E_{kin,00}" "E_{kin,01}"
2.0000E-02 3.4233E-02 1.0000E-05 0.0000E+00 1.7895E-14 5.0000E-14
4.0000E-02 3.4233E-02 2.0000E-05 0.0000E+00 7.6417E-14 8.0000E-14
6.0000E-02 3.4233E-02 4.0000E-05 0.0000E+00 1.6230E-13 1.2000E-13
8.0000E-02 3.4233E-02 8.0000E-05 0.0000E+00 2.6307E-13 3.0000E-13
1.0000E-01 3.4233E-02 1.6000E-04 0.0000E+00 3.7313E-13 5.0000E-13
1.2000E-01 3.4233E-02 3.2000E-04 0.0000E+00 4.9063E-13 8.0000E-13
"""
    with open("simulation_data.dat", "w") as f:
        f.write(content)
    print("已生成测试文件 'simulation_data.dat'")

if __name__ == "__main__":
    # 1. 生成测试数据
    #create_dat_file()

    # 2. 运行绘图
    # 模式 A: 线性坐标 (log_y=False)
    # plot_dat_file("simulation_data.dat", log_y=False)
    
    # 模式 B: 对数坐标 (log_y=True) -> 推荐用于观察 E-14 这种小数值
    #Xpt_loc = [[3.1590,0.01],[3.5723,0.01]]
    #Xpt_loc = [3.1590,3.5723]
    Xpt_loc = [[4.8372,1e-10],[4.9173,1e-10]]
    plot_dat_file("/home/ac_desktop/syncfiles/energies.dat", log_y=True, si_index=4.1006E-07, Xpt_loc=Xpt_loc, Right_legend = True)
    