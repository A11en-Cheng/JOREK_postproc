import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import sys

# ==========================================
# 1. 核心解析逻辑
# ==========================================
def parse_file(file_path, custom_header=None):
    """
    解析物理模拟数据文件。
    自动寻找以 # 开头的列定义，忽略 # time step 等无关注释。
    """
    if not os.path.exists(file_path):
        print(f"Error: 文件未找到 -> {file_path}")
        return None

    detected_header = None
    
    # 预扫描：寻找表头行
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            clean_line = line.strip()
            # 假设包含 "Psi_N" 或 "rho" 的那一行是表头
            # 这里做一个通用的启发式判断：以 # 开头，且分割后长度大于1，且不是纯数字
            if clean_line.startswith('#') and not "time step" in clean_line:
                # 尝试解析潜在的表头
                parts = clean_line.replace('#', '').strip().split()
                if len(parts) > 1:
                    detected_header = parts
                    break
    
    # 读取数据：comment='#' 会自动忽略掉所有带 # 的行 (包括表头和 time step)
    try:
        df = pd.read_csv(file_path, sep=r'\s+', comment='#', header=None, engine='python')
    except pd.errors.EmptyDataError:
        print(f"Warning: 文件为空或格式无法识别 -> {file_path}")
        return None

    # 应用表头
    if custom_header:
        # 如果用户在命令行指定了表头，优先使用用户的
        if len(custom_header) == len(df.columns):
            df.columns = custom_header
        else:
            print(f"Warning: 自定义表头长度 ({len(custom_header)}) 与数据列数 ({len(df.columns)}) 不匹配。使用默认索引。")
    elif detected_header:
        # 如果自动检测到了表头
        if len(detected_header) == len(df.columns):
            df.columns = detected_header
        else:
            # 这种情况通常是因为表头行有一些多余的空格或符号，尝试截取
            df.columns = detected_header[:len(df.columns)]
    else:
        # 既没指定也没检测到，使用 Col_0, Col_1...
        df.columns = [f"Col_{i}" for i in range(len(df.columns))]

    return df

# ==========================================
# 2. 绘图逻辑
# ==========================================
def plot_data(files, args):
    # 风格设置
    sns.set_context("notebook", font_scale=1.2)
    sns.set_style("ticks")
    
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    
    # 动态调色板
    # 估算总线条数：文件数 * (列数 - 1)
    # 为了防止颜色混乱，这里采用策略：不同文件用不同色系/标记，或者简单地按顺序给色
    total_lines_estimate = len(files) * 3 
    colors = sns.color_palette("husl", total_lines_estimate)
    color_idx = 0
    ax2 = ax.twinx()
    lines = []  # 收集图例句柄
    labels = [] # 收集图例标签

    for file_path in files:
        df = parse_file(file_path, args.header)
        if df is None: continue

        file_name = os.path.basename(file_path)
        title= args.title
        axis = ax
        if args.right_axis and file_path == files[-1]:
            axis = ax2
        # 默认第一列是 X 轴
        x_col = df.columns[0]
        y_cols = df.columns[1:]
        
        X = df[x_col]

        for y_col in y_cols:
            Y = df[y_col]
            
            # 处理绝对值
            is_abs_applied = False
            if args.abs:
                Y = Y.abs()
                is_abs_applied = True
            
            # 构建 Label
            label_str = fr"{y_col}"
            if is_abs_applied:
                label_str = fr"$|{y_col}|$"
            
            # 线型区分
            linetype = '-'
            
            if args.right_axis and file_path == files[-1]:
                linetype = '--'
                label_str += " (Right Axis)"
            
            
            line = axis.plot(X, Y, 
                    label=label_str, 
                    color=colors[color_idx % len(colors)], 
                    linewidth=2.5, 
                    alpha=0.85,
                    linestyle=linetype)
            lines.append(line[0])
            labels.append(label_str)
            
            
            
            if args.right_axis and file_path == files[-1]:
                axis.set_ylabel(fr"$|{y_col}|$ (Right Axis)", fontsize=14)
                max_val = Y.max()
                ax2.set_ylim(0, max_val)
            color_idx += 1

    # 坐标轴与标题
    ax.set_xlabel(fr"${x_col}$ (Axis 0)", fontsize=14)
    y_label_str = "Values"
    if args.abs: y_label_str = "Absolute Value"
    if args.logy: y_label_str += " (Log Scale)"
    ax.set_ylabel(y_label_str, fontsize=14)
    
    if args.logy:
        ax.set_yscale('log')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # 网格与边框
    ax.grid(True, which="major", linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
    if args.logy:
        ax.grid(True, which="minor", linestyle=':', linewidth=0.3, color='gray', alpha=0.3)
    
    # 这一行会默认移除 ax 的右边框和上边框
    sns.despine()
    
    # ------------------ 修改开始 ------------------
    # 针对右侧坐标轴 (ax2) 的特殊处理
    if args.right_axis:
        # 1. 确保右侧的轴线（脊柱）是可见的
        ax2.spines['right'].set_visible(True)
        # 2. 设置轴线颜色（可选，保持黑色或跟随主题）
        ax2.spines['right'].set_color('black')
        # 3. 关键步骤：隐藏刻度线 (length=0 或 right=False)，但保留文字标签
        ax2.tick_params(axis='y', which='both', right=False) 
    else:
        # 如果没用到右轴，最好把它彻底隐藏，防止显示多余的边框
        ax2.axis('off')
    
    # 图例处理
    ax.legend(lines, labels, frameon=True, framealpha=0.9, loc='upper right', fontsize=14)
    
    plt.tight_layout()
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        plt.savefig(os.path.join(args.save_dir, f'profile_evolution.png'), dpi=150)
    else:
        plt.show()

# ==========================================
# 3. 命令行入口 (Argparse)
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="物理模拟数据绘图工具")
    
    # 必须参数：文件名 (支持多个文件)
    parser.add_argument('files', nargs='+', help='输入的数据文件路径 (支持多个文件, 用空格分隔)')
    
    # 可选参数：强制替换 Header
    parser.add_argument('--header', nargs='+', help='自定义列名 (按顺序)，如果不指定则尝试自动从文件读取', default=None)
    
    parser.add_argument('--right_axis', action='store_true', help='将最后一个文件绘制在右侧 Y 轴上')
    
    parser.add_argument('--title', type=str, help='图表标题', default='Profile Evolution')
    
    # 可选参数：取绝对值
    parser.add_argument('--abs', action='store_true', help='对所有 Y 轴数据取绝对值 (|y|)')
    
    # 可选参数：Log Y
    parser.add_argument('--logy', action='store_true', help='使用对数纵坐标')
    
    parser.add_argument('--save_dir', type=str, help='保存图像的目录路径', default=None)

    # 解析参数
    # 如果是在 Jupyter/Colab 环境下运行，sys.argv 可能包含无关参数，需要特殊处理
    # 这里假设是在标准终端运行。如果在 IDE 中运行，请手动模拟 args
    if 'ipykernel_launcher.py' in sys.argv[0]:
        # === 调试模式：在此处手动输入参数进行测试 ===
        print("检测到 Jupyter 环境，使用模拟参数...")
        # 生成测试文件以便调试
        create_test_file()
        dummy_args = ['test_profile.txt', '--abs', '--logy'] 
        args = parser.parse_args(dummy_args)
    else:
        args = parser.parse_args()

    plot_data(args.files, args)

# ==========================================
# 辅助：生成符合你描述的测试文件
# ==========================================
def create_test_file():
    content = """# Psi_N                 rho                    T                      zj
# time step #000000
 1.000E-03  9.999E-01  4.313E-02 -1.366E+00
 1.106E-02  9.999E-01  4.227E-02 -1.348E+00
 2.114E-02  9.998E-01  4.142E-02 -1.330E+00
 3.121E-02  9.995E-01  4.058E-02 -1.312E+00
 4.127E-02  9.992E-01  3.975E-02 -1.294E+00
"""
    with open("test_profile.txt", "w") as f:
        f.write(content)
    print("已生成测试文件 'test_profile.txt'")

if __name__ == "__main__":
    main()