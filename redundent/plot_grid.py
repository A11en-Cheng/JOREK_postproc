import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

def parse_grid_blocks(file_path):
    """
    解析被空行分隔的多块数据文件。
    返回一个列表，列表包含多个 numpy 数组，每个数组代表一条网格线。
    """
    if not os.path.exists(file_path):
        print(f"Grid file not found: {file_path}")
        return []

    blocks = []
    current_block = []

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            # 如果是空行，且当前块有数据，说明这一块结束了
            if not line:
                if current_block:
                    blocks.append(np.array(current_block))
                    current_block = []
                continue
            
            # 读取数值
            try:
                parts = line.split()
                if len(parts) >= 2:
                    x, y = float(parts[0]), float(parts[1])
                    current_block.append([x, y])
            except ValueError:
                continue
    
    # 处理文件末尾最后一块
    if current_block:
        blocks.append(np.array(current_block))
        
    return blocks

def load_boundary(file_path):
    """读取简单的两列边界数据"""
    if not os.path.exists(file_path):
        print(f"Boundary file not found: {file_path}")
        return None
    try:
        return np.loadtxt(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def plot_mesh_system(grid_file=None, boundary_files=[], title=None,output_name=None):
    """
    绘制网格系统
    :param grid_file: 主网格文件路径
    :param boundary_files: 包含边界文件路径的列表 ['b1.dat', 'b2.dat']
    """
    
    # 1. 风格设置 (保持统一)
    sns.set_context("notebook", font_scale=1.2)
    sns.set_style("ticks")
    
    fig, ax = plt.subplots(figsize=(4, 10), dpi=120) # 方形画布适合物理网格

    # 2. 绘制网格线 (Grid Lines)
    # 解析数据
    if grid_file is None:
        pass
    else:
        grid_lines = parse_grid_blocks(grid_file)
        print(f"Parsed {len(grid_lines)} grid lines from {grid_file}")

        # 循环绘制每一条网格线
        # 使用较细的线条和低饱和度颜色，作为背景
        for line_data in grid_lines:
            ax.plot(line_data[:, 0], line_data[:, 1], 
                    color='#2c3e50',  # 深灰蓝色
                    linewidth=0.8,    # 细线
                    alpha=0.5)        # 半透明

    # 3. 绘制边界 (Boundaries)
    # 使用高对比度颜色强调边界
    boundary_colors = ['#e74c3c', '#e67e22', '#8e44ad'] # 红、橙、紫
    
    for i, b_file in enumerate(boundary_files):
        b_data = load_boundary(b_file)
        if b_data is not None:
            color = boundary_colors[i % len(boundary_colors)]
            ax.plot(b_data[:, 0], b_data[:, 1],
                    label=f"Boundary {i+1}",
                    color=color,
                    linewidth=1.5, # 边界线加粗
                    linestyle='-')

    # 4. 关键设置：等比例缩放
    # 这一步对于物理网格至关重要，否则圆形会变成椭圆
    ax.set_aspect('equal')

    
    # 5. 装饰
    if title is None:
        title = "Mesh Grid System"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("R/m", fontsize=14)
    ax.set_ylabel("Z/m", fontsize=14)
    
    ax.grid(False) # 通常网格图本身就是网格，不需要背景网格，或者设为很淡
    sns.despine()

    # 如果有边界图例，显示出来
    if boundary_files:
        ax.legend(frameon=True, fancybox=True, loc='best')

    plt.tight_layout()
    plt.show()

# ==========================================
# 辅助：生成测试文件 (完全复制你的数据格式)
# ==========================================
def create_test_files():
    # 1. 主网格数据 (复制你的片段)
    grid_content = """
  8.56773293E-01  1.81024947E-03
  8.56029255E-01 -1.08282165E-03
  8.55285218E-01 -3.97589278E-03
  8.54541181E-01 -6.86896393E-03
  8.53797144E-01 -9.76203508E-03
  8.53053107E-01 -1.26551062E-02
  8.52309070E-01 -1.55481774E-02
  8.51565032E-01 -1.84412485E-02
  8.50820995E-01 -2.13343197E-02
  8.50076958E-01 -2.42273908E-02
  8.49332921E-01 -2.71204619E-02


  8.49332921E-01 -2.71204619E-02
  8.49426371E-01 -2.71622883E-02
  8.49527092E-01 -2.71792817E-02
  8.49633358E-01 -2.71774927E-02
  8.49743444E-01 -2.71629720E-02
  8.49855623E-01 -2.71417703E-02
  8.49968169E-01 -2.71199384E-02
  8.50079357E-01 -2.71035268E-02
  8.50187461E-01 -2.70985863E-02
  8.50290755E-01 -2.71111676E-02
  8.50387513E-01 -2.71473214E-02


  8.50387513E-01 -2.71473214E-02
  8.51019033E-01 -2.42500204E-02
  8.51655258E-01 -2.13537487E-02
  8.52295013E-01 -1.84582489E-02
  8.52937119E-01 -1.55632638E-02
  8.53580403E-01 -1.26685360E-02
  8.54223686E-01 -9.77380817E-03
  8.54865793E-01 -6.87882304E-03
  8.55505547E-01 -3.98332327E-03
  8.56141773E-01 -1.08705154E-03
  8.56773293E-01  1.81024947E-03
"""
    with open("mesh_data.dat", "w") as f:
        f.write(grid_content)

    # 2. 模拟边界数据 (Boundary 1)
    # 创建一个包围这些点的外框
    b1_data = """
    0.8490 -0.030
    0.8570 -0.030
    0.8570  0.005
    0.8490  0.005
    0.8490 -0.030
    """
    with open("boundary_inner.dat", "w") as f:
        f.write(b1_data)

    # 3. 模拟边界数据 (Boundary 2)
    b2_data = """
    0.8450 -0.035
    0.8600 -0.035
    """
    with open("boundary_wall.dat", "w") as f:
        f.write(b2_data)

    print("已生成测试文件: mesh_data.dat, boundary_inner.dat, boundary_wall.dat")

if __name__ == "__main__":
    # 1. 生成数据
    #create_test_files()
    
    # 2. 绘图
    # 将你的文件名填入这里
    plot_mesh_system(
        grid_file="/home/ac_desktop/XL50-U/XL50-U_1.4.5/grid_xpoint.dat", 
        boundary_files=[],
        title="X-point Grid"
    )
    '''
    plot_mesh_system(
        boundary_files=["/home/ac_desktop/XL50-U_nl/wallcontour_updated.dat", "/home/ac_desktop/XL50-U_nl/wallcontour_adjusted.dat"],
        title="Boundary Comparison"
    )
    '''