"""
Configuration and command-line argument parsing for boundary quantities processing.
"""

import argparse
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import sys
from jorek_postproc import get_device_geometry
import numpy as np

@dataclass
class ProcessingConfig:
    """
    处理流程的配置参数。

    Attributes
    ----------
    file_path : str
        边界量数据文件路径
    timesteps : list of str
        要处理的时间步列表
    iplane : int
        环向平面数 (默认1080 for EXL50-U)
    data_name : str
        要提取的物理量列名
    device : str
        设备名称 ('EXL50U', 'ITER', 等)
    data_limits : list of 2 float, optional
        数据显示范围 [min, max]
    norm_factor : float, optional
        归一化因子 (e.g., 单位转换)
    plot_surface : bool
        是否绘制表面图，否则绘制散点图
    plot_overall : bool
        是否绘制整体视图
    log_norm : bool
        是否使用对数色图
    find_max : bool
        是否标记最大值位置
    output_dir : str, optional
        输出目录路径
    xpoints : list of 4 float, optional
        X点坐标 [x1, z1, x2, z2]，用于双X点表面整理
    debug : bool
        调试模式标志
    energy_impact : bool
        是否启用能量冲击计算
    save_convolution : bool
        是否保存卷积计算结果 (.npz)
    mode : str
        处理模式: 'standard', 'energy_impact', 'plot_set'
    """
    file_path: str
    timesteps: List[str]
    data_name: str = 'heatF_tot_cd'
    device: str = 'EXL50U'
    data_limits: List[float] = field(default_factory=lambda: [1e5, 3e8])
    norm_factor: Optional[float] = None
    plot_surface: bool = True
    plot_overall: bool = False
    log_norm: bool = False
    find_max: bool = True
    output_dir: Optional[str] = None
    xpoints: List[float] = field(default_factory=list)
    debug: bool = False
    energy_impact: bool = False
    save_convolution: bool = False
    mode: str = 'standard'
    dim: Optional[str] = None # '2d' or '3d'
    show_left_plot: bool = True
    show_right_plot: bool = True
    use_arc_length: bool = False

    def __post_init__(self):
        """
        验证配置参数的逻辑一致性并应用互斥规则。
        """
        # 兼容性处理
        if self.energy_impact:
            self.mode = 'energy_impact'

        # 维度相关的互斥逻辑
        if self.dim == '2d':
            # 2D模式下，强制关闭3D特有的表面/散点选项（避免混淆）
            self.plot_surface = False 
            # 2D模式下没有"整体视图"开关的概念，通常总是绘制整体图
            # 这里的 plot_overall 在2D下将不再生效或作为默认行为
            
        elif self.dim == '3d':
            # 3D模式下确保如果有相关逻辑能正确回退
            pass

        # 模式相关的默认值修正
        if self.mode == 'plot_set' and not self.dim:
            self.dim = '2d'



def create_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器。

    Returns
    -------
    parser : argparse.ArgumentParser
        配置好的参数解析器
    """
    parser = argparse.ArgumentParser(
        description="JOREK边界量可视化工具",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
示例使用:
  jorek_postproc -f boundary_quantities_s04200.dat -t 4200 --dim 3d
  jorek_postproc -f boundary_quantities.dat -t 5000 --mode plot_set --use_arc_length
  jorek_postproc -f boundary_quantities.dat -m energy_impact --dim 2d --log_norm
        """
    )
    
    # --- 1. 基础文件与数据配置 ---
    group_file = parser.add_argument_group('基础文件配置 (File & Data)')
    group_file.add_argument(
        "-f", "--file_path",
        required=True,
        help="边界量文件路径 (.dat)"
    )
    group_file.add_argument(
        "-t", "--timesteps",
        nargs='+',
        required=True,
        type=str,
        help="时间步列表 (e.g., 4200 4650 5000)"
    )
    group_file.add_argument(
        "-n", "--data_name",
        default='heatF_tot_cd',
        help="物理量名称 (默认: heatF_tot_cd)"
    )
    group_file.add_argument(
        "-d", "--device",
        default='EXL50U',
        choices=['EXL50U', 'ITER'],
        help="设备名称"
    )
    group_file.add_argument(
        "-o", "--output_dir",
        default=None,
        help="输出目录 (默认当前目录下生成)"
    )

    # --- 2. 模式选择 ---
    group_mode = parser.add_argument_group('运行模式 (Mode)')
    group_mode.add_argument(
        "-m", "--mode",
        type=str,
        default='standard',
        choices=['standard', 'energy_impact', 'plot_set'],
        help="""处理模式:
  standard      : 标准单步处理 (默认)
  energy_impact : 能量冲击/积分计算
  plot_set      : 直接生成图集 (默认2D视图)"""
    )
    group_mode.add_argument(
        "--energy_impact",
        action='store_true',
        help="[Deprecated] 兼容旧标志，等同于 --mode energy_impact，将弃用"
    )

    # --- 3. 维度与视图配置 ---
    group_dim = parser.add_argument_group('维度与绘图 (Dimension & View)')
    group_dim.add_argument(
        "--dim",
        type=str,
        choices=['2d', '3d'],
        default=None,
        help="绘图维度: '2d' (展开图), '3d' (空间分布). 若不指定，standard默认为3d，plot_set默认为2d"
    )
    group_dim.add_argument(
        "--use_arc_length",
        action='store_true',
        default=False,
        help="使用弧长(Arc Length)作为Y轴 (仅2D模式有效)"
    )
    group_dim.add_argument(
        "--plot_surface",
        action='store_true',
        default=True,
        help="[3D] 绘制表面图 (默认)"
    )
    group_dim.add_argument(
        "--plot_scatter",
        action='store_true',
        help="[3D] 绘制散点图"
    )
    
    # 2D 视图特定选项 (针对 plot_set 和 dim=2d)
    group_dim.add_argument(
        "--left_only",
        action='store_true',
        default=False,
        help="[2D] 仅绘制左侧 R-Z 截面图"
    )
    group_dim.add_argument(
        "--right_only",
        action='store_true',
        default=False,
        help="[2D] 仅绘制右侧展开图"
    )
    group_dim.add_argument(
        "--overall",
        action='store_true',
        default=False,
        help="[3D/EI] 绘制整体视图 (默认绘制所有部件)"
    )

    # --- 4. 数据处理与调试 ---
    group_proc = parser.add_argument_group('数据处理选项 (Processing)')
    group_proc.add_argument(
        "--limits","-lim",
        nargs=2,
        type=float,
        metavar=('MIN', 'MAX'),
        default=None,
        help="数据截断范围 [min max]"
    )
    group_proc.add_argument(
        "--norm_factor",
        type=float,
        default=4.1006E-07,
        help="归一化因子"
    )
    group_proc.add_argument(
        "--log_norm",
        action='store_true',
        default=False,
        help="使用对数色标"
    )
    group_proc.add_argument(
        "--find_max",
        action='store_true',
        default=False,
        help="标记最大值点"
    )
    group_proc.add_argument(
        "--xpoints",
        nargs=4,
        type=float,
        default=None,
        metavar=('X1', 'Z1', 'X2', 'Z2'),
        help="X点坐标校正"
    )
    group_proc.add_argument(
        "--save_convolution",
        action='store_true',
        default=False,
        help="保存卷积中间结果"
    )
    group_proc.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="调试模式"
    )

    return parser


def parse_args(args=None) -> ProcessingConfig:
    """
    解析命令行参数并返回配置对象。
    """
    parser = create_parser()
    parsed = parser.parse_args(args)
    
    # 处理互斥选项
    plot_surface = True
    if parsed.plot_scatter:
        plot_surface = False
    
    # 如果指定了dim，则根据模式覆盖默认行为，或者保持兼容
    # plot_set 模式隐含 dim=2d，除非强制修改(暂时不支持plot_set画3d)
    
    # 处理xpoints
    xpoints = []
    if parsed.xpoints is not None:
        xpoints = np.array(parsed.xpoints, dtype=float).reshape(2, -1)
    elif parsed.device is not None:
        try:
            device_geom = get_device_geometry(parsed.device, None, None, debug=parsed.debug)
            xpoints = device_geom.xpoints
        except Exception as e:
            if parsed.debug:
                print(f"[Config] Failed to get default xpoints for {parsed.device}: {e}")
            xpoints = None
            
    # Determine mode
    mode = parsed.mode
    if parsed.energy_impact:
        mode = 'energy_impact'
    
    # Handle plot selection flags
    show_left = True
    show_right = True
    if parsed.left_only:
        show_right = False
    if parsed.right_only:
        show_left = False
        
    return ProcessingConfig(
        dim=parsed.dim,
        file_path=parsed.file_path,
        timesteps=parsed.timesteps,
        data_name=parsed.data_name,
        device=parsed.device,
        data_limits=parsed.limits,
        norm_factor=parsed.norm_factor,
        plot_surface=plot_surface,
        plot_overall=parsed.overall,
        log_norm=parsed.log_norm,
        find_max=parsed.find_max,
        output_dir=parsed.output_dir,
        xpoints=xpoints,
        debug=parsed.debug,
        energy_impact=(mode == 'energy_impact'),
        save_convolution=parsed.save_convolution,
        mode=mode,
        show_left_plot=show_left,
        show_right_plot=show_right,
        use_arc_length=parsed.use_arc_length
    )


def create_debug_config() -> ProcessingConfig:
    """
    创建调试用的默认配置。

    Returns
    -------
    config : ProcessingConfig
        调试配置对象
    """
    return ProcessingConfig(
        file_path='/home/ac_desktop/syncfiles/postproc_152_new/boundary_quantities_s05000.dat',
        timesteps=['5000', '6000'],
        data_name='heatF_total',
        device='EXL50U',
        data_limits=[1e5, 3e9],
        norm_factor=4.1006E-07,
        plot_surface=True,
        plot_overall=False,
        log_norm=True,
        find_max=False,
        output_dir=None,
        xpoints= [0.73,  -0.877, 0.73,   0.877],
        debug=True,
        energy_impact=False,
        save_convolution=False
    )
