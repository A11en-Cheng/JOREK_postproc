"""
Configuration and command-line argument parsing for boundary quantities processing.
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional


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
        X点坐标 [x1, z1, x2, z2]，用于双X点撕裂模
    debug : bool
        调试模式标志
    """
    file_path: str
    timesteps: List[str]
    iplane: int = 1080
    data_name: str = 'heatF_tot_cd'
    device: str = 'EXL50U'
    data_limits: Optional[List[float]] = None
    norm_factor: Optional[float] = None
    plot_surface: bool = True
    plot_overall: bool = False
    log_norm: bool = False
    find_max: bool = True
    output_dir: Optional[str] = None
    xpoints: Optional[List[float]] = None
    debug: bool = False


def create_parser() -> argparse.ArgumentParser:
    """
    创建命令行参数解析器。

    Returns
    -------
    parser : argparse.ArgumentParser
        配置好的参数解析器
    """
    parser = argparse.ArgumentParser(
        description="JOREK边界量三维可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例使用:
  python -m jorek_postproc.cli -f boundary_quantities_s04200.dat -t 4200 --iplane 1080 -n heatF_tot_cd
  python -m jorek_postproc.cli -f boundary_quantities_s04200.dat -t 4200 4650 5000 --device ITER --log-norm
  python -m jorek_postproc.cli -f boundary_quantities_s04200.dat -t 4200 --xpoints 0.73 0.877 0.75 -0.8
        """
    )
    
    # 必选参数
    parser.add_argument(
        "-f", "--file",
        required=True,
        help="边界量文件路径"
    )
    
    parser.add_argument(
        "-t", "--timesteps",
        nargs='+',
        required=True,
        type=str,
        help="要处理的时间步列表 (e.g., 4200 4650 5000)"
    )
    
    # 可选参数
    parser.add_argument(
        "--iplane",
        type=int,
        default=1080,
        help="环向平面数，默认1080"
    )
    
    parser.add_argument(
        "-n", "--name",
        default='heatF_tot_cd',
        help="物理量列名，默认heatF_tot_cd"
    )
    
    parser.add_argument(
        "-d", "--device",
        default='EXL50U',
        choices=['EXL50U', 'ITER'],
        help="设备名称，默认EXL50U"
    )
    
    parser.add_argument(
        "-lim", "--limits",
        nargs=2,
        type=float,
        metavar=('MIN', 'MAX'),
        default=None,
        help="数据显示范围 [min max]"
    )
    
    parser.add_argument(
        "-nf", "--norm-factor",
        type=float,
        default=None,
        help="归一化因子 (e.g., 单位转换系数)"
    )
    
    parser.add_argument(
        "-sf", "--plot-surface",
        action='store_true',
        default=True,
        help="绘制3D表面图（默认），否则绘制散点图"
    )
    
    parser.add_argument(
        "--plot-scatter",
        action='store_true',
        help="绘制3D散点图而不是表面图"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="输出目录，默认为当前目录"
    )
    
    parser.add_argument(
        "--log-norm",
        action='store_true',
        default=False,
        help="使用对数色图"
    )
    
    parser.add_argument(
        "-fm", "--find-max",
        action='store_true',
        default=True,
        help="在图上标记最大值位置"
    )
    
    parser.add_argument(
        "--overall",
        action='store_true',
        default=False,
        help="绘制整体视图，否则绘制位形特定的视图"
    )
    
    parser.add_argument(
        "-xpt", "--xpoints",
        nargs=4,
        type=float,
        default=None,
        metavar=('X1', 'Z1', 'X2', 'Z2'),
        help="X点坐标，用于双X点撕裂模 (x1 z1 x2 z2)"
    )
    
    parser.add_argument(
        "--debug",
        action='store_true',
        default=False,
        help="启用调试模式"
    )
    
    return parser


def parse_args(args=None) -> ProcessingConfig:
    """
    解析命令行参数并返回配置对象。

    Parameters
    ----------
    args : list of str, optional
        命令行参数，如果为None使用sys.argv

    Returns
    -------
    config : ProcessingConfig
        处理配置对象
    """
    parser = create_parser()
    parsed = parser.parse_args(args)
    
    # 处理互斥选项
    plot_surface = not parsed.plot_scatter
    
    # 处理xpoints
    xpoints = None
    if parsed.xpoints is not None:
        import numpy as np
        xpoints = np.array(parsed.xpoints, dtype=float).reshape(2, -1)
    
    return ProcessingConfig(
        file_path=parsed.file,
        timesteps=parsed.timesteps,
        iplane=parsed.iplane,
        data_name=parsed.name,
        device=parsed.device,
        data_limits=parsed.limits,
        norm_factor=parsed.norm_factor,
        plot_surface=plot_surface,
        plot_overall=parsed.overall,
        log_norm=parsed.log_norm,
        find_max=parsed.find_max,
        output_dir=parsed.output_dir,
        xpoints=xpoints,
        debug=parsed.debug
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
        file_path='/home/ac_desktop/syncfiles/postproc_145/boundary_quantities_s04200.dat',
        timesteps=['4200'],
        iplane=1080,
        data_name='heatF_tot_cd',
        device='EXL50U',
        data_limits=[1e5, 3e8],
        norm_factor=4.1006E-07,
        plot_surface=True,
        plot_overall=False,
        log_norm=True,
        find_max=False,
        output_dir=None,
        xpoints=None,
        debug=True
    )
