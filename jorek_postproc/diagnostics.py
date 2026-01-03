"""
诊断和验证模块

提供包的诊断、健康检查和验证功能。
"""

import sys
import platform
from typing import Dict, List, Tuple

from . import __version__
from .logging import get_logger

logger = get_logger(__name__)


def check_environment() -> Dict[str, str]:
    """
    检查运行环境信息
    
    Returns
    -------
    dict
        包含环境信息的字典
        
    Examples
    --------
    >>> env = check_environment()
    >>> print(env['python_version'])
    3.9.13
    """
    try:
        import numpy as np
    except ImportError:
        np = None
    
    try:
        import matplotlib
    except ImportError:
        matplotlib = None
    
    try:
        import scipy
    except ImportError:
        scipy = None
    
    env_info = {
        'python_version': platform.python_version(),
        'python_implementation': platform.python_implementation(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'jorek_postproc_version': __version__,
        'numpy_version': getattr(np, '__version__', 'Not installed'),
        'matplotlib_version': getattr(matplotlib, '__version__', 'Not installed'),
        'scipy_version': getattr(scipy, '__version__', 'Not installed'),
    }
    
    return env_info


def print_environment():
    """打印环境信息"""
    env = check_environment()
    print("\n" + "="*70)
    print("jorek_postproc 环境信息")
    print("="*70)
    for key, value in env.items():
        print(f"{key:.<30} {value}")
    print("="*70 + "\n")


def validate_installation() -> Tuple[bool, List[str]]:
    """
    验证包安装的完整性
    
    Returns
    -------
    is_valid : bool
        安装是否有效
    issues : list
        发现的问题列表
        
    Examples
    --------
    >>> valid, issues = validate_installation()
    >>> if not valid:
    ...     for issue in issues:
    ...         print(f"问题: {issue}")
    """
    issues = []
    
    # 检查核心依赖
    required_packages = {
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'scipy': 'SciPy',
    }
    
    for module_name, display_name in required_packages.items():
        try:
            __import__(module_name)
            logger.debug(f"✓ {display_name} 已安装")
        except ImportError:
            issues.append(f"{display_name} 未安装或导入失败")
    
    # 检查核心模块
    core_modules = [
        'jorek_postproc.data_models',
        'jorek_postproc.io',
        'jorek_postproc.reshaping',
        'jorek_postproc.processing',
        'jorek_postproc.geometry',
        'jorek_postproc.plotting',
        'jorek_postproc.config',
    ]
    
    for module_name in core_modules:
        try:
            __import__(module_name)
            logger.debug(f"✓ {module_name} 可导入")
        except ImportError as e:
            issues.append(f"模块 {module_name} 导入失败: {e}")
    
    return len(issues) == 0, issues


def run_diagnostics(verbose: bool = False) -> bool:
    """
    运行完整的诊断检查
    
    Parameters
    ----------
    verbose : bool
        是否显示详细信息
        
    Returns
    -------
    bool
        诊断是否通过
        
    Examples
    --------
    >>> passed = run_diagnostics(verbose=True)
    >>> if not passed:
    ...     print("诊断失败！")
    """
    print("\n" + "="*70)
    print("运行 jorek_postproc 诊断检查")
    print("="*70 + "\n")
    
    # 环境检查
    if verbose:
        print("[1/3] 检查环境...")
        print_environment()
    
    # 安装验证
    print("[2/3] 验证安装完整性...")
    is_valid, issues = validate_installation()
    
    if is_valid:
        print("✓ 所有依赖已安装")
        print("✓ 所有模块可导入")
    else:
        print("✗ 发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
    
    # 功能测试
    print("\n[3/3] 测试基本功能...")
    try:
        from jorek_postproc import (
            BoundaryQuantitiesData,
            PlottingConfig,
            get_device_geometry,
        )
        import numpy as np
        
        # 测试数据模型
        data = BoundaryQuantitiesData(
            R=np.array([[1, 2], [3, 4]]),
            Z=np.array([[1, 2], [3, 4]]),
            phi=np.array([[1, 2], [3, 4]]),
            data=np.array([[1, 2], [3, 4]]),
            data_name='test'
        )
        assert data.is_2d_grid()
        
        # 测试配置
        config = PlottingConfig(log_norm=True)
        assert config.log_norm is True
        
        print("✓ 数据模型功能正常")
        print("✓ 配置系统功能正常")
        
    except Exception as e:
        print(f"✗ 功能测试失败: {e}")
        return False
    
    print("\n" + "="*70)
    print("✓ 诊断检查通过！")
    print("="*70 + "\n")
    
    return is_valid and True


def get_dependency_versions() -> Dict[str, str]:
    """
    获取所有依赖的版本信息
    
    Returns
    -------
    dict
        依赖包版本信息
    """
    versions = {}
    
    packages = ['numpy', 'matplotlib', 'scipy', 'setuptools', 'wheel']
    
    for package_name in packages:
        try:
            module = __import__(package_name)
            version = getattr(module, '__version__', 'Unknown')
            versions[package_name] = version
        except ImportError:
            versions[package_name] = 'Not installed'
    
    return versions


def system_info() -> str:
    """
    获取完整的系统信息字符串
    
    Returns
    -------
    str
        格式化的系统信息
    """
    env = check_environment()
    deps = get_dependency_versions()
    
    info = "系统信息:\n"
    info += "="*50 + "\n"
    info += f"Python: {env['python_version']} ({env['python_implementation']})\n"
    info += f"平台: {env['platform']}\n"
    info += f"处理器: {env['processor']}\n"
    info += "\njork_postproc:\n"
    info += f"版本: {env['jorek_postproc_version']}\n"
    info += "\n依赖包:\n"
    for package, version in deps.items():
        info += f"  {package}: {version}\n"
    
    return info
