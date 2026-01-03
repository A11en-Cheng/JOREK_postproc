"""
版本信息模块

定义包版本，所有版本信息应在此处更新。
"""

__version__ = "0.1.0"
__author__ = "Allen Cheng"
__email__ = "Allencheng@buaa.edu.cn"
__license__ = "MIT"

# 语义化版本分量
VERSION_MAJOR = 0
VERSION_MINOR = 1
VERSION_PATCH = 0

def get_version_info() -> dict:
    """
    获取版本信息字典
    
    Returns
    -------
    dict
        包含所有版本信息的字典
    """
    return {
        'version': __version__,
        'major': VERSION_MAJOR,
        'minor': VERSION_MINOR,
        'patch': VERSION_PATCH,
        'author': __author__,
        'email': __email__,
        'license': __license__,
    }


def check_version(required_version: str) -> bool:
    """
    检查当前版本是否满足要求
    
    Parameters
    ----------
    required_version : str
        要求的最低版本，格式为 'X.Y.Z'
        
    Returns
    -------
    bool
        如果当前版本 >= 要求版本返回True
        
    Examples
    --------
    >>> check_version('0.1.0')
    True
    >>> check_version('1.0.0')
    False
    """
    from packaging import version
    return version.parse(__version__) >= version.parse(required_version)
