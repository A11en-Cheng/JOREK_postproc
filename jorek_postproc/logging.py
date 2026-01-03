"""
日志管理模块

提供统一的日志配置和管理功能。
"""

import logging
import sys
from typing import Optional


# 全局日志记录器
_logger: Optional[logging.Logger] = None


def setup_logging(
    name: str = "jorek_postproc",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    verbose: bool = False
) -> logging.Logger:
    """
    配置包的日志系统
    
    Parameters
    ----------
    name : str
        日志记录器名称，默认"jorek_postproc"
    level : int
        日志级别，默认INFO
    log_file : str, optional
        日志文件路径，如果为None则仅输出到控制台
    verbose : bool
        是否启用详细日志，默认False
        
    Returns
    -------
    logger : logging.Logger
        配置好的日志记录器
        
    Examples
    --------
    >>> logger = setup_logging(verbose=True)
    >>> logger.debug("Debug message")
    [jorek_postproc] DEBUG: Debug message
    """
    global _logger
    
    logger = logging.getLogger(name)
    
    # 避免重复添加handler
    if logger.handlers:
        return logger
    
    # 设置日志级别
    if verbose:
        level = logging.DEBUG
    
    logger.setLevel(level)
    
    # 创建格式化器
    if verbose:
        formatter = logging.Formatter(
            '[%(name)s] %(levelname)s [%(filename)s:%(lineno)d]: %(message)s'
        )
    else:
        formatter = logging.Formatter('[%(name)s] %(levelname)s: %(message)s')
    
    # 添加控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 可选：添加文件handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _logger = logger
    return logger


def get_logger(name: str = "jorek_postproc") -> logging.Logger:
    """
    获取日志记录器
    
    如果未配置，则使用默认配置。
    
    Parameters
    ----------
    name : str
        日志记录器名称
        
    Returns
    -------
    logger : logging.Logger
        日志记录器
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logging(name)
    return logger


def disable_logging():
    """禁用日志输出"""
    logging.disable(logging.CRITICAL)


def enable_logging():
    """启用日志输出"""
    logging.disable(logging.NOTSET)
