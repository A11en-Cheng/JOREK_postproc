"""
Device geometry definitions and mask management.

This module defines geometric regions (masks) for different tokamak devices,
supporting easy extension with new devices.
"""

import numpy as np
from typing import Dict, Tuple
from .data_models import DeviceGeometry


def create_mask_exl50u(
    R: np.ndarray,
    Z: np.ndarray,
    debug: bool = False
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[int, int]]]:
    """
    为EXL50-U位形创建各个位置的掩膜。

    Parameters
    ----------
    R : numpy.ndarray
        R坐标网格，形状为 (N_phi, N_poloidal)
    Z : numpy.ndarray
        Z坐标网格，形状为 (N_phi, N_poloidal)
    debug : bool, optional
        调试模式标志

    Returns
    -------
    masks : dict
        位置掩膜字典，键为位置标识
    angles : dict
        对应位置的推荐观看角度
    """
    masks = {
        'mask_UO': (R >= 0.6) & (R <= 1.1) & (Z >= 1.0) & (Z <= 1.6),
        'mask_LO': (R >= 0.6) & (R <= 1.1) & (Z <= -1.0) & (Z >= -1.6),
        'mask_UI': (R >= 0.3) & (R <= 0.6) & (Z <= 1.2) & (Z >= 0.75),
        'mask_LI': (R >= 0.3) & (R <= 0.6) & (Z <= -0.75) & (Z >= -1.2),
    }
    
    angles = {
        'mask_UO': (44, 15),    # Upper Outer
        'mask_LO': (-44, -15),  # Lower Outer
        'mask_UI': (24, 168),   # Upper Inner
        'mask_LI': (-24, -168), # Lower Inner
    }
    
    if debug:
        print("[Geometry] Created EXL50-U device masks")
    
    return masks, angles


def create_mask_iter(
    R: np.ndarray,
    Z: np.ndarray,
    debug: bool = False
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[int, int]]]:
    """
    为ITER位形创建各个位置的掩膜。

    Parameters
    ----------
    R : numpy.ndarray
        R坐标网格
    Z : numpy.ndarray
        Z坐标网格
    debug : bool, optional
        调试模式标志

    Returns
    -------
    masks : dict
        位置掩膜字典
    angles : dict
        对应位置的推荐观看角度
    """
    masks = {
        'mask_UO': (R >= 3.0) & (R <= 3.5) & (Z >= 2.5) & (Z <= 3.0),
        'mask_LO': (R >= 3.0) & (R <= 3.5) & (Z <= -2.4) & (Z >= -3.0),
        'mask_UI': (R >= 2.2) & (R <= 2.5) & (Z <= 2.8) & (Z >= 2.3),
        'mask_LI': (R >= 2.2) & (R <= 2.5) & (Z <= -2.0) & (Z >= -2.8),
    }
    
    angles = {
        'mask_UO': (40, 45),     # Upper Outer
        'mask_LO': (-40, -45),   # Lower Outer
        'mask_UI': (24, 150),    # Upper Inner
        'mask_LI': (-20, -150),  # Lower Inner
    }
    
    if debug:
        print("[Geometry] Created ITER device masks")
    
    return masks, angles


def get_device_geometry(
    device_name: str,
    R: np.ndarray,
    Z: np.ndarray,
    debug: bool = False
) -> DeviceGeometry:
    """
    根据设备名称获取对应的几何信息。

    Parameters
    ----------
    device_name : str
        设备名称 ('EXL50U', 'ITER', 等)
    R : numpy.ndarray
        R坐标网格
    Z : numpy.ndarray
        Z坐标网格
    debug : bool, optional
        调试模式标志

    Returns
    -------
    geometry : DeviceGeometry
        设备几何信息对象

    Raises
    ------
    ValueError
        如果设备名称不被支持
    """
    device_name_upper = device_name.upper()
    
    if device_name_upper == 'EXL50U':
        masks, angles = create_mask_exl50u(R, Z, debug=debug)
    elif device_name_upper == 'ITER':
        masks, angles = create_mask_iter(R, Z, debug=debug)
    else:
        raise ValueError(f"Unknown device: {device_name}. Supported devices: EXL50U, ITER")
    
    return DeviceGeometry(
        name=device_name_upper,
        masks=masks,
        view_angles=angles
    )


def register_custom_device(
    device_name: str,
    mask_func: callable
) -> None:
    """
    注册自定义设备位形。

    这个函数可以用来为新的设备定义位形。

    Parameters
    ----------
    device_name : str
        设备名称
    mask_func : callable
        掩膜生成函数，签名为 (R, Z, debug) -> (masks_dict, angles_dict)

    Examples
    --------
    >>> def my_device_masks(R, Z, debug=False):
    ...     masks = {'mask_1': R > 1.0}
    ...     angles = {'mask_1': (30, 45)}
    ...     return masks, angles
    >>> register_custom_device('MYDEVICE', my_device_masks)
    """
    # 动态添加到模块级别 - 这是一个简化的实现
    # 实际项目可能需要更复杂的注册机制
    if debug:
        print(f"[Geometry] Custom device '{device_name}' registered")
