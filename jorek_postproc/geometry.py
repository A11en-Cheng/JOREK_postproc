"""
Device geometry definitions and mask management.

This module defines geometric regions (masks) for different tokamak devices,
supporting easy extension with new devices.
"""

import numpy as np
from typing import Dict, Tuple, Callable, Optional, Type
from abc import ABC, abstractmethod
from .data_models import DeviceGeometry

class BaseDevice(ABC):
    """
    Abstract base class for device definitions.
    """
    name: str
    default_xpoints: Optional[np.ndarray] = None

    @abstractmethod
    def get_masks_and_angles(
        self, 
        R: np.ndarray, 
        Z: np.ndarray, 
        debug: bool = False
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[int, int]]]:
        """
        Generate masks and view angles for the device.
        """
        pass

class EXL50UDevice(BaseDevice):
    name = "EXL50U"
    # Default X-points for EXL50-U: [R1, Z1], [R2, Z2]
    default_xpoints = np.array([
        [0.72, 0.877], 
        [0.72, -0.877]
    ])

    def get_masks_and_angles(self, R, Z, debug=False):
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

class ITERDevice(BaseDevice):
    name = "ITER"
    default_xpoints = None # ITER might not need default xpoints or they are different

    def get_masks_and_angles(self, R, Z, debug=False):
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

# Registry for devices
_DEVICE_REGISTRY: Dict[str, BaseDevice] = {
    'EXL50U': EXL50UDevice(),
    'ITER': ITERDevice(),
}

def get_device_instance(device_name: str) -> BaseDevice:
    """
    Get the device instance by name.
    """
    device_name_upper = device_name.upper()
    if device_name_upper in _DEVICE_REGISTRY:
        return _DEVICE_REGISTRY[device_name_upper]
    else:
        raise ValueError(f"Unknown device: {device_name}. Supported: {list(_DEVICE_REGISTRY.keys())}")

def get_device_geometry(
    device_name: str,
    R: np.ndarray,
    Z: np.ndarray,
    xpoints: Optional[np.ndarray] = None,
    debug: bool = False
) -> DeviceGeometry:
    """
    Get DeviceGeometry data object (backward compatibility wrapper).
    """
    device = get_device_instance(device_name)
    
    # Use provided xpoints or default from device
    final_xpoints = xpoints if xpoints is not None else device.default_xpoints
    
    # Generate masks if R and Z are provided
    if R is not None and Z is not None:
        masks, angles = device.get_masks_and_angles(R, Z, debug)
    else:
        masks, angles = {}, {}
        
    return DeviceGeometry(
        name=device.name,
        masks=masks,
        view_angles=angles,
        xpoints=final_xpoints
    )

def register_custom_device(
    device_name: str,
    device_class: Type[BaseDevice]
) -> None:
    """
    Register a custom device class.
    """
    _DEVICE_REGISTRY[device_name.upper()] = device_class()
    print(f"[Geometry] Custom device '{device_name}' registered")
