"""
Test geometry module
"""
import pytest
import numpy as np
from jorek_postproc.geometry import EXL50UDevice, ITERDevice, get_device_instance

def test_exl50u_device():
    device = EXL50UDevice()
    assert device.name == "EXL50U"
    assert device.default_xpoints is not None
    
    # Create dummy coordinate arrays
    R = np.array([0.8, 0.8, 0.4, 0.4])
    Z = np.array([1.2, -1.2, 0.9, -0.9])
    
    masks, angles = device.get_masks_and_angles(R, Z)
    
    assert 'mask_UO' in masks
    assert 'mask_LO' in masks
    assert 'mask_UI' in masks
    assert 'mask_LI' in masks
    
    assert angles['mask_UO'] == (44, 15)

def test_iter_device():
    device = ITERDevice()
    assert device.name == "ITER"
    
    R = np.array([3.2])
    Z = np.array([2.7])
    
    masks, angles = device.get_masks_and_angles(R, Z)
    assert 'mask_UO' in masks

def test_get_device_instance():
    dev = get_device_instance("EXL50U")
    assert isinstance(dev, EXL50UDevice)
    
    dev2 = get_device_instance("ITER")
    assert isinstance(dev2, ITERDevice)
    
    # helper handles case insensitivity
    dev3 = get_device_instance("exl50u")
    assert isinstance(dev3, EXL50UDevice)
    
    with pytest.raises(ValueError):
        get_device_instance("UNKNOWN")
