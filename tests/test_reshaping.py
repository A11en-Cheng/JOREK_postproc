"""
测试数据重塑/重构模块
"""

import pytest
import numpy as np
from jorek_postproc.reshaping import reshape_to_grid
from jorek_postproc.data_models import BoundaryQuantitiesData


class TestReshapeModule:
    """测试数据重塑模块"""
    
    def test_reshape_to_grid_basic(self, sample_1d_data):
        """测试基本重塑功能"""
        # Construct raw block from sample data
        # sample_1d_data: R, Z, phi, data
        N = len(sample_1d_data.R)
        block = np.column_stack([
            sample_1d_data.R,
            sample_1d_data.Z,
            sample_1d_data.phi,
            sample_1d_data.data
        ])
        
        col_names = ['R', 'Z', 'phi', 'val']
        names = ('R', 'Z', 'phi', 'val')
        
        grid_data = reshape_to_grid(block, col_names, names)
        
        assert isinstance(grid_data, BoundaryQuantitiesData)
        assert grid_data.is_2d_grid()

    def test_reshape_real_usage(self):
        """测试真实调用方式"""
        # Create raw block
        N = 100
        block = np.zeros((N, 4))
        block[:, 0] = np.random.rand(N) + 2.0 # R
        block[:, 1] = np.random.rand(N) # Z
        block[:, 2] = np.concatenate([np.zeros(50), np.ones(50)]) # phi
        block[:, 3] = np.random.rand(N) # val
        
        col_names = ['R', 'Z', 'phi', 'val']
        names = ('R', 'Z', 'phi', 'val')
        
        grid_data = reshape_to_grid(block, col_names, names)
        
        assert isinstance(grid_data, BoundaryQuantitiesData)
        assert grid_data.grid_shape[0] == 2 # 2 phi slices
        
    def test_reshape_arc_length(self):
        """测试弧长计算"""
        # Create points on a circle to satisfy centroid sorting logic
        theta = np.linspace(0, 2*np.pi - 0.1, 10)
        R = 3.0 + np.cos(theta)
        Z = np.sin(theta)
        phi = np.zeros_like(theta)
        val = np.ones_like(theta)
        
        block = np.column_stack([R, Z, phi, val])
        col_names = ['R', 'Z', 'phi', 'val']
        names = ('R', 'Z', 'phi', 'val')
        
        grid_data = reshape_to_grid(block, col_names, names)
        
        assert grid_data.arc_length is not None
        # Arc length should increase roughly monotonically (sorting might reorder)
        # Check monotonicity of arc_length[0]
        al = grid_data.arc_length[0, :]
        assert np.all(np.diff(al) >= 0)
        assert al[-1] > 0
