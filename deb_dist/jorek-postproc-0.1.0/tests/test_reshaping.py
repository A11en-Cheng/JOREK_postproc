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
        grid_data = reshape_to_grid(
            sample_1d_data,
            grid_shape=(20, 30),
            interpolation='linear'
        )
        
        assert isinstance(grid_data, BoundaryQuantitiesData)
        assert grid_data.is_2d_grid()
        assert grid_data.grid_shape == (20, 30)
    
    def test_reshape_preserves_data_name(self, sample_1d_data):
        """测试重塑后保留数据名称"""
        original_name = sample_1d_data.data_name
        
        grid_data = reshape_to_grid(
            sample_1d_data,
            grid_shape=(15, 25),
            interpolation='linear'
        )
        
        assert grid_data.data_name == original_name
    
    def test_reshape_different_shapes(self, sample_1d_data):
        """测试不同网格形状"""
        shapes = [(10, 10), (20, 30), (50, 40), (100, 100)]
        
        for shape in shapes:
            grid_data = reshape_to_grid(
                sample_1d_data,
                grid_shape=shape,
                interpolation='linear'
            )
            
            assert grid_data.grid_shape == shape
            assert len(grid_data.value) == shape[0] * shape[1]
    
    def test_reshape_interpolation_methods(self, sample_1d_data):
        """测试不同插值方法"""
        methods = ['linear', 'nearest', 'cubic']
        
        results = {}
        for method in methods:
            grid_data = reshape_to_grid(
                sample_1d_data,
                grid_shape=(15, 20),
                interpolation=method
            )
            results[method] = grid_data
        
        # 检查所有方法都返回有效结果
        for method, data in results.items():
            assert data.is_2d_grid()
            assert not np.any(np.isnan(data.value)), f"Method {method} produced NaN values"
    
    def test_reshape_from_2d_fails(self, sample_grid_data):
        """测试从2D数据重塑应该失败或不同处理"""
        # 已经是2D数据，重塑应该处理得当
        result = reshape_to_grid(
            sample_grid_data,
            grid_shape=(10, 20),
            interpolation='linear'
        )
        
        assert result.is_2d_grid()
    
    def test_reshape_value_consistency(self, sample_1d_data):
        """测试重塑后值的一致性"""
        original_min = sample_1d_data.value.min()
        original_max = sample_1d_data.value.max()
        
        grid_data = reshape_to_grid(
            sample_1d_data,
            grid_shape=(20, 30),
            interpolation='linear'
        )
        
        # 重塑后的min/max不应显著超出原始范围
        assert grid_data.value.min() >= original_min * 0.9
        assert grid_data.value.max() <= original_max * 1.1


class TestGridProperties:
    """测试网格属性"""
    
    def test_grid_shape_attribute(self, sample_grid_data):
        """测试网格形状属性"""
        assert hasattr(sample_grid_data, 'grid_shape')
        assert sample_grid_data.grid_shape == (10, 20)
    
    def test_grid_is_2d_check(self, sample_grid_data, sample_1d_data):
        """测试2D网格检查"""
        assert sample_grid_data.is_2d_grid() is True
        assert sample_1d_data.is_2d_grid() is False
    
    def test_grid_dimensions_match_data(self, sample_grid_data):
        """测试网格尺寸与数据匹配"""
        expected_size = sample_grid_data.grid_shape[0] * sample_grid_data.grid_shape[1]
        actual_size = len(sample_grid_data.value)
        
        assert actual_size == expected_size


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
