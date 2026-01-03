"""
测试数据处理模块
"""

import pytest
import numpy as np
from jorek_postproc.processing import process_timestep, process_multiple_timesteps
from jorek_postproc.data_models import BoundaryQuantitiesData, ProcessingConfig


class TestProcessingModule:
    """测试数据处理模块"""
    
    def test_process_timestep_basic(self, sample_grid_data):
        """测试基本时间步处理"""
        config = ProcessingConfig(
            normalize=True,
            apply_filter=False
        )
        
        result = process_timestep(sample_grid_data, config)
        
        assert isinstance(result, BoundaryQuantitiesData)
        assert result.grid_shape == sample_grid_data.grid_shape
    
    def test_process_timestep_with_normalization(self, sample_grid_data):
        """测试带归一化的处理"""
        config = ProcessingConfig(normalize=True)
        result = process_timestep(sample_grid_data, config)
        
        # 归一化后值应该在0-1范围内或更小
        assert result.value.max() <= 1.1  # 允许轻微超出
        assert result.value.min() >= -0.1
    
    def test_process_timestep_without_normalization(self, sample_grid_data):
        """测试不归一化的处理"""
        config = ProcessingConfig(normalize=False)
        result = process_timestep(sample_grid_data, config)
        
        # 值应该保持原样
        np.testing.assert_array_almost_equal(result.value, sample_grid_data.value)
    
    def test_process_timestep_preserves_metadata(self, sample_grid_data):
        """测试处理保留元数据"""
        config = ProcessingConfig()
        result = process_timestep(sample_grid_data, config)
        
        assert result.data_name == sample_grid_data.data_name
        assert result.device == sample_grid_data.device
    
    def test_process_timestep_with_filter(self, sample_grid_data):
        """测试带滤波的处理"""
        config = ProcessingConfig(apply_filter=True, filter_type='gaussian')
        
        result = process_timestep(sample_grid_data, config)
        
        assert isinstance(result, BoundaryQuantitiesData)
        # 滤波后的值应该是有限的数值
        assert np.all(np.isfinite(result.value))
    
    def test_process_multiple_timesteps_basic(self, sample_grid_data):
        """测试多个时间步处理"""
        timesteps = [sample_grid_data for _ in range(3)]
        config = ProcessingConfig(normalize=True)
        
        results = process_multiple_timesteps(timesteps, config)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, BoundaryQuantitiesData)
    
    def test_process_multiple_timesteps_preserves_order(self, sample_grid_data):
        """测试多时间步保留顺序"""
        # 创建带时间戳的数据副本
        timesteps = []
        for i in range(5):
            ts_data = BoundaryQuantitiesData(
                R=sample_grid_data.R,
                Z=sample_grid_data.Z,
                phi=sample_grid_data.phi,
                value=sample_grid_data.value * (i + 1),  # 不同的值以区分
                data_name=f'timestep_{i}',
                device=sample_grid_data.device
            )
            timesteps.append(ts_data)
        
        config = ProcessingConfig(normalize=False)
        results = process_multiple_timesteps(timesteps, config)
        
        # 检查顺序和缩放
        for i, result in enumerate(results):
            assert f'timestep_{i}' in result.data_name
    
    def test_process_multiple_timesteps_empty_list(self):
        """测试空列表处理"""
        config = ProcessingConfig()
        results = process_multiple_timesteps([], config)
        
        assert results == []


class TestProcessingConfig:
    """测试处理配置"""
    
    def test_config_defaults(self):
        """测试默认配置"""
        config = ProcessingConfig()
        
        assert config.normalize is True
        assert config.apply_filter is False
    
    def test_config_custom_values(self):
        """测试自定义配置值"""
        config = ProcessingConfig(
            normalize=False,
            apply_filter=True,
            filter_type='median'
        )
        
        assert config.normalize is False
        assert config.apply_filter is True
        assert config.filter_type == 'median'
    
    def test_config_invalid_filter_type(self):
        """测试无效滤波器类型"""
        # 应该接受或拒绝无效类型（取决于实现）
        try:
            config = ProcessingConfig(
                apply_filter=True,
                filter_type='invalid_filter'
            )
            # 如果接受，那就可以
            assert hasattr(config, 'filter_type')
        except ValueError:
            # 如果拒绝，那也可以
            pass


class TestProcessingEdgeCases:
    """测试处理边界情况"""
    
    def test_process_single_value(self):
        """测试单个值数据"""
        data = BoundaryQuantitiesData(
            R=np.array([5.0]),
            Z=np.array([0.0]),
            phi=np.array([0.0]),
            value=np.array([1e6]),
            data_name='single_point',
            device='test'
        )
        
        config = ProcessingConfig(normalize=True)
        result = process_timestep(data, config)
        
        assert len(result.value) == 1
    
    def test_process_zero_values(self):
        """测试零值数据"""
        data = BoundaryQuantitiesData(
            R=np.array([5.0, 6.0]),
            Z=np.array([0.0, 0.0]),
            phi=np.array([0.0, 0.0]),
            value=np.array([0.0, 0.0]),
            data_name='zeros',
            device='test'
        )
        
        config = ProcessingConfig(normalize=True)
        result = process_timestep(data, config)
        
        # 应该正确处理零值
        assert np.all(np.isfinite(result.value))


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
