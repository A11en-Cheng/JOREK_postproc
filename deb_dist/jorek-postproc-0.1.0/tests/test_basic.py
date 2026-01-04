"""
测试基本功能的单元测试
"""

import pytest
from jorek_postproc import (
    BoundaryQuantitiesData,
    PlottingConfig,
    get_device_geometry,
)


class TestDataModels:
    """测试数据模型"""
    
    def test_boundary_quantities_data_creation(self, sample_grid_data):
        """测试BoundaryQuantitiesData创建"""
        assert sample_grid_data.data_name == 'test_data'
        assert sample_grid_data.grid_shape == (10, 20)
        assert sample_grid_data.is_2d_grid()
    
    def test_boundary_quantities_data_1d(self, sample_1d_data):
        """测试1D数据"""
        assert len(sample_1d_data.R) == 1000
        assert not sample_1d_data.is_2d_grid()
    
    def test_plotting_config_defaults(self):
        """测试绘图配置默认值"""
        config = PlottingConfig()
        assert config.log_norm is False
        assert config.cmap == 'viridis'
        assert config.dpi == 300
        assert config.find_max is True
    
    def test_plotting_config_custom(self, plotting_config):
        """测试自定义绘图配置"""
        assert plotting_config.log_norm is True
        assert plotting_config.dpi == 150
        assert plotting_config.find_max is False


class TestGeometry:
    """测试几何/位形模块"""
    
    def test_get_device_geometry_exl50u(self, sample_grid_data):
        """测试EXL50U位形获取"""
        device = get_device_geometry('EXL50U', sample_grid_data.R, sample_grid_data.Z)
        
        assert device.name == 'EXL50U'
        assert 'mask_UO' in device.masks
        assert 'mask_LO' in device.masks
        assert 'mask_UI' in device.masks
        assert 'mask_LI' in device.masks
    
    def test_get_device_geometry_iter(self, sample_grid_data):
        """测试ITER位形获取"""
        device = get_device_geometry('ITER', sample_grid_data.R, sample_grid_data.Z)
        
        assert device.name == 'ITER'
        assert len(device.masks) == 4
        assert len(device.view_angles) == 4
    
    def test_get_device_geometry_invalid(self, sample_grid_data):
        """测试无效装置名称"""
        with pytest.raises(ValueError):
            get_device_geometry('INVALID', sample_grid_data.R, sample_grid_data.Z)


class TestVersionInfo:
    """测试版本信息"""
    
    def test_version_string(self):
        """测试版本字符串"""
        from jorek_postproc import __version__
        assert isinstance(__version__, str)
        assert len(__version__.split('.')) == 3  # 语义化版本
    
    def test_author_info(self):
        """测试作者信息"""
        from jorek_postproc import __author__, __email__
        assert __author__ == 'Allen Cheng'
        assert __email__ == 'Allencheng@buaa.edu.cn'


class TestDiagnostics:
    """测试诊断功能"""
    
    def test_check_environment(self):
        """测试环境检查"""
        from jorek_postproc import check_environment
        env = check_environment()
        
        assert 'python_version' in env
        assert 'jorek_postproc_version' in env
        assert 'numpy_version' in env
    
    def test_validate_installation(self):
        """测试安装验证"""
        from jorek_postproc.diagnostics import validate_installation
        is_valid, issues = validate_installation()
        
        assert isinstance(is_valid, bool)
        assert isinstance(issues, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
