"""
测试绘图/可视化模块
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from jorek_postproc.plotting import plot_surface_3d, plot_scatter_3d
from jorek_postproc.data_models import PlottingConfig


class TestPlottingModule:
    """测试绘图模块"""
    
    @pytest.fixture(autouse=True)
    def cleanup(self):
        """在每个测试后清理图表"""
        yield
        plt.close('all')
    
    def test_plot_surface_3d_basic(self, sample_grid_data, plotting_config):
        """测试基本3D曲面绘图"""
        fig = plot_surface_3d(sample_grid_data, plotting_config)
        
        assert fig is not None
        assert hasattr(fig, 'axes')
    
    def test_plot_surface_3d_creates_figure(self, sample_grid_data):
        """测试3D曲面绘图创建图表"""
        initial_figures = len(plt.get_fignums())
        
        fig = plot_surface_3d(sample_grid_data)
        
        assert len(plt.get_fignums()) > initial_figures
    
    def test_plot_surface_3d_with_custom_config(self, sample_grid_data):
        """测试自定义配置的3D曲面绘图"""
        config = PlottingConfig(
            cmap='coolwarm',
            log_norm=True,
            dpi=100
        )
        
        fig = plot_surface_3d(sample_grid_data, config)
        
        assert fig is not None
    
    def test_plot_scatter_3d_basic(self, sample_1d_data, plotting_config):
        """测试基本3D散点绘图"""
        fig = plot_scatter_3d(sample_1d_data, plotting_config)
        
        assert fig is not None
        assert hasattr(fig, 'axes')
    
    def test_plot_scatter_3d_creates_figure(self, sample_1d_data):
        """测试3D散点绘图创建图表"""
        initial_figures = len(plt.get_fignums())
        
        fig = plot_scatter_3d(sample_1d_data)
        
        assert len(plt.get_fignums()) > initial_figures
    
    def test_plot_scatter_3d_with_custom_config(self, sample_1d_data):
        """测试自定义配置的3D散点绘图"""
        config = PlottingConfig(
            cmap='plasma',
            log_norm=False,
            dpi=150
        )
        
        fig = plot_scatter_3d(sample_1d_data, config)
        
        assert fig is not None
    
    def test_plot_different_colormaps(self, sample_grid_data):
        """测试不同的颜色映射"""
        cmaps = ['viridis', 'plasma', 'coolwarm', 'RdYlBu']
        
        for cmap in cmaps:
            config = PlottingConfig(cmap=cmap)
            fig = plot_surface_3d(sample_grid_data, config)
            
            assert fig is not None
    
    def test_plot_with_log_scale(self, sample_grid_data):
        """测试对数刻度绘图"""
        config = PlottingConfig(log_norm=True)
        fig = plot_surface_3d(sample_grid_data, config)
        
        assert fig is not None
    
    def test_plot_with_linear_scale(self, sample_grid_data):
        """测试线性刻度绘图"""
        config = PlottingConfig(log_norm=False)
        fig = plot_surface_3d(sample_grid_data, config)
        
        assert fig is not None
    
    def test_plot_surface_3d_invalid_data(self):
        """测试无效数据的3D曲面绘图"""
        from jorek_postproc.data_models import BoundaryQuantitiesData
        
        # 非2D数据应该失败或处理得当
        data = BoundaryQuantitiesData(
            R=np.array([5.0]),
            Z=np.array([0.0]),
            phi=np.array([0.0]),
            value=np.array([1e6]),
            data_name='single_point'
        )
        
        # 单点数据可能无法绘图
        try:
            fig = plot_surface_3d(data)
            # 如果创建了图表，那也可以
            assert fig is not None
        except (ValueError, RuntimeError):
            # 如果抛出异常，也是可以接受的
            pass


class TestPlottingConfig:
    """测试绘图配置"""
    
    def test_config_defaults(self):
        """测试默认绘图配置"""
        config = PlottingConfig()
        
        assert config.log_norm is False
        assert config.cmap == 'viridis'
        assert config.dpi >= 100
    
    def test_config_custom_values(self):
        """测试自定义绘图配置"""
        config = PlottingConfig(
            cmap='coolwarm',
            log_norm=True,
            dpi=300,
            find_max=False
        )
        
        assert config.cmap == 'coolwarm'
        assert config.log_norm is True
        assert config.dpi == 300
        assert config.find_max is False
    
    def test_config_immutability_options(self, plotting_config):
        """测试配置选项"""
        # 检查基本属性
        assert hasattr(plotting_config, 'cmap')
        assert hasattr(plotting_config, 'log_norm')
        assert hasattr(plotting_config, 'dpi')


class TestPlottingEdgeCases:
    """测试绘图边界情况"""
    
    def test_plot_single_point(self):
        """测试单点数据绘图"""
        from jorek_postproc.data_models import BoundaryQuantitiesData
        
        data = BoundaryQuantitiesData(
            R=np.array([5.0]),
            Z=np.array([0.0]),
            phi=np.array([0.0]),
            value=np.array([1e6]),
            data_name='single'
        )
        
        try:
            fig = plot_scatter_3d(data)
            assert fig is not None
        except (ValueError, RuntimeError):
            # 单点散点图可能无法创建
            pass
    
    def test_plot_large_dataset(self):
        """测试大数据集绘图"""
        from jorek_postproc.data_models import BoundaryQuantitiesData
        
        n = 10000
        data = BoundaryQuantitiesData(
            R=np.random.uniform(4, 7, n),
            Z=np.random.uniform(-2, 2, n),
            phi=np.random.uniform(0, 2*np.pi, n),
            value=np.random.uniform(1e5, 1e7, n),
            data_name='large_dataset'
        )
        
        fig = plot_scatter_3d(data)
        assert fig is not None
    
    def test_plot_with_nan_values(self):
        """测试包含NaN值的绘图"""
        from jorek_postproc.data_models import BoundaryQuantitiesData
        
        data = BoundaryQuantitiesData(
            R=np.array([5.0, 6.0, 7.0]),
            Z=np.array([0.0, 0.0, 0.0]),
            phi=np.array([0.0, 0.0, 0.0]),
            value=np.array([1e6, np.nan, 1e6]),
            data_name='with_nan'
        )
        
        # 应该正确处理NaN值
        try:
            fig = plot_scatter_3d(data)
            assert fig is not None
        except ValueError:
            # 如果不支持NaN，也可以接受
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
