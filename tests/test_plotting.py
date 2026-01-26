"""
测试绘图/可视化模块
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
from jorek_postproc.plotting import plot_surface_3d, plot_scatter_3d, plot_heat_flux_analysis
from jorek_postproc.data_models import PlottingConfig, BoundaryQuantitiesData


class TestPlottingModule:
    """测试绘图模块"""
    
    @pytest.fixture(autouse=True)
    def cleanup(self):
        """在每个测试后清理图表"""
        yield
        plt.close('all')
    
    @patch('jorek_postproc.plotting.plt.show')
    def test_plot_surface_3d_basic(self, mock_show, sample_grid_data, plotting_config):
        """测试基本3D曲面绘图"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # plot_surface_3d currently returns None and closes fig
        with patch.object(ax, 'plot_surface') as mock_plot:
             plot_surface_3d(sample_grid_data, fig, ax, plotting_config)
             assert mock_plot.called
             
        # Verify show called since save_path is None
        assert mock_show.called
    
    @patch('jorek_postproc.plotting.plt.show')
    def test_plot_scatter_3d_basic(self, mock_show, sample_1d_data, plotting_config):
        """测试基本3D散点绘图"""
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        with patch.object(ax, 'scatter') as mock_scatter:
            plot_scatter_3d(sample_1d_data, fig, ax, plotting_config)
            assert mock_scatter.called
        
        assert mock_show.called

    @patch('jorek_postproc.plotting.plt.show')
    def test_plot_heat_flux_analysis_2d(self, mock_show, sample_grid_data, plotting_config):
        """测试2D热流分析图"""
        try:
            plot_heat_flux_analysis(sample_grid_data, plotting_config)
        except Exception as e:
            pytest.fail(f"plot_heat_flux_analysis failed: {e}")
            
        # It calls show() if no save_path
        assert mock_show.called

    @patch('jorek_postproc.plotting.plt.show')
    def test_plot_surface_with_max_point(self, mock_show, sample_grid_data):
        """测试带有最大值标记的曲面绘图"""
        config = PlottingConfig(find_max=True)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        with patch.object(ax, 'plot_surface') as mock_plot:
            with patch('jorek_postproc.plotting.plot_max_point') as mock_max_point:
                plot_surface_3d(sample_grid_data, fig, ax, config)
                
                assert mock_plot.called
                assert mock_max_point.called

 
class TestPlottingEdgeCases:
    """测试边界情况"""
    
    @patch('jorek_postproc.plotting.plt.show')
    def test_plot_single_point(self, mock_show):
        """测试单点数据绘图"""
        data = BoundaryQuantitiesData(
            R=np.array([5.0]),
            Z=np.array([0.0]),
            phi=np.array([0.0]),
            data=np.array([1e6]),
            data_name='single'
        )
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_scatter_3d(data, fig, ax)

    @patch('jorek_postproc.plotting.plt.show')
    def test_plot_with_nan_values(self, mock_show):
        """测试包含NaN值的绘图"""
        # Should not crash
        data = BoundaryQuantitiesData(
            R=np.array([5.0, 6.0, 7.0]),
            Z=np.array([0.0, 0.0, 0.0]),
            phi=np.array([0.0, 0.0, 0.0]),
            data=np.array([1e6, np.nan, 1e6]),
            data_name='with_nan'
        )
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plot_scatter_3d(data, fig, ax)

