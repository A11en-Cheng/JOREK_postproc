"""
测试数据处理模块
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from jorek_postproc.processing import process_multiple_timesteps, apply_data_limits
from jorek_postproc.data_models import BoundaryQuantitiesData

class TestProcessingModule:
    """测试数据处理模块"""
    
    @patch('jorek_postproc.processing.read_boundary_file')
    @patch('jorek_postproc.processing.reshape_to_grid')
    @patch('os.path.exists', return_value=True)
    def test_process_multiple_timesteps_flow(self, mock_exists, mock_reshape, mock_read):
        """测试多时间步处理流程"""
        # Setup mocks
        mock_read.return_value = (
            ['R', 'Z', 'phi', 'val'], 
            {'000100': np.zeros((10, 4))}, # blocks
            {'000100': 1.0}                # time mapping
        )
        
        mock_grid_data = MagicMock(spec=BoundaryQuantitiesData)
        mock_grid_data.time_step = None
        mock_grid_data.time = None
        mock_reshape.return_value = mock_grid_data
        
        # Call function
        results = process_multiple_timesteps(
            timesteps=['100'],
            file_addr='/tmp',
            column_names=['R', 'Z', 'phi', 'val'],
            names=('R', 'Z', 'phi', 'val'),
            xpoints=None,
            debug=True
        )
        
        assert '000100' in results
        assert results['000100'] == mock_grid_data
        assert mock_grid_data.time_step == '000100'
        assert mock_grid_data.time == 1.0

    def test_apply_data_limits(self, sample_grid_data):
        """测试数据限制应用"""
        # sample_grid_data from conftest
        
        # Set some known values
        sample_grid_data.data = np.array([-100, 0, 50, 200, 1000])
        
        clipped = apply_data_limits(sample_grid_data, limits=[0, 100])
        
        assert np.all(clipped.data >= 0)
        assert np.all(clipped.data <= 100)
        # Check specific values
        expected = np.array([0, 0, 50, 100, 100])
        np.testing.assert_array_equal(clipped.data, expected)

