"""
测试能量冲击分析模块
"""

import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from jorek_postproc.energy_impact import compute_delta_q_batch, load_timestep_data
from jorek_postproc.data_models import BoundaryQuantitiesData

class TestEnergyImpactMath:
    """测试数学计算核心函数"""
    
    def test_compute_delta_q_batch_basic(self):
        """测试基本卷积计算"""
        # 构造简单输入
        # q(t) = 1 (constant)
        # delta_q should increase with time
        
        n_pixels = 10
        n_raw = 5
        n_eval = 5
        
        t_raw = np.linspace(0, 1, n_raw, dtype=np.float32)
        q_raw_batch = np.ones((n_pixels, n_raw), dtype=np.float32)
        t_eval = np.linspace(0, 1, n_eval, dtype=np.float32)
        
        dq = compute_delta_q_batch(t_raw, q_raw_batch, t_eval)
        
        assert dq.shape == (n_pixels, n_eval)
        # check monotonicity (energy deposition increases)
        # Actually delta_q is temperature rise ~ integral of q / sqrt(t-tau)
        # for constant q, it should increase.
        assert np.all(np.diff(dq, axis=1) >= 0)

    def test_compute_delta_q_batch_zero_input(self):
        """测试零输入"""
        n_pixels = 5
        t_raw = np.array([0, 1], dtype=np.float32)
        q_raw = np.zeros((n_pixels, 2), dtype=np.float32)
        t_eval = np.array([0.5, 1.0], dtype=np.float32)
        
        dq = compute_delta_q_batch(t_raw, q_raw, t_eval)
        
        assert np.all(dq == 0)

class TestEnergyImpactLoading:
    """测试数据加载任务"""
    
    @patch('jorek_postproc.energy_impact.read_boundary_file')
    @patch('jorek_postproc.energy_impact.reshape_to_grid')
    @patch('os.path.exists', return_value=True)
    def test_load_timestep_data_success(self, mock_exists, mock_reshape, mock_read):
        """测试成功加载单步数据"""
        # Mock setup
        mock_read.return_value = (
            ['val_col'],
            {'block_key_s0100': np.array([[1,2,3]])},
            {'block_key_s0100': 0.05}
        )
        mock_reshape.return_value = MagicMock(spec=BoundaryQuantitiesData)
        
        args = (100, 'main.dat', '/dir', 'val_col', None, False)
        
        time_val, data = load_timestep_data(args)
        
        assert time_val == 0.05
        assert data is not None

    @patch('jorek_postproc.energy_impact.read_boundary_file')
    @patch('os.path.exists', return_value=True)
    def test_load_timestep_data_missing_column(self, mock_exists, mock_read):
        """测试列不存在情况"""
        mock_read.return_value = (
            ['other_col'], 
            {'block_key_s0100': np.array([[1]])}, 
            {}
        )
        
        args = (100, 'main.dat', '/dir', 'target_col', None, True)
        
        time_val, data = load_timestep_data(args)
        
        assert data is None

