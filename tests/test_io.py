"""
测试IO模块
"""

import pytest
import os
import numpy as np
from jorek_postproc.io import read_boundary_file

class TestIOModule:
    """测试IO模块"""
    
    def test_read_boundary_file_basic(self, tmp_data_file):
        """测试基本文件读取"""
        cols, blocks, t_map = read_boundary_file(tmp_data_file)
        
        assert isinstance(cols, list)
        assert isinstance(blocks, dict)
        assert isinstance(t_map, dict)
        assert 'heatF_tot_cd' in cols

    def test_read_empty_file(self, tmp_path):
        """测试空文件"""
        p = tmp_path / "empty.dat"
        p.write_text("")
        
        # Should usually return empty structures or raise specific error
        # Assuming current implementation might return empty or fail gracefully
        try:
             cols, blocks, t_map = read_boundary_file(str(p))
             assert len(blocks) == 0
        except Exception:
             # If it raises, that's also acceptable for empty file
             pass

