"""
测试I/O模块功能
"""

import pytest
import os
from jorek_postproc.io import read_boundary_file
from jorek_postproc.data_models import BoundaryQuantitiesData


class TestIOModule:
    """测试输入输出模块"""
    
    def test_read_boundary_file_basic(self, tmp_data_file):
        """测试基本文件读取"""
        data = read_boundary_file(tmp_data_file)
        
        assert isinstance(data, BoundaryQuantitiesData)
        assert data.data_name == 'boundary_quantities'
        assert len(data.R) > 0
        assert len(data.Z) > 0
    
    def test_read_boundary_file_shape(self, tmp_data_file):
        """测试读取数据形状"""
        data = read_boundary_file(tmp_data_file)
        
        # 检查数据一致性
        assert len(data.R) == len(data.Z)
        assert len(data.R) == len(data.phi)
        assert len(data.R) == len(data.value)
    
    def test_read_boundary_file_nonexistent(self):
        """测试读取不存在的文件"""
        with pytest.raises((FileNotFoundError, OSError)):
            read_boundary_file('/nonexistent/path/file.dat')
    
    def test_read_boundary_file_data_integrity(self, tmp_data_file):
        """测试数据完整性"""
        data = read_boundary_file(tmp_data_file)
        
        # 检查值范围
        assert data.value.min() >= 0
        assert data.value.max() > 0
        
        # 检查R和Z值合理性（应在物理合理范围内）
        assert data.R.min() > 0
        assert data.R.max() < 1000
    
    def test_read_boundary_file_metadata(self, tmp_data_file):
        """测试元数据"""
        data = read_boundary_file(tmp_data_file)
        
        assert hasattr(data, 'timestamp')
        assert hasattr(data, 'data_name')
        assert hasattr(data, 'device')


class TestFileHandling:
    """测试文件处理"""
    
    def test_file_exists(self, tmp_data_file):
        """测试临时文件存在"""
        assert os.path.exists(tmp_data_file)
    
    def test_file_readable(self, tmp_data_file):
        """测试文件可读"""
        assert os.access(tmp_data_file, os.R_OK)
    
    def test_file_format_validation(self, tmp_data_file):
        """测试文件格式验证"""
        # 确保文件不为空
        assert os.path.getsize(tmp_data_file) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
