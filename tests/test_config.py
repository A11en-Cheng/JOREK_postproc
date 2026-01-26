"""
测试配置模块
"""

import pytest
import argparse
from jorek_postproc.config import ProcessingConfig, create_parser

class TestProcessingConfig:
    """测试ProcessingConfig数据类逻辑"""
    
    def test_default_init(self):
        """测试默认初始化"""
        conf = ProcessingConfig(file_path="test.dat", timesteps=["100"])
        assert conf.dim is None
        assert conf.mode == "standard"
        assert conf.input_file == "test.dat" if hasattr(conf, 'input_file') else conf.file_path == "test.dat"
        
    def test_post_init_2d(self):
        """测试2D模式下的互斥逻辑"""
        # 手动设置 dim='2d' 和 plot_surface=True
        conf = ProcessingConfig(
            file_path="test.dat", 
            timesteps=["100"],
            dim="2d",
            plot_surface=True
        )
        # 验证 __post_init__ 是否将其强制关闭
        assert conf.plot_surface is False
        
    def test_post_init_energy_impact_flag(self):
        """测试backward compatibility flag"""
        conf = ProcessingConfig(
            file_path="test.dat", 
            timesteps=["100"],
            energy_impact=True
        )
        assert conf.mode == "energy_impact"
        
    def test_post_init_plot_set_defaults(self):
        """测试plot_set模式默认维度"""
        conf = ProcessingConfig(
            file_path="test.dat", 
            timesteps=["100"],
            mode="plot_set"
        )
        assert conf.dim == "2d"


class TestArgParser:
    """测试参数解析器"""
    
    @pytest.fixture
    def parser(self):
        return create_parser()
        
    def test_parse_timesteps(self, parser):
        args = parser.parse_args(['-f', 'file.dat', '-t', '100', '200'])
        assert args.timesteps == ['100', '200']
        
    def test_parse_dim_flag(self, parser):
        args = parser.parse_args(['-f', 'file.dat', '-t', '100', '--dim', '2d'])
        assert args.dim == '2d'
        
    def test_parse_arc_length_flag(self, parser):
        args = parser.parse_args(['-f', 'file.dat', '-t', '100', '--use_arc_length'])
        assert args.use_arc_length is True
        
    def test_parse_mode_flag(self, parser):
        args = parser.parse_args(['-f', 'file.dat', '-t', '100', '-m', 'energy_impact'])
        assert args.mode == 'energy_impact'
