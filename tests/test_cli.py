"""
测试命令行接口
"""

import pytest
import sys
import subprocess
from unittest.mock import patch, MagicMock
from jorek_postproc.cli import main
import jorek_postproc.config as cfg

class TestCLIExecution:
    """测试CLI执行流程"""
    
    @patch('jorek_postproc.cli.process_single_timestep')
    def test_standard_mode_call(self, mock_process):
        """测试标准模式调用"""
        test_args = ['program', '-f', 'test.dat', '-t', '100', '--dim', '2d']
        with patch.object(sys, 'argv', test_args):
            # Mock file check inside config parsing or processing if needed
            # But parse_args doesn't check file existence, processing func does.
            # We mock file existence check if it happens in verify_config
            with patch('os.path.exists', return_value=True):
                main()
                
        assert mock_process.called
        conf = mock_process.call_args[0][0]
        assert conf.dim == '2d'
        assert conf.mode == 'standard'

    @patch('jorek_postproc.energy_impact.run_energy_impact_analysis')
    def test_energy_impact_mode_call(self, mock_energy):
        """测试能量冲击模式调用"""
        test_args = ['program', '-f', 'test.dat', '-t', '100', '-m', 'energy_impact']
        with patch.object(sys, 'argv', test_args):
            with patch('os.path.exists', return_value=True):
                main()
        
        assert mock_energy.called
        conf = mock_energy.call_args[0][0]
        assert conf.mode == 'energy_impact'

    @patch('jorek_postproc.boundary_analysis.run_boundary_analysis')
    def test_plot_set_mode_call(self, mock_boundary):
        """测试图集模式调用"""
        test_args = ['program', '-f', 'test.dat', '-t', '100', '--mode', 'plot_set']
        with patch.object(sys, 'argv', test_args):
            with patch('os.path.exists', return_value=True):
                 main()
                 
        assert mock_boundary.called
        conf = mock_boundary.call_args[0][0]
        # Check defaults logic for plot_set
        assert conf.dim == '2d' # Config __post_init__ logic


class TestCLIIntegration:
    """集成测试：作为子进程运行"""
    
    def test_help_output(self):
        """测试帮助信息"""
        result = subprocess.run(
            [sys.executable, '-m', 'jorek_postproc.cli', '--help'],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert 'usage:' in result.stdout.lower() or 'usage:' in result.stderr.lower()
        assert '--dim' in result.stdout  # Check for new flags
        assert '--use_arc_length' in result.stdout

    def test_missing_args(self):
        """测试缺少必要参数"""
        result = subprocess.run(
            [sys.executable, '-m', 'jorek_postproc.cli'],
            capture_output=True,
            text=True
        )
        assert result.returncode != 0
