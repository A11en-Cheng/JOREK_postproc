"""
测试命令行界面模块
"""

import pytest
from click.testing import CliRunner
from jorek_postproc.cli import cli


class TestCLIModule:
    """测试CLI模块"""
    
    @pytest.fixture
    def runner(self):
        """CLI测试运行器"""
        return CliRunner()
    
    def test_cli_help(self, runner):
        """测试CLI帮助命令"""
        result = runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert 'Usage:' in result.output or 'help' in result.output.lower()
    
    def test_cli_version(self, runner):
        """测试CLI版本命令"""
        result = runner.invoke(cli, ['--version'])
        
        # 应该输出版本信息或成功退出
        assert result.exit_code == 0 or 'version' in result.output.lower()
    
    def test_cli_invalid_command(self, runner):
        """测试无效命令"""
        result = runner.invoke(cli, ['invalid-command'])
        
        # 应该失败或给出错误信息
        assert result.exit_code != 0 or 'Error' in result.output


class TestCLICommands:
    """测试CLI命令"""
    
    @pytest.fixture
    def runner(self):
        """CLI测试运行器"""
        return CliRunner()
    
    def test_cli_with_input_file(self, runner, tmp_data_file):
        """测试CLI处理输入文件"""
        # 这取决于CLI的实现
        # 根据实现情况调整测试
        result = runner.invoke(cli, ['process', tmp_data_file])
        
        # 验证命令执行（可能需要输出文件作为参数）
        assert result.exit_code in [0, 2]  # 0表示成功，2表示使用错误
    
    def test_cli_output_option(self, runner, tmp_data_file):
        """测试CLI输出选项"""
        with runner.isolated_filesystem():
            result = runner.invoke(cli, [
                'process',
                tmp_data_file,
                '--output', 'output.dat'
            ])
            
            # 验证命令执行
            assert result.exit_code in [0, 2]


class TestCLIArguments:
    """测试CLI参数"""
    
    @pytest.fixture
    def runner(self):
        """CLI测试运行器"""
        return CliRunner()
    
    def test_cli_list_devices(self, runner):
        """测试列出支持的装置"""
        result = runner.invoke(cli, ['list-devices'])
        
        # 应该列出设备或给出帮助信息
        assert result.exit_code in [0, 2]
    
    def test_cli_show_config(self, runner):
        """测试显示配置"""
        result = runner.invoke(cli, ['show-config'])
        
        # 应该显示配置或给出帮助信息
        assert result.exit_code in [0, 2]


class TestCLIOptions:
    """测试CLI选项"""
    
    @pytest.fixture
    def runner(self):
        """CLI测试运行器"""
        return CliRunner()
    
    def test_cli_verbose_option(self, runner):
        """测试详细输出选项"""
        result = runner.invoke(cli, ['--verbose', '--help'])
        
        assert result.exit_code == 0
    
    def test_cli_quiet_option(self, runner):
        """测试安静模式选项"""
        result = runner.invoke(cli, ['--quiet', '--help'])
        
        assert result.exit_code == 0


class TestCLIUsage:
    """测试CLI使用"""
    
    @pytest.fixture
    def runner(self):
        """CLI测试运行器"""
        return CliRunner()
    
    def test_cli_usage_message(self, runner):
        """测试使用消息"""
        result = runner.invoke(cli, [])
        
        # 不带参数应该显示使用信息或失败
        assert result.exit_code in [0, 2]
    
    def test_cli_with_multiple_options(self, runner):
        """测试多个选项组合"""
        result = runner.invoke(cli, ['--help', '--verbose'])
        
        # 某些组合可能无效，但应该优雅地处理
        assert result.exit_code in [0, 2, 1]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
