"""
Test diagnostics module
"""
import pytest
from unittest.mock import patch, MagicMock
from jorek_postproc.diagnostics import check_environment, print_environment, validate_installation

def test_check_environment():
    env = check_environment()
    assert 'python_version' in env
    assert 'jorek_postproc_version' in env
    assert 'numpy_version' in env

def test_print_environment(capsys):
    print_environment()
    captured = capsys.readouterr()
    assert "jorek_postproc 环境信息" in captured.out
    assert "python_version" in captured.out

def test_validate_installation():
    is_valid, issues = validate_installation()
    # In a proper dev environment this should ideally be valid, 
    # but we just want to ensure it runs and returns bool/list.
    assert isinstance(is_valid, bool)
    assert isinstance(issues, list)
