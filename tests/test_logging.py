"""
Test logging module
"""
import pytest
import logging
import os
from unittest.mock import patch, MagicMock
from jorek_postproc.logging import setup_logging, get_logger

def test_setup_logging():
    logger = setup_logging("test_logger", verbose=True)
    assert logger.name == "test_logger"
    assert logger.level == logging.DEBUG # Verbose -> DEBUG

    logger2 = setup_logging("test_logger_2", verbose=False)
    assert logger2.level == logging.INFO

def test_get_logger():
    logger = get_logger("my_module")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "my_module"

    # Test singleton-like behavior if configured
    logger2 = get_logger("my_module")
    assert logger is logger2
