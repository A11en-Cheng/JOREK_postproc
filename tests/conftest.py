"""
pytest配置文件

定义测试fixtures和配置
"""

import pytest
import numpy as np
from jorek_postproc import BoundaryQuantitiesData, PlottingConfig


@pytest.fixture
def sample_grid_data():
    """创建示例网格数据"""
    iplane = 10
    npoloidal = 20
    
    phi = np.linspace(0, 2*np.pi, iplane)
    theta = np.linspace(0, 2*np.pi, npoloidal)
    
    PHI, THETA = np.meshgrid(phi, theta, indexing='ij')
    
    R = 1.0 + 0.3 * np.cos(THETA)
    Z = 0.3 * np.sin(THETA)
    value = np.random.rand(iplane, npoloidal) * 1e6
    
    return BoundaryQuantitiesData(
        R=R,
        Z=Z,
        phi=PHI,
        theta=THETA,
        data=value,
        data_name='test_data',
        grid_shape=(iplane, npoloidal)
    )

@pytest.fixture
def sample_1d_data():
    """创建示例1D数据"""
    n_points = 1000
    
    R = np.random.uniform(0.5, 1.5, n_points)
    Z = np.random.uniform(-1.0, 1.0, n_points)
    phi = np.random.uniform(0, 2*np.pi, n_points)
    value = np.random.rand(n_points) * 1e6
    
    return BoundaryQuantitiesData(
        R=R,
        Z=Z,
        phi=phi,
        data=value,
        data_name='test_data_1d'
    )


@pytest.fixture
def plotting_config():
    """创建测试用绘图配置"""
    return PlottingConfig(
        log_norm=True,
        cmap='viridis',
        dpi=150,
        data_limits=[1e5, 1e7],
        find_max=False
    )


@pytest.fixture
def tmp_data_file(tmp_path):
    """创建临时JOREK数据文件"""
    file_path = tmp_path / "test_boundary_quantities.dat"
    
    content = """# time step (0001,)     0.00000000E+00
# R Z phi theta heatF_tot_cd
0.5  0.0  0.0  0.0  1e5
0.6  0.1  0.1  0.1  1.5e5
0.7  0.2  0.2  0.2  2e5
0.5  0.0  0.5  1.0  1.2e5
0.6  0.1  0.5  1.1  1.6e5
0.7  0.2  0.5  1.2  2.1e5
"""
    
    file_path.write_text(content)
    return str(file_path)
