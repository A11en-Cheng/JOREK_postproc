"""
Setup配置文件 - 允许安装jorek_postproc包

安装方法：
  pip install -e .
  
或者：
  python setup.py develop
"""

from setuptools import setup, find_packages

with open("jorek_postproc/README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="jorek_postproc",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="JOREK后处理包 - 边界量可视化工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/jorek_postproc",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.1.0",
        "scipy>=1.5.0",
    ],
    entry_points={
        "console_scripts": [
            "jorek-postproc=jorek_postproc.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
