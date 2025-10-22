"""
CausalMMM Setup Configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="causalmmm",
    version="0.1.0",
    author="CausalMMM Contributors",
    author_email="your.email@example.com",
    description="Learning Causal Structure for Marketing Mix Modeling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/causalmmm",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=3.0",
            "black>=22.0",
            "flake8>=4.0",
            "mypy>=0.950",
        ],
        "viz": [
            "matplotlib>=3.5",
            "seaborn>=0.11",
            "plotly>=5.0",
            "networkx>=2.8",
            "pyvis>=0.2",
        ],
        "cdnots": [
            "causal-learn>=0.1.3",
        ],
    },
    include_package_data=True,
    package_data={
        "causalmmm": ["py.typed"],
    },
)