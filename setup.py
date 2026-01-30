"""Setup configuration for Water Meter AI package"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="water-meter-ai",
    version="1.0.0",
    author="Arsenius Purbandono",
    author_email="your.email@example.com",
    description="On-Device AI for Analog Water Meter Reading",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/water-meter-ai",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
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
    install_requires=[
        "ultralytics>=8.0.0",
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pyyaml>=6.0",
        "loguru>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "jupyter>=1.0.0",
        ],
    },
)
