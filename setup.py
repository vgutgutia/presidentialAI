"""Setup script for marine-debris-detection package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="marine-debris-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="AI-powered marine debris detection using satellite imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/marine-debris-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "mdd-train=scripts.train:main",
            "mdd-predict=scripts.predict:main",
            "mdd-evaluate=scripts.evaluate:main",
            "mdd-download=scripts.download_marida:main",
        ],
    },
)
