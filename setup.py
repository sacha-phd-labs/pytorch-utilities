from pathlib import Path
from setuptools import setup, find_packages

HERE = Path(__file__).parent

# Read long description from README.md if present
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

setup(
    name="pytorcher",
    version="0.1.0",
    description="Pytorcher - A PyTorch-based framework for deep learning applications",
    long_description=README,
    long_description_content_type="text/markdown",
    author="sacha.bouchez-delotte",
    author_email="",
    url="",
    packages=find_packages(exclude=["tests", "docs"]),
    include_package_data=True,
    install_requires=[
    ],
    python_requires=">=3.8",
    license="MIT",
    zip_safe=False,
)
