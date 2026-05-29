from setuptools import setup, find_packages

setup(
    name="pcd_green_ai",
    version="1.1.0",
    author="Hara",
    author_email="hara@domain.com",
    description="Phenomenological Computational Dynamics (PCD) multi-dimensional hardware-software metric space engine.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/PikkiHara/phenomenological_general_science",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch>=2.0.0",
    ],
    python_requires=">=3.8",
)
