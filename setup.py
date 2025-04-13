from setuptools import setup, find_packages
from lockin import __version__


setup(
    name="lockin",
    version=__version__,
    packages=find_packages(include=["lockin"]),
    install_requires=[
        "blobfile==3.0.0",
        "pydantic==2.11.3",
        "tiktoken==0.9.0",
        "tqdm==4.67.1"
    ],
    extras_require={
        "dev": [
            "pytest==8.3.5"
        ]
    }
)