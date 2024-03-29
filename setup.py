from setuptools import find_packages
from setuptools import setup

NAME = "discovery_engine_lite"
VERSION = "0.1.0"
# To install the library, run the following
#
# python setup.py install
#
# prerequisite: setuptools
# http://pypi.python.org/pypi/setuptools

REQUIRES = [
    "fastapi",
    "nltk",
    "numpy",
    "pydantic",
    "pandas",
    "requests",
    "scikit-learn",
    "scipy",
    "uvicorn",
    "xxhash",
    "elasticsearch>=7.0.0,<8.0.0",
    "networkx==3.2.1",
    "pyahocorasick",
    "tiktoken",
]

setup(
    name=NAME,
    version=VERSION,
    description="NLM Discovery Engine Lite",
    author_email="info@nlmatics.com",
    url="nlmatics.com",
    keywords=["NLM Discovery Engine Lite"],
    install_requires=REQUIRES,
    packages=find_packages(),
    package_dir={"discovery_engine_lite": ""},
    include_package_data=True,
    long_description="""\
    API specification for discovery_engine_lite  # noqa: E501
    """,
)
