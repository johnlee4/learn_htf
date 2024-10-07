# setup.py
from setuptools import setup, find_packages

setup(
    name="learn_htf",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "pandas"
    ],
    author="John Lee",
    description="A custom machine learning package.\
        Based on 'The Elements of Statistical Learning' by Hastie, Tibshirani, Friedman",
)
