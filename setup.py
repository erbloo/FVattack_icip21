from setuptools import setup
from setuptools import find_packages

requirements = [
    'numpy',
    'matplotlib',
    'urllib3',
    'tqdm',
    'pillow',
    'scipy'
]

setup(
    name='tidr_icip21',
    description='Code for ICIP2021 submission TI-DR.',
    version='1.0',
    url="https://github.com/erbloo/tidr_icip21",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements
)
