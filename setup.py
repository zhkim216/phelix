from setuptools import setup, find_packages

setup(
    name='allatom_design',
    version='0.1',
    packages=find_packages(exclude=["tests", "scripts"]),
    include_package_data=True,
    python_requires='>=3.9',
)
