from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name = "Project1_house_reservation",
    version= "0.1.0",
    author = "Shreya Singh",
    packages = find_packages(),
    install_requires = requirements,
)



## run using
# pip install -e .