from setuptools import find_packages
from setuptools import setup


install_requires = ['numpy', 'matplotlib', 'gym', 'pandas']
#tests_require = ['pytest']
#setup_requires = ["pytest-runner"]

setup(
    name='MPCBenchmark',
    version='1.0',
    description='Implementing MPC Algorithms and a Testing framework',
    author='Tristan Schulz, Darya Nikitina',
    author_email='',
    install_requires=install_requires,
    url='https://github.com/MrDiver/RoboticsIPBenchmark',
    #license='MIT License',
    packages=find_packages(exclude=('tests')),
    #setup_requires=setup_requires,
    #test_suite='tests',
    #tests_require=tests_require
)