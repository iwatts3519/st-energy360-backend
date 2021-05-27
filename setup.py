"""
    TIM Engine API Client

    To install the library, run the following

    python setup.py install

    prerequisite: setuptools
    http://pypi.python.org/pypi/setuptools
"""

from setuptools import setup
from tim_client._version import __version__

setup(
    name='tim_client',
    version=__version__,
    description='TIM Engine API Client',
    author='Tangent Works',
    author_email='info@tangent.works',
    url='',
    keywords=['TIM Engine'],
    install_requires=['pylint', 'autopep8', 'pandas', 'requests', 'pyyaml', 'python-dateutil'],
    packages=['tim_client'],
    include_package_data=True,
    long_description="""
    TIM Engine API Client
    """
)