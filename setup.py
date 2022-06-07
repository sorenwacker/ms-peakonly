from setuptools import setup, find_packages

install_requires = [
    'bintrees',
    'matplotlib',
    'numpy',
    'pandas',
    'pymzML',
    'scipy',
    'torch>=1.2.0',
    'tqdm'
]


config = {
    'description': 'High level API for peakonly (https://github.com/Arseha/peakonly)',
    'author': 'Soren Wacker',
    'url': 'https://github.com/sorenwacker',
    'download_url': 'https://github.com/sorenwacker/ms_peak_only',
    'author_email': 'swacker@ucalgary.ca',
    'version': '0.0.1',
    'install_requires': install_requires,
    'packages': find_packages(),
    'scripts': [],
    'name': 'ms_peak_only'
}

setup(**config)
