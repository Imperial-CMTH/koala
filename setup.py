from setuptools import setup, find_packages

setup(
    name="koala",
    version='0.0',
    description='Topological Amorphous quantum system simulations',
    long_description='',
    author="Peru D'Ornellas, Gino Cassella, Tom Hodson",
    author_email='',
    license='Apache Software License',
    home_page='',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'flake8',
        'python-sat',
        'pytest',
        'pytest-cov',
        'nbmake',
        'pytest-github-actions-annotate-failures',
        'tinyarray'
    ]
)