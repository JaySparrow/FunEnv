from setuptools import setup, find_packages

setup(
    name='amusepark',
    version='0.1.0',
    description='OpenAI GYM environments of fun games.',
    author='Yuhan Liu',
    author_email='jaysparrowthegreat@gmail.com',
    packages=find_packages(include=['amusepark', 'amusepark.*']),
    install_requires=[
        'numpy',
        'gym'
    ]
)
