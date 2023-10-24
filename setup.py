from setuptools import setup, find_packages

setup(
    name="learn_to_pick",
    version="0.1",
    install_requires=[
        'numpy',
        'pandas',
        'vowpal-wabbit-next',
        'sentence-transformers',
        'torch',
        'pyskiplist',
        'parameterfree',
    ],
    extras_require={
        'dev': [
            'pytest'
        ]
    },
    author="VowpalWabbit",
    description="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    url="https://github.com/VowpalWabbit/learn_to_pick",
    python_requires='>=3.8',
)
