from setuptools import setup

setup(
    name='Exploring fairness in Integrated Gradients methods',
    version='0.1',
    url='',
    install_requires=[
        'tensorflow>=2.6.0',
        'tensorflow-probability',
        'numpy',
        'scipy',
        'matplotlib',
        'pandas',
        'tensorflow-datasets',
        'pylint',
        'imageio',
        'path-explain'
    ],
    zip_safe=False
)
