try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

long_description = """\
Another version of sklearn.cluster.MeanShift that adapts labeling by attraction mechanism.
The attraction mechanism function as connected component labeling powered by scipy
"""

setup(
    name='pymeanshiftclustering',
    version='0.1.0',
    description=('another version of sklearn.cluster.MeanShift with attraction labeling'),
    long_description=long_description,
    py_modules=['pymeanshiftclustering'],
    install_requires=['numpy','sklearn','scipy'],
    author='Taesik Yoon',
    author_email='taesik.yoon.02@gmail.com',
    url='https://github.com/otterrrr/pymeanshiftclustering',
    license='BSD-3-Clause',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD-3-Clause license",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Operating System :: OS Independent"
    ]
)