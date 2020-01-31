from setuptools import setup

from setuptools import find_packages

setup(

    name='general',
    version='0.1',
    description='A Deep Reinforcement Learning Framework Base  on huskarl',
    author='Enjoy Zhao',
	author_email='yingjun.xuda@gmail.com',
	url='https://github.com/zhaoyingjun/general.git',
	classifiers=[
        'Development Status :: 1 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
	python_requires='>=3.6',
	install_requires=[
		'cloudpickle',
		'tensorflow==2.1',
		'scipy',
        'gym',
        'numpy',
	],
	packages=find_packages()
















)