from setuptools import setup

from setuptools import find_packages

setup(

    name='general',
    version='1.0',
    description='A Deep Reinforcement Learning Framework Base  on huskarl',
    author='Enjoy Zhao',
	author_email='934389697@qq.com',
	url='https://github.com/zhaoyingjun/general.git',

	python_requires='>=3.6',
	install_requires=[
		'cloudpickle',
		'tensorflow==2.6.4',
		'scipy',
        'gym',
        'numpy', 'wxpython', 'matplotlib','flask','flask_restful'
    ],
	packages=find_packages()
















)