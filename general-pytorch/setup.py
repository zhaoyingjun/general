

from setuptools import setup, find_packages

setup(
    name='general',
    version='1.0',
    description='A Deep Reinforcement Learning Framework Base on huskarl',
    author='Enjoy Zhao',
    author_email='934389697@qq.com',
    url='https://github.com/zhaoyingjun/general.git',
    python_requires='>=3.6',
    install_requires=[
        'cloudpickle',
        'torch',
        'scipy',
        'gym',
        'numpy',
        'wxpython',
        'matplotlib',
        'flask',
        'flask_restful'
    ],
    packages=find_packages(),
    # 推荐加上 long_description
    long_description=open('README.md', encoding='utf-8').read() if os.path.exists('README.md') else '',
    long_description_content_type='text/markdown',
)















)