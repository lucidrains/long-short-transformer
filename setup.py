from setuptools import setup, find_packages

setup(
  name = 'long-short-transformer',
  packages = find_packages(),
  version = '0.0.4',
  license='MIT',
  description = 'Long Short Transformer - Pytorch',
  author = 'Phil Wang',
  author_email = 'lucidrains@gmail.com',
  url = 'https://github.com/lucidrains/long-short-transformer',
  keywords = [
    'artificial intelligence',
    'deep learning',
    'transformers',
    'efficient attention'
  ],
  install_requires=[
    'einops>=0.3',
    'rotary-embedding-torch',
    'torch>=1.6'
  ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
