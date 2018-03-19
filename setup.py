# encoding: utf-8
from setuptools import setup

setup(name='imaginet',
      version='2.1',
      description='Visually grounded word and sentence representations',
      url='https://github.com/gchrupala/speechvision',
      author='Grzegorz Chrupała',
      author_email='g.chrupala@uvt.nl',
      license='MIT',
      packages=['imaginet','vg'],
      install_requires=[
          'Theano',
          'funktional==1.2'
                    ],
      zip_safe=False)
