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
          'Theano==1.0.0rc1',
          'funktional==1.3',
          'sklearn==0.0',
          'python_speech_features==0.6',
          'soundfile==0.9.0.post1'
                    ],
      zip_safe=False)
