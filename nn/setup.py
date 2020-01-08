# -*- coding:utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from glob import glob

from Cython.Distutils import build_ext

'''
    USAGE: 

    WARNING:BACKUP origin file!!!!!

        Step1: put this file into the src dir.
        Step2: run the command  "python ./setup.py build_ext  --inplace"
    Step3: del the py and pyc files.
'''

path = './'
exclude_file = ["__init__.py", "setup.py"]

pylist = glob(path + "*.py")
pylist = [i.split("/")[-1] for i in pylist if i.split("/")[-1] not in exclude_file]

print(pylist)

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension(pl.split('.')[0].replace("/", "."), [pl]) for pl in pylist], requires=['tensorflow',
                                                                                                  'scikit-learn',
                                                                                                  'Cython', 'numpy',
                                                                                                  'pandas']
)
