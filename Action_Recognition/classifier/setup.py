from distutils.core import setup, Extension

module = Extension("DTW", sources=["DTWmodule.c"])

setup(name="DTW", \
      version="1.0", \
      description="Use dynamic time warping to align sequences", \
      ext_modules=[module])
