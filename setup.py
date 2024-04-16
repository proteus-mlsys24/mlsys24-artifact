from setuptools import setup, find_packages

setup(name='proteus', 
      verson='1.0', 
      packages=find_packages(),
      install_requires=[
        "z3-solver",
        "onnx",
        "graphviz",
        "termcolor",
        "tqdm",
        "matplotlib"
      ])
