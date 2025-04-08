# 加密代码
# 使用说明
# 1. 运行python setup.py build_ext --inplace
# 2. 将所有同名的py文件删除，so文件会替代py文件，如果py文件和so文件同时存在，那么会优先使用py文件
from setuptools import setup
from Cython.Build import cythonize

# python setup.py build_ext --inplace
if __name__ == "__main__":
    setup(ext_modules=cythonize([".\*.py"], compiler_directives={"language_level": "3"}, build_dir="build"))
