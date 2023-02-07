""" rmnn: Data representation library to represent data with various of data based on PyTorch
"""


__title__ = 'rmnn'
__version__ = '0.0.2'
#__build__ = 0x021300
__author__ = 'Zhemin Li'
__license__ = 'MIT'
__copyright__ = 'Copyright 2023 Zhemin Li'


## Top Level Modules
from rmnn import toolbox,represent
# from rmnn.represent import get_nn
# TODO 最后只需要开放app即可
__all__ = ["toolbox","represent"]
# "apps","frame","toolbox","represent"]

### Representations
# TODO add 
# from impyute.imputation.cs import mean
# from impyute.imputation.cs import median
# from impyute.imputation.cs import mode
# from impyute.imputation.cs import em
# from impyute.imputation.cs import fast_knn
# from impyute.imputation.cs import buck_iterative
# from impyute.imputation.cs import random

# __all__.extend([
#     "mean",
#     "median",
#     "mode",
#     "em",
#     "fast_knn",
#     "buck_iterative",
#     "random"
# ])

### Construct loss function



### Low-level inverse problems solving

