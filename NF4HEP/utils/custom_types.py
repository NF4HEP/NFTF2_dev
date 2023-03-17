__all__ = ["Array",
           "ArrayInt",
           "ArrayStr",
           "DataType",
           "DictStr",
           "DTypeStr",
           "DTypeStrList",
           "StrPath",
           "IntBool",
           "StrBool",
           "StrList",
           "FigDict",
           "LogPredDict",
           "Number"]

import numpy as np
from numpy import typing as npt
import h5py # type: ignore
from pathlib import Path
from typing import Union, List, Dict, Optional, Any, Callable

# types definitions
Array = Union[List, npt.NDArray]
ArrayInt = Union[List[int], npt.NDArray[np.int_]]
ArrayStr = Union[List[str], npt.NDArray[np.str_]]
DataType = Union[npt.NDArray,h5py.Dataset]
DictStr = Union[Dict,str]
DTypeStr = Union[str,npt.DTypeLike]
DTypeStrList = Union[List[str],List[npt.DTypeLike]]
StrPath = Union[str,Path]
IntBool = Union[int, bool]
StrBool = Union[str, bool]
StrList = Union[str, List[str]]
StrArray = Union[str, List, npt.NDArray]
FigDict = Dict[str,List[Path]]
LogPredDict = Dict[str,Dict[str,Any]]
Number = Union[int,float]