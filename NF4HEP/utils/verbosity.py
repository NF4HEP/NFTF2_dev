import sys
import builtins
#from builtins import print
from pathlib import Path

from typing import Union, List, Dict, Callable, Tuple, Optional, NewType, Type, Generic, Any, TypeVar, TYPE_CHECKING, Text, TextIO
from typing_extensions import TypeAlias
from .custom_types import Array, ArrayInt, ArrayStr, DataType, StrPath, IntBool, StrBool, StrList, FigDict, LogPredDict, Number, DTypeStr, DTypeStrList, DictStr

class Verbosity():
    """
    Class inherited by all other classes to provide the 
    :meth:`Verbosity.get_verbosity <DNNLikelihood.Verbosity.get_verbosity>` method.
    """
    def __init__(self,
                 verbose: Optional[IntBool] = True
                ) -> None:
        self.verbose = verbose if verbose is not None else True

    @property
    def verbose(self) -> IntBool:
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: Optional[IntBool]) -> None:
        self._verbose, self._verbose_sub = self.get_verbosity(verbose = verbose)

    @property
    def verbose_sub(self) -> IntBool:
        return self._verbose_sub

    def get_verbosity(self, verbose: Optional[IntBool]) -> list[IntBool]:
        """
        Method inherited by all classes (from the :class:`Verbosity <DNNLikelihood.Verbosity>` class)
        used to set the verbosity mode. If the input argument ``verbose`` is ``None``, ``verbose`` is
        set to the default class verbosity ``self._verbose``. If the input argument ``verbose`` is negative
        then ``verbose_sub`` is set to ``0`` (``False``), otherwise it is set to ``verbose``.
        
        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Returns**

            ``[verbose,verbose_sub]``.
        """
        #global verbose
        if verbose is None:
            verbose_main: IntBool = self._verbose
            verbose_sub: IntBool = self._verbose
        elif verbose < 0:
            verbose_main = verbose
            verbose_sub = 0
        else:
            verbose_main = verbose
            verbose_sub = verbose
        return [verbose_main, verbose_sub]

def print(*objects: Any,
          sep: str =' ',
          end: str ='\n',
          file: Union[Text, Path, TextIO] = sys.stdout,
          flush: bool = False,
          show: Union[int,bool,None] = True
         ) -> None:
    """
    Redefinition of the built-in print function.
    It accepts an additional argument ``show`` allowing to switch print on and off.
    """
    if show is None:
        show = True
    if show:
        builtins.print(*objects, sep =' ', end ='\n', file = sys.stdout, flush = False)

