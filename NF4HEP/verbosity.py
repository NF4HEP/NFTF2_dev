import sys
import builtins
#from builtins import print
from pathlib import Path
from typing import Union, Any, TextIO, Text, Optional

IntBool = Union[int,bool]
#verbose = True
#def print(*args, **kwargs):
#    global verbose
#    if type(verbose) is bool:
#        if verbose:
#            return builtins.print(*args, **kwargs)
#    if type(verbose) is int:
#        if verbose != 0:
#            return builtins.print(*args, **kwargs)

class Verbosity():
    """
    Class inherited by all other classes to provide the 
    :meth:`Verbosity.set_verbosity <DNNLikelihood.Verbosity.set_verbosity>` method.
    """
    def __init__(self,
                 verbose: Optional[IntBool] = True) -> None:
                 self.verbose = True if verbose is None else verbose

    def set_verbosity(self, 
                      verbose: Optional[IntBool]) -> list[IntBool]:
        """
        Method inherited by all classes (from the :class:`Verbosity <DNNLikelihood.Verbosity>` class)
        used to set the verbosity mode. If the input argument ``verbose`` is ``None``, ``verbose`` is
        set to the default class verbosity ``self.verbose``. If the input argument ``verbose`` is negative
        then ``verbose_sub`` is set to ``0`` (``False``), otherwise it is set to ``verbose``.
        
        - **Arguments**

            - **verbose**
            
                See :argument:`verbose <common_methods_arguments.verbose>`.

        - **Returns**

            ``[verbose,verbose_sub]``.
        """
        #global verbose
        if verbose is None:
            verbose_main: IntBool = self.verbose
            verbose_sub: IntBool = self.verbose
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

