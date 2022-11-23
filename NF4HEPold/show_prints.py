import builtins

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
    :meth:`Verbosity.set_verbosity <NF4HEP.Verbosity.set_verbosity>` method.
    """
    def set_verbosity(self, verbose):
        """
        Method inherited by all classes (from the :class:`Verbosity <NF4HEP.Verbosity>` class)
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
        if verbose == None:
            verbose = self.verbose
        if verbose < 0:
            verbose_sub = 0
        else:
            verbose_sub = verbose
        return [verbose, verbose_sub]

def print(*args, **kwargs):
    """
    Redefinition of the built-in print function.
    It accepts an additional argument ``show`` allowing to switch print on and off.
    """
    try:
        show = kwargs.pop("show")
    except:
        # default show=True
        show = True
    if show:
        return builtins.print(*args, **kwargs)
