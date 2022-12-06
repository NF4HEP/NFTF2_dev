import sys
from os import path

sys.dont_write_bytecode = True

from . import utils
from .utils.verbosity import print, Verbosity
from . import bijectors
from . import base
#from . import inputs
from .inputs import data, distributions
from . import nf
from .inputs.data import DataMain
from .nf import NFMain

#from . import base
#import bijectors
#import base
#import nf
#from . import bijectors
#from . import inputs
#from . import utils
#from . import nf
#from .utils.verbosity import print

#import bijectors
#import inputs
#import utils

#from . import base
#from . import inference
#from .base import Name, FileManager, ParsInput, ParsManager, Predictions, Figures, Inference, Plots
#from .inference import corner
#from .inference import inference
#from .inference import plots
#from .inputs import data
#from .inputs.distributions import Distributions
#from .utils.resources import Resources
#from .utils import utils
#from .utils.verbosity import Verbosity, print

#from .nf import NF, NFFileManager, NFParsManager, NFPredictions
#from .data import Data, DataFileManager, DataParsManager, DataPredictions
#
#from .cspline import CSplineNetwork, CSplineBijector, CSplineFlow
#from .realnvp import RealNVPNetwork, RealNVPBijector, RealNVPFlow
#from .rqspline import RQSplineNetwork_default, RQSplineNetwork_custom, RQSplineBijector_default, RQSplineBijector_custom, RQSplineFlow
#


#from .DNNLik import DNNLik
#from . import files

# Print strategy: most functions have an optional argument verbose. 
#   Warnings and errors always print. When verbose=0 no information is printed.
#   When verbose=-1 only information from the current function is printed, while
#   information from embedded functions is not printed. When verbose > 0 all information are
#   printed. Finally, for functions with different verbose modes (like tf.keras model.fit) the
#   verbose argument is passed to the function and set to zero if verbose = -1.
#   !!!!!!!!!!!!! Should review the ShowPrints strategy: check if it's useful or if we can get rid of it