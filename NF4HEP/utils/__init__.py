import sys
from os import path

sys.dont_write_bytecode = True

from . import corner
from . import custom_types
from . import resources
from . import utils
from . import verbosity

mplstyle_path = path.join(path.split(path.realpath(__file__))[0],"matplotlib.mplstyle")