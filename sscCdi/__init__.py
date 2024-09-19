# -*- coding: utf-8 -*-

from ._version import __version__

try:
    from .cditypes import *
    from .cditypes_planewave import *
    import atexit
    log_start(level="error")
    atexit.register(log_stop)

except:
    import logging
    logging.error("Could not load cuda libraries")
    
from .cditypes import *
from .cditypes_planewave import *
from .processing import *
from .ptycho import *
from .misc import *
from .beamlines.gui_jupyter import *
from .beamlines import *
