# -*- coding: utf-8 -*-

from ._version import __version__

try:
    from .cditypes import *
    import atexit
    log_start(level="info")
    atexit.register(log_stop)

except:
    import logging
    logging.error("Could not load cuda libraries")
    
from .cditypes import *
from .processing import *
from .ptycho import *
from .misc import *
