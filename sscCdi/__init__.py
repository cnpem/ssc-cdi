# -*- coding: utf-8 -*-

from ._version import __version__
from ._telem import __telem__

try:
    from .cditypes import *
except:
    import logging
    logging.error("Could not load cuda libraries")

import atexit
from .lib.ssccommons_wrapper import (
    log_event, log_start, log_stop, event_start, event_stop
)
log_start(project="sscCdi",
          version=__version__,
          level="error",
          telem_key=__telem__)
atexit.register(log_stop)

from .cditypes import *
from .processing import *
from .ptycho import *
from .misc import *
from .jupyter import *
from .beamlines import *
