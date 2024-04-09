# -*- coding: utf-8 -*-

from ._version import __version__

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
          level="info",
          telem_key="https://aa8e85a7f92d3fa14e2cab36d7a686ec@o1066143.ingest.us.sentry.io/4506592964116481")
atexit.register(log_stop)

from .cditypes import *
from .processing import *
from .ptycho import *
from .misc import *
from .jupyter import *
from .beamlines import *
