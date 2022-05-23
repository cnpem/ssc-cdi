# -*- coding: utf-8 -*-

try:
    import pkg_resources 
    __version__ = pkg_resources.require("sscCdi")[0].version
except:
    pass

from .cditypes import *
from .caterete import *
from .carnauba import *
