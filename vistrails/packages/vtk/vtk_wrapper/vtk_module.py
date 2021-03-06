"""Abstracts all VTK related modules into one module.  This makes it trivial to
support local VTK classes that a user may have built.

By default it imports all of VTK and then looks for a tvtk_local module and
imports everything from that.  In order to add local classes to the TVTK build
one may simply provide a tvtk_local.py module somewhere with any classes that
need to be wrapped.

"""

# Author: Prabhu Ramachandran <prabhu [at] aero.iitb.ac.in>
# Copyright (c) 2007,  Enthought, Inc.
# License: BSD Style.

from __future__ import division

from vtk import *

try:
    from tvtk_local import *
except ImportError:
    pass
