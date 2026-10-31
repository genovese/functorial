#
# Helper module to load all the names into the namespace
#
# Do 'from FP.all import *' to load all the needed objects.
#
# ruff: noqa: F401, F403, F405

from __future__ import annotations


#
# Submodules
#
from . import monoids
from . import optics


#
# Higher Order Types
#
from .alternative import *
from .applicative import *
from .bicofunctor import *
from .bifunctor   import *
from .cofunctor   import *
from .foldable    import *
from .functor     import *
from .monad       import *
from .profunctor  import *
from .traversable import *


#
# Concrete Functors, Applicatives, Monads
#
from .const       import *
from .dict        import *
from .either      import *
from .identity    import *
from .io          import *
from .list        import *
from .maybe       import *
from .monoids     import Monoid, munit, mcombine
from .ntuple      import *
from .pair        import *
from .reader      import *
from .resource    import *
from .set         import *
from .state       import *
from .trees       import *
from .writer      import *


#
# Utility and Infrastructure
#
from .functions   import *
from .ops         import *
from .utils       import *
from .wrappers    import *

from .pair        import pair  # More powerful version over .functions.pair


#
# Special Sublibraries
#
from .optics.all  import *


#
# Conveniences
#
c = compose
