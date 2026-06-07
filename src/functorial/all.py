#
# Helper module to load all the names into the namespace
#
# Do 'from FP.all import *' to load all the needed objects.
#
# ruff: noqa: F401, F403, F405

from __future__ import annotations

from . import monoids
from . import optics

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

from .const       import *
from .either      import *
from .identity    import *
from .list        import *
from .maybe       import *
from .ntuple      import *
from .pair        import *

from .dict        import *
from .monoids     import Monoid, munit, mcombine
from .reader      import *
from .set         import *
from .state       import *
from .trees       import *
from .writer      import *

from .functions   import *
from .io          import *
from .ops         import *
from .utils       import *
from .wrappers    import *

from .pair        import pair  # More powerful version over .functions.pair

from .optics.all  import *

#
# Conveniences
#

c = compose
