"""Domain plugins for physics-specific workflows.

Plugins stay above the reusable compute backends.  A new scientific domain can
register metadata and request builders here without changing the MPS or
statevector engines.
"""

from .registry import catalog, get_action, get_plugin, load_entry_points, register
from .fermion import map_fermion_terms
from .materials import HubbardMaterialsPlugin
from .spin_lattice import SpinLatticePlugin

register(SpinLatticePlugin())
register(HubbardMaterialsPlugin())

__all__ = ["catalog", "get_action", "get_plugin", "load_entry_points", "map_fermion_terms", "register"]
