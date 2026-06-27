'''NeXus file loader — re-exported from nexus_io for backward compatibility.

All functionality lives in :mod:`nexus_io.loader`.  Import directly from
there in new code::

    from nexus_io import NeXusLoader, slice_by_coords
'''

from nexus_io import (  # noqa: F401
    NeXusLoader,
    coord_to_index,
    slice_by_coords,
    on_same_nav_grid,
    apply_nav,
)
from nexus_io.loader import (  # noqa: F401
    _nxdata_to_nxdata,
    _find_default_nxdata,
    _visit_nxdata,
    _collect_nxdata,
    _collect_scalars,
    _nx_class,
    _read_str,
    _read_strlist,
)
