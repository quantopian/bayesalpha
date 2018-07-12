from .model import fit_single, fit_population
from .author_model import fit_authors

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
