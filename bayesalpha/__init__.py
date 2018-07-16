from .load import load
from .returns_model import fit_returns_single, fit_returns_population
from .author_model import fit_authors

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
