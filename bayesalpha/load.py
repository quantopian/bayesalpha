import xarray as xr
from .returns_model import ReturnsModelResult
from .author_model import AuthorModelResult


def load(filename, group=None):
    trace = xr.open_dataset(filename, group=group)
    model_type = trace.attrs.get('model-type')
    if model_type == 'returns-model':
        return ReturnsModelResult._load(trace)
    elif model_type == 'author-model':
        return AuthorModelResult._load(trace)
    # Default to returns model, so we can still load old traces
    elif model_type is None:
        return ReturnsModelResult._load(trace)
    else:
        ValueError('Unknown model type: {}.'.format(model_type))
