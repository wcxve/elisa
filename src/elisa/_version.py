import importlib.metadata as metadata

try:
    __version__: str = metadata.version('astro-elisa')
except metadata.PackageNotFoundError:
    __version__: str = 'dev'

__all__ = ['__version__']
