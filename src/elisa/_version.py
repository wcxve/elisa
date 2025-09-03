import importlib.metadata as metadata

try:
    __version__: str = metadata.version(
        __package__.split('.', 1)[0] if __package__ else 'astro-elisa'
    )
except metadata.PackageNotFoundError:
    __version__: str = 'dev'

__all__ = ['__version__']
