"""Handle data loading."""
from __future__ import annotations

import os
import warnings

import numpy as np
import xarray as xr
from astropy.io import fits
from scipy.stats import norm

# __all__ = ['Data']


class Data:
    """Class to load data.

    Load the observation data and corresponding telescope response, and handle
    the grouping of spectrum and response matrix.

    Parameters
    ----------
    erange : list, or list of list
        Energy range of interested, e.g., ``erange=[(10., 20.), (50., 200.)]``.
    specfile : str
        Spectrum file path. For type II pha file, the row specifier must be in
        the end of path, e.g., ``specfile="./spec.phaii{1}"``.
    backfile : str or None, optional
        Background file path. Read from the spectrum header when it is None.
        For type II pha file, the row specifier must be in the end of path,
        e.g., ``backfile="./back.phaii{1}"``.
    respfile : str or None, optional
        Response file path. Read from the spectrum header when it is None.
        The path must be given if ``RESPFILE`` is undefined in the header.
    ancrfile : str or None, optional
        Ancillary response path. Read from the spectrum header when it is None.
    name : str or None, optional
        Data name. Read from the spectrum header when it is None. The name must
        be given when ``DETNAM``, ``INSTRUME`` and ``TELESCOP`` are all
        undefined in the spectrum header.
    group : str or None, optional
        Grouping method to be applied to the data.
    scale : float or None, optional
        Grouping scale to be applied to the data. Ignored when group is None.
    is_spec_poisson : bool or None, optional
        Whether the spectral data follows counting statistics. Read from the
        spectrum header when it is None. Must be given when ``POISSERR`` is
        undefined in the header.
    is_back_poisson : bool or None, optional
        Whether the background data follows counting statistics. Read from the
        background header when it is None. Must be given when ``POISSERR`` is
        undefined in the header.
    ignore_bad : bool, optional
        Whether to ignore channels whose ``QUALITY`` are 5.
        The default is True. The possible values for ``QUALITY`` are
            *  0: good
            *  1: defined bad by software
            *  2: defined dubious by software
            *  5: defined bad by user
            * -1: reason for bad flag unknown
    keep_channel_info : bool, optional
        Whether to record original channel information in grouped channel.
        The default is True. This only take effects when `group` is not None
        or spectral data has ``GROUPING`` defined.

    """

    def __init__(
        self,
        erange: list,
        specfile: str,
        backfile: str | None = None,
        respfile: str | None = None,
        ancrfile: str | None = None,
        name: str | None = None,
        group: str | None = None,
        scale: float | int | None = None,
        is_spec_poisson: bool | None = None,
        is_back_poisson: bool | None = None,
        ignore_bad: bool = True,
        keep_channel_info: bool = False
    ):
        ...


class Spectrum:
    def __init__(self, specfile):
        ...

    @property
    def groupping(self):
        return self._groupping

    @groupping.setter
    def groupping(self, value):
        self._groupping = value


class Response:
    def __init__(self, respfile, ancrfile):
        ...

    @property
    def groupping(self):
        return self._groupping

    @groupping.setter
    def groupping(self, value):
        self._groupping = value
