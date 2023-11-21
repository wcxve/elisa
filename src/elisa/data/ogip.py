"""Handle data loading."""
from __future__ import annotations

import os
import re
import warnings

import numpy as np
from astropy.io import fits

# __all__ = ['Data']
# TODO: support creating Data object from array


class Data:
    """Class to read information of an observation in OGIP/92-007 standard.

    Load the observation spectrum, the telescope response and the possible
    background, and handle the grouping of spectrum and response.

    Parameters
    ----------
    spec : Spectrum
        The :class:`Spectrum` instance containing spectrum data.
    resp : Response
        The :class:`Response` instance containing telescope response.
    back : Spectrum, optional
        The :class:`Spectrum` instance containing background data.

    """

    def __init__(
        self,
        spec: Spectrum,
        resp: Response,
        back: Spectrum | None = None
    ):
        self._spec = spec
        self._back = back
        self._resp = resp

        # TODO: warn on different quality flag of spec and back
        # TODO: merge backscale and areascale into exposure scale

    @classmethod
    def from_file(
        cls,
        erange: list,
        specfile: str,
        backfile: str | None = None,
        respfile: str | None = None,
        ancrfile: str | None = None,
        name: str | None = None,
        group: str | None = None,
        scale: float | int | None = None,
        poisson_spec: bool | None = None,
        poisson_back: bool | None = None,
        ignore_bad: bool = True,
        record_channel: bool = False,
        corrfile: str | None = None,
        corrnorm: str | None = None
    ) -> Data:
        """Load observation data, telescope response and possible background.

        Parameters
        ----------
        erange : array_like
            Energy range of interested, e.g., ``erange=[(0.5, 2), (5, 200)]``.
        specfile : str
            Spectrum file path. For type II pha file, the row specifier must be
            given in the end of path, e.g., ``specfile="./spec.pha2{1}"``.
        backfile : str or None, optional
            Background file path. Read from the `specfile` header if None.
            For type II pha file, the row specifier must be given in the end of
            path, e.g., ``backfile="./back.pha2{1}"``.
        respfile : str or None, optional
            Response file path. Read from the `specfile` header if None.
            The path must be given if ``RESPFILE`` is undefined in the header.
        ancrfile : str or None, optional
            Ancillary response path. Read from the `specfile` header if None.
        name : str or None, optional
            Data name. Read from the `specfile` header if None. The name must
            be given if ``DETNAM``, ``INSTRUME`` and ``TELESCOP`` are all
            undefined in the header.
        group : str or None, optional
            Grouping method to be applied to the spectrum and background.
        scale : float or None, optional
            Grouping scale to be applied. Only takes effect if `group` is not
            None.
        poisson_spec : bool or None, optional
            Whether the spectrum data follows counting statistics, reading from
            the `specfile` header. This value must be set if ``POISSERR`` is
            undefined in the header.
        poisson_back : bool or None, optional
            Whether the background data follows counting statistics, reading
            from the `backfile` header. This value must be set if ``POISSERR``
            is undefined in the header.
        ignore_bad : bool, optional
            Whether to ignore channels whose ``QUALITY`` are 5.
            The default is True. The possible values for ``QUALITY`` are
                *  0: good
                *  1: defined bad by software
                *  2: defined dubious by software
                *  5: defined bad by user
                * -1: reason for bad flag unknown
        record_channel : bool, optional
            Whether to record channel information in the label of grouped
            channel. Only takes effect if `group` is not None or spectral data
            has ``GROUPING`` defined. The default is True.
        corrfile : str or None, optional
            Correction file applied to `specfile`. Read from the `specfile`
            header if None. The default is None.
        corrnorm : float or None, optional
            Scaling factor to be applied to `corrfile`. Read from the
            `specfile` header if None. The default is None.

        """
        ...


class Spectrum:
    """Class to handle spectrum data in OGIP standard.

    Parameters
    ----------
    specfile : str
        Spectrum file path. For type II pha file, the row specifier must be
        given in the end of path, e.g., ``specfile="./spec.pha2{1}"``.
    name : str or None, optional
        Data name. Read from ``DETNAM``, ``INSTRUME`` or ``TELESCOP`` in the
        `specfile` header if None. If these are all empty, defaults to ``''``.
    ignore_bad : bool, optional
        Whether to ignore channels whose ``QUALITY`` are 5.
        The default is True. The possible values for ``QUALITY`` are
            *  0: good
            *  1: defined bad by software
            *  2: defined dubious by software
            *  5: defined bad by user
            * -1: reason for bad flag unknown
    poisson : bool or None, optional
        Whether the spectrum data follows counting statistics, reading from the
        `specfile` header. This value must be set if ``POISSERR`` is undefined
        in the header.

    """

    def __init__(
        self,
        specfile: str,
        name: str | None = None,
        ignore_bad: bool = True,
        poisson: bool | None = None
    ):
        # test if file is '/path/to/specfile{n}'
        match = re.compile(r'(.+){(\d+)}').match(specfile)
        if match:
            file = match.group(1)
            type_ii = True  # spectrum file is of type II
            spec_id = int(match.group(2)) - 1  # row specifier to index
        else:
            file = specfile
            type_ii = False
            spec_id = 0

        with fits.open(file) as hdul:
            header = hdul['SPECTRUM'].header
            data = hdul['SPECTRUM'].data

        # check if data is type II
        if not type_ii:
            msg = f'row id must be provided for type II spectrum {specfile}'

            if 'DETCHANS' in header:
                if int(header.get('DETCHANS')) != len(data):
                    raise ValueError(msg)

            elif header.get('HDUCLAS4', '') == 'TYPE:II':
                raise ValueError(msg)
        else:
            data = data[spec_id].array  # set data to the specified row

        # check if COUNTS or RATE exists
        if 'COUNTS' not in data.names and 'RATE' not in data.names:
            raise ValueError(f'"COUNTS" or "RATE" not found in {specfile}')

        # get poisson flag
        poisson = header.get('POISSERR', poisson)
        if poisson is None:
            msg = '`poisson` must be set if "POISSERR" is undefined in header'
            raise ValueError(msg)

        # check if STAT_ERR exists for non-Poisson spectrum
        if not poisson and 'STAT_ERR' not in data.names:
            raise ValueError(f'"STAT_ERR" not found in {specfile}')

        def get_field(field, default=None, excluded=None):
            """Get value of specified field, return default if not found."""
            if field in data.names:
                value = data[field]
                if type_ii:
                    value = value[0]
            else:
                value = header.get(field, default)

            if excluded is not None and value in excluded:
                return default
            else:
                return value

        # get exposure
        exposure = np.float64(get_field('EXPOSURE'))

        # get counts
        if 'COUNTS' in data.names:
            counts = get_field('COUNTS')
            counts = np.array(counts, dtype=np.float64, order='C')
        else:  # calculate counts using 'RATE' and 'EXPOSURE'
            rate = get_field('RATE')
            rate = np.array(rate, dtype=np.float64, order='C')
            counts = rate * exposure

        # get statistical error of counts
        if poisson:
            stat_err = np.sqrt(counts)
        else:
            stat_err = get_field('STAT_ERR')
            stat_err = np.array(stat_err, dtype=np.float64, order='C')
            if 'RATE' in data.names:
                stat_err *= exposure

                if 'COUNTS' in data.names:
                    msg = f'"STAT_ERR" in {specfile} is assumed for "RATE"'
                    warnings.warn(msg, Warning, stacklevel=2)

        # get fractional systematic error of counts
        sys_err = get_field('SYS_ERR', 0)
        if np.shape(sys_err) == () and sys_err == 0:
            sys_err = np.zeros(len(counts))
        else:
            sys_err = np.array(sys_err, dtype=np.float64, order='C')

        # get quality flag
        quality = get_field('QUALITY', np.zeros(len(counts)))
        if np.shape(quality) == () and quality == 0:
            quality = np.zeros(len(counts), dtype=np.int64)
        else:
            quality = np.array(quality, dtype=np.int64, order='C')

        # get grouping flag
        grouping = get_field('GROUPING', 0)
        if np.shape(grouping) == () and grouping == 0:
            grouping = np.ones(len(counts), np.int64)
        else:
            grouping = np.array(grouping, dtype=np.int64, order='C')

        if poisson:  # check if counts are integers
            diff = np.abs(counts - np.round(counts))
            if np.any(diff > 1e-8 * counts):
                msg = f'spectrum ({specfile}) has non-integer counts, '
                msg += 'which may lead to wrong result'
                warnings.warn(msg, Warning, stacklevel=2)

        # check if statistical error are positive
        mask = stat_err < 0.0
        if np.any(mask):
            stat_err[mask] = 0.0
            msg = f'spectrum ({specfile}) has some statistical errors < 0, '
            msg += 'which are reset to 0'
            warnings.warn(msg, Warning, stacklevel=2)

        # check if fractional systematic error are positive
        mask = sys_err < 0.0
        if np.any(mask):
            sys_err[mask] = 0.0
            msg = f'spectrum ({specfile}) has some systematic errors < 0, '
            msg += 'which are reset to 0'
            warnings.warn(msg, Warning, stacklevel=2)

        # search name in header
        if name is None:
            excluded_name = ('', 'none', 'unknown')
            for key in ('DETNAM', 'INSTRUME', 'TELESCOP'):
                name = str(header.get(key, ''))
                if name.lower() not in excluded_name:
                    break
                else:
                    name = ''
        self._name = str(name)

        excluded_file = ('none', 'None', 'NONE')

        # get backfile
        self._backfile = get_field('BACKFILE', '', excluded_file)

        # get respfile
        self._respfile = get_field('RESPFILE', '', excluded_file)

        # get ancrfile
        self._ancrfile = get_field('ANCRFILE', '', excluded_file)

        # get corrfile
        self._corrfile = get_field('CORRFILE', '', excluded_file)

        # get background scaling factor
        self._back_scale = np.float64(get_field('BACKSCAL', 1.0))

        # get area scaling factor
        area_scale = get_field('AREASCAL', 1.0)
        if isinstance(area_scale, np.ndarray):
            area_scale = np.array(area_scale, dtype=np.float64, order='C')
        self._area_scale = np.float64(area_scale)

        # get correction scaling factor
        self._corr_scale = np.float64(get_field('CORRSCAL', 0.0))

        self._header = header
        self._data = data
        self._counts = self.__counts = counts
        self._stat_err = self.__stat_err = stat_err
        self._sys_err = self.__sys_err = sys_err
        self._grouping = grouping
        self._exposure = exposure
        self._poisson = poisson
        self._quality = quality
        self._ignore_bad = ignore_bad

        self.group(grouping)

    def group(self, grouping: np.ndarray):
        """Group spectrum channel.

        Parameters
        ----------
        grouping : array_like
            Channel with a grouping flag of 1 with all successive channels with
            grouping flags of -1 are combined.

        Notes
        -----
        If there are some bad channels in a channel group, this may cause an
        inconsistency of in a spectral plot, i.e., the error bar of a channel
        group will cover these bad channels, whilst these bad channels are
        never used in fitting.

        """
        bad = (1, 5) if self._ignore_bad else (1,)
        good = ~np.isin(self._quality, bad)
        ...

    @property
    def counts(self) -> np.ndarray:
        """Counts in each measuring channel."""
        return self._counts

    @property
    def stat_err(self) -> np.ndarray:
        """Statistical uncertainty of counts in each measuring channel."""
        return self._stat_err

    @property
    def sys_err(self) -> np.ndarray:
        """Fractional systematic error of counts in each measuring channel."""
        return self._sys_err

    @property
    def grouping(self) -> np.ndarray:
        """Grouping flag for channel."""
        return self._grouping

    @property
    def quality(self) -> np.ndarray:
        """Quality flag indicating which channel to be used in analysis."""
        return self._quality

    @property
    def exposure(self) -> float:
        """Exposure time of the spectrum, in unit of second."""
        return self._exposure

    @property
    def poisson(self) -> bool:
        """Whether the spectrum data follows counting statistics."""
        return self._poisson

    @property
    def name(self) -> str:
        """``DETNAM``, ``INSTRUME`` or ``TELESCOP`` in `specfile` header."""
        return self._name

    @property
    def backfile(self) -> str:
        """Background file."""
        return self._backfile

    @property
    def respfile(self) -> str:
        """Response file."""
        return self._respfile

    @property
    def ancrfile(self) -> str:
        """Ancillary response file."""
        return self._ancrfile

    @property
    def corrfile(self) -> str:
        """Correction file."""
        return self._corrfile

    @property
    def back_scale(self) -> float:
        """Background scaling factor."""
        return self._back_scale

    @property
    def area_scale(self) -> float | np.ndarray:
        """Area scaling factor."""
        return self._area_scale

    @property
    def corr_scale(self) -> float:
        """Correction scaling factor."""
        return self._corr_scale


class Response:
    """Class to store and group telescope response.

    Parameters
    ----------
    respfile : str
        Response file path.
    ancrfile : str or None, optional
        Ancillary response path. The default is None.

    """

    def __init__(self, respfile: str, ancrfile: str | None = None):
        with fits.open(respfile) as rsp_hdul:
            ebounds = rsp_hdul['EBOUNDS'].data

            if 'MATRIX' in rsp_hdul:
                resp = rsp_hdul['MATRIX'].data
            elif 'SPECRESP MATRIX' in rsp_hdul:
                resp = rsp_hdul['SPECRESP MATRIX'].data

        channel = tuple(ebounds['CHANNEL'])

        # assume ph_egrid is continuous
        ph_egrid = np.append(resp['ENERG_LO'], resp['ENERG_HI'][-1])
        ph_egrid = np.asarray(ph_egrid, dtype=np.float64, order='C')
        ch_egrid = np.column_stack((ebounds['E_MIN'], ebounds['E_MAX']))
        ch_egrid = np.asarray(ch_egrid, dtype=np.float64, order='C')

        # extract response matrix
        matrix = resp['MATRIX']

        # wrap around N/A of matrix
        if matrix.dtype == np.dtype('O'):
            tmp = matrix.tolist()
            lens = np.array([len(i) for i in tmp])
            mask = np.arange(len(ch_egrid)) < lens[:, None]
            matrix = np.zeros(matrix.shape)
            matrix[mask] = np.concatenate(tmp, dtype=np.float64)

        if ancrfile:
            with fits.open(ancrfile) as arf_hdul:
                arf = arf_hdul['SPECRESP'].data['SPECRESP']

            if len(arf) != len(matrix):
                msg = f'rmf ({respfile}) and arf ({ancrfile}) are not matched'
                raise ValueError(msg)

            matrix *= arf[:, None]

        self._ph_egrid = ph_egrid
        self._channel = self.__channel = channel
        self._channel_egrid = self.__channel_egrid = ch_egrid
        self._matrix = self.__matrix = matrix

    def group(self, grouping: np.ndarray, good: np.ndarray | None = None):
        """Group response matrix.

        Parameters
        ----------
        grouping : array_like
            Channel with a grouping flag of 1 with all successive channels with
            grouping flags of -1 are combined.
        good : array_like or None, optional
            Quality flag indicating which channel to be used in analysis.

        Notes
        -----
        If there are some bad channels indicated in `good` in a channel group,
        this may cause an inconsistency of in a spectral plot, i.e., the error
        bar of a channel group will cover these bad channels, whilst these bad
        channels are never used in fitting.

        """
        l0 = len(self.__channel)

        if good is None:
            good = np.full(l0, True)

        l1 = len(grouping)
        l2 = len(good)
        if not l0 == l1 == l2:
            msg = f'length of grouping ({l1}) and good ({l2}) must match to '
            msg += f'original channel ({l0})'
            raise ValueError(msg)

        grouping = np.flatnonzero(grouping == 1)  # transform to index

        if len(grouping) == l0:  # case of no group, apply good mask
            self._channel = self.__channel[good]
            self._channel_egrid = self.__channel_egrid[good]
            self._matrix = self.__matrix[:, good]

        else:
            any_good_in_group = np.add.reduceat(good, grouping) != 0

            edge_indices = np.append(grouping, l0)
            channel = self.__channel
            emin, emax = self.__channel_egrid.T
            group_channel = []
            group_emin = []
            group_emax = []

            for i in range(len(grouping)):
                if not any_good_in_group[i]:
                    continue
                slice_i = slice(edge_indices[i], edge_indices[i + 1])
                quality_slice = good[slice_i]
                channel_slice = channel[slice_i]
                group_channel.append(channel_slice[quality_slice].astype(str))
                group_emin.append(min(emin[slice_i]))
                group_emax.append(max(emax[slice_i]))

            self._channel = tuple(map(lambda g: tuple(g), group_channel))
            self._channel_egrid = np.column_stack([group_emin, group_emax])

            to_zero = np.where(good, 1.0, 0.0)
            matrix = np.add.reduceat(self.__matrix * to_zero, grouping, axis=1)
            any_good_in_group = np.add.reduceat(good, grouping) != 0
            self._matrix = matrix[:, any_good_in_group]

    @property
    def ph_egrid(self) -> np.ndarray:
        """Photon energy grid."""
        return self._ph_egrid

    @property
    def channel(self) -> tuple:
        """Measured signal channel numbers."""
        return self._channel

    @property
    def ch_emin(self) -> np.ndarray:
        """Left edge of measured energy grid."""
        return self._channel_egrid[:, 0]

    @property
    def ch_emax(self) -> np.ndarray:
        """Right edge of measured energy grid."""
        return self._channel_egrid[:, 1]

    @property
    def ch_emid(self) -> np.ndarray:
        """Middle of measured energy grid."""
        return np.mean(self._channel_egrid, axis=1)

    @property
    def ch_width(self) -> np.ndarray:
        """Width of measured energy grid."""
        return self._channel_egrid[:, 1] - self._channel_egrid[:, 0]

    @property
    def ch_mean(self) -> np.ndarray:
        """Geometric mean of measured energy grid."""
        return np.sqrt(np.prod(self._channel_egrid, axis=1))

    @property
    def ch_error(self) -> np.ndarray:
        """Width between left/right and geometric mean of energy grid."""
        mean = self.ch_mean
        return np.abs([self.ch_emin - mean, self.ch_emax - mean])

    @property
    def matrix(self) -> np.ndarray:
        """Response matrix."""
        return self._matrix
