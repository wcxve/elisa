"""Handle data loading."""
from __future__ import annotations

import os
import re
import warnings

import numpy as np
from astropy.io import fits

# __all__ = ['Data']


class Data:
    """Class to store observation data.

    Load the observation spectrum, the telescope response and the possible
    background, and handle the grouping of spectrum and response.

    Parameters
    ----------
    spec : Spectrum
        The :class:`Spectrum` instance containing spectrum data.
    resp : Response, optional
        The :class:`Response` instance containing telescope response.
    back : Spectrum
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
        record_channel: bool = False
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
            Background file path. Read from the spectrum header if None.
            For type II pha file, the row specifier must be given in the end of
            path, e.g., ``backfile="./back.pha2{1}"``.
        respfile : str or None, optional
            Response file path. Read from the spectrum header if None.
            The path must be given if ``RESPFILE`` is undefined in the header.
        ancrfile : str or None, optional
            Ancillary response path. Read from the spectrum header if None.
        name : str or None, optional
            Data name. Read from the spectrum header if None. The name must
            be given if ``DETNAM``, ``INSTRUME`` and ``TELESCOP`` are all
            undefined in the header.
        group : str or None, optional
            Grouping method to be applied to the spectrum and background.
        scale : float or None, optional
            Grouping scale to be applied. Only takes effect if `group` is not
            None.
        poisson_spec : bool or None, optional
            Whether the spectrum data follows counting statistics, which is
            first read from the spectrum header. This value will be used and
            must be set if ``POISSERR`` is undefined in the header.
        poisson_back : bool or None, optional
            Whether the background data follows counting statistics, which is
            first read from the background header. This value will be used and
            must be set if ``POISSERR`` is undefined in the header.
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

        """
        ...


class Spectrum:
    """Class to store spectrum data.

    Parameters
    ----------
    specfile : str
        Spectrum file path. For type II pha file, the row specifier must be
        given in the end of path, e.g., ``specfile="./spec.pha2{1}"``.
    name : str or None, optional
        Data name. Read from ``DETNAM``, ``INSTRUME`` or ``TELESCOP`` defined
        in the header if None. If all these field are empty, default to ``''``.
    ignore_bad : bool, optional
        Whether to ignore channels whose ``QUALITY`` are 5.
        The default is True. The possible values for ``QUALITY`` are
            *  0: good
            *  1: defined bad by software
            *  2: defined dubious by software
            *  5: defined bad by user
            * -1: reason for bad flag unknown
    poisson : bool or None, optional
        Whether the spectrum data follows counting statistics, which is first
        read from the spectrum header. This value will be used and must be set
        if ``POISSERR`` is undefined in the header.

    """

    def __init__(
        self,
        specfile: str,
        name: str = '',
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

        if not os.path.exists(file):
            raise FileNotFoundError(file)

        with fits.open(file) as hdul:
            header = hdul['SPECTRUM'].header
            data = hdul['SPECTRUM'].data

        poisson = header.get('POISSERR', poisson)
        if poisson is None:
            msg = '`poisson` must be set if "POISSERR" is undefined in header'
            raise ValueError(msg)

        if poisson and 'STAT_ERR' not in data.names:
            raise ValueError(f'"STAT_ERR" not found in {specfile}')

        if type_ii:
            data = data[spec_id].array
            exposure = float(data['EXPOSURE'])
        else:
            exposure = float(header['EXPOSURE'])

        if 'COUNTS' in data.names:
            counts = np.array(data['COUNTS'], dtype=np.float64, order='C')

            if poisson:
                error = np.sqrt(counts)
            else:
                error = np.array(data['STAT_ERR'], dtype=np.float64, order='C')

        elif 'RATE' in data.names:
            rate = np.array(data['RATE'], dtype=np.float64, order='C')
            counts = rate * exposure

            if poisson:
                error = np.sqrt(counts)
            else:
                error = np.array(data['STAT_ERR'], dtype=np.float64, order='C')
                error *= exposure
        else:
            raise ValueError(f'"COUNTS" or "RATE" not found in {specfile}')

        if 'QUALITY' in data.names:
            quality = np.array(data['QUALITY'], dtype=np.int64, order='C')
        else:
            quality = np.zeros(len(counts), dtype=np.int64)

        if 'GROUPING' in data.names:
            grouping = np.flatnonzero(data['GROUPING'] == 1)
        else:
            grouping = np.arange(len(counts), dtype=np.int64)

        if poisson:  # check if counts are integers
            diff = np.abs(counts - np.round(counts))
            if np.any(diff > 1e-8 * counts):
                msg = f'spectrum ({specfile}) has non-integer counts, '
                msg += 'which may lead to wrong result'
                warnings.warn(msg, stacklevel=2)

        else:  # check if error are positive
            mask = error <= 0.0
            if np.any(mask):
                error[mask] = 1e-10
                msg = f'spectrum ({specfile}) has non-positive errors, '
                msg += 'which are reset to 1e-10'
                warnings.warn(msg, stacklevel=2)

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

        # get backfile
        if type_ii and 'BACKFILE' in data.names:
            self._backfile = data['BACKFILE']
        else:
            self._backfile = header.get('BACKFILE', '')

        # get respfile
        if type_ii and 'RESPFILE' in data.names:
            self._respfile = data['RESPFILE']
        else:
            self._respfile = header.get('RESPFILE', '')

        # get ancrfile
        if type_ii and 'ANCRFILE' in data.names:
            self._ancrfile = data['ANCRFILE']
        else:
            self._ancrfile = header.get('ANCRFILE', '')

        self._header = header
        self._data = data
        self._counts = self.__counts = counts
        self._error = self.__error = error
        self._grouping = self.__grouping = grouping
        self._exposure = exposure
        self._poisson = poisson
        self._quality = quality

        bad = (1, 5) if ignore_bad else (1,)
        good = ~np.isin(quality, bad)
        self.group(grouping, good)

    def group(self, grouping: np.ndarray, good: np.ndarray | None = None):
        """Group spectrum channel.

        Parameters
        ----------
        grouping : array_like
            Channel with a grouping flag of 1 with all successive channels with
            grouping flags of -1 are combined.
        good : array_like or None, optional
            Quality flag indicating which channel to be used in analysis.

        """
        ...

    @property
    def counts(self) -> np.ndarray:
        """Counts in each measured channel."""
        return self._counts

    @property
    def error(self) -> np.ndarray:
        """Uncertainty of counts in each measured channel."""
        return self._error

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
        """``DETNAM``, ``INSTRUME`` or ``TELESCOP`` defined in header."""
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

        # self._ph_channel = tuple(range(len(matrix)))
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
        In a channel group, if there are some bad channels indicated in `good`,
        this may cause an inconsistency of spectral plot, i.e., the error bar
        of a channel group will cover these bad channels, whilst these bad
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

    # @property
    # def ph_channel(self) -> tuple:
    #     """Photon channel numbers."""
    #     return self._ph_channel

    @property
    def ph_egrid(self) -> np.ndarray:
        """Photon egrid."""
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
