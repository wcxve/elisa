"""Containers of OGIP/92-007 format data."""
from __future__ import annotations

import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

from elisa.data.grouping import (
    group_const,
    group_min,
    group_opt,
    group_optmin,
    group_optsig,
    group_pos,
    group_sig,
)
from elisa.util.typing import NumPyArray as NDArray

# TODO: support multiple response in a single data object
# TODO: support creating Data object from array


class Data:
    """Handle observation data in OGIP standards [1]_ [2]_.

    Load the observation spectrum, the telescope response and the possible
    background, and handle the grouping of spectrum and response.

    Parameters
    ----------
    erange : array_like
        Energy range of interest in keV, e.g., ``erange=[(0.5, 2), (5, 200)]``.
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
        Method to group spectrum and background adaptively, these options are
        available so that each channel group has:

            * ``'const'``: `scale` number channels
            * ``'min'``: total (source + background) counts >= `scale`
            * ``'sig'``: source significance >= `scale` sigma
            * ``'opt'``: optimal binning, see Kaastra & Bleeker (2016) [3]_
            * ``'optmin'``: optimal binning with total counts >= `scale`
            * ``'optsig'``: optimal binning with source significance >= `scale`
              sigma
            * ``'bmin'``: background counts >= `scale`, used to avoid bias when
              using ``wstat`` to simultaneously fit the source and background
            * ``'bpos'``: background counts < 0 with probability < `scale`,
              used to avoid bias when using ``pgstat`` to simultaneously fit
              the source and background

        The default is None.
    scale : float or None, optional
        Grouping scale for the method specified in `group`.
    spec_poisson : bool or None, optional
        Whether the spectrum data follows counting statistics, reading from
        the `specfile` header. This value must be set if ``POISSERR`` is
        undefined in the header.
    back_poisson : bool or None, optional
        Whether the background data follows counting statistics, reading
        from the `backfile` header. This value must be set if ``POISSERR``
        is undefined in the header.
    ignore_bad : bool, optional
        Whether to ignore channels with ``QUALITY==5``.
        The default is True. The possible values for spectral ``QUALITY`` are

            * ``0``: good
            * ``1``: defined bad by software
            * ``2``: defined dubious by software
            * ``5``: defined bad by user
            * ``-1``: reason for bad flag unknown

    record_channel : bool, optional
        Whether to record channel information in the label of grouped
        channel. Only takes effect if `group` is not None or spectral data
        has ``GROUPING`` defined. The default is False.
    corrfile : str or None, optional
        Correction file applied to `specfile`. Read from the `specfile`
        header if None. The default is None.
    corrnorm : float or None, optional
        Scaling factor to be applied to `corrfile`. Read from the
        `specfile` header if None. The default is None.

    Notes
    -----
    Reading and applying correction to data is not yet supported.

    References
    ----------
    .. [1] `The OGIP Spectral File Format <https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/ogip_92_007.html>`__
            and `Addendum: Changes log <https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007a/ogip_92_007a.html>`__
    .. [2] `The Calibration Requirements for Spectral Analysis (Definition of
            RMF and ARF file formats) <https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html>`__
            and `Addendum: Changes log <https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002a/cal_gen_92_002a.html>`__
    .. [3] `Kaastra & Bleeker 2016, A&A, 587, A151 <https://doi.org/10.1051/0004-6361/201527395>`__

    """

    def __init__(
        self,
        erange: list | tuple,
        specfile: str,
        backfile: str | None = None,
        respfile: str | None = None,
        ancrfile: str | None = None,
        name: str | None = None,
        group: str | None = None,
        scale: float | int | None = None,
        spec_poisson: bool | None = None,
        back_poisson: bool | None = None,
        ignore_bad: bool = True,
        record_channel: bool = False,
        corrfile: bool | None = None,
        corrnorm: bool | None = None,
    ):
        erange = np.array(erange, dtype=np.float64, order='C', ndmin=2)

        # check if erange is increasing
        if np.any(np.diff(erange, axis=1) <= 0.0):
            raise ValueError('erange must be increasing')

        # check if erange is overlapped
        erange = erange[erange[:, 0].argsort()]
        if np.any(np.diff(np.hstack(erange)) <= 0.0):
            raise ValueError('erange must not be overlapped')

        spec = Spectrum(specfile, spec_poisson)

        # check data name
        if name:
            name = str(name)
        elif spec.name:
            name = spec.name
        else:
            raise ValueError('name is required for data')

        # check ancillary response file
        if not ancrfile:
            ancrfile = spec.ancrfile

        # check response file
        if respfile:
            resp = Response(respfile, ancrfile)
        elif spec.respfile:
            resp = Response(spec.respfile, ancrfile)
        else:
            raise ValueError('respfile is required for data')

        if len(spec._raw_counts) != len(resp._raw_channel):
            raise OSError(
                f'specfile ({specfile}) and respfile ({respfile}) are not '
                'matched'
            )

        # check background file
        if backfile:
            back = Spectrum(backfile, back_poisson)
        elif spec.backfile:
            back = Spectrum(spec.backfile, back_poisson)
        else:
            back = None

        if back and len(spec._raw_counts) != len(back._raw_counts):
            raise OSError(
                f'specfile ({specfile}) and backfile ({backfile}) are not '
                'matched'
            )

        # bad quality
        bad = (1, 5) if ignore_bad else (1,)

        # check if the quality of spectrum and background are matched
        good_quality = ~np.isin(spec.quality, bad)
        if back:
            back_good = ~np.isin(back.quality, bad)
            if not np.all(good_quality == back_good):
                good_quality &= back_good
                warnings.warn(
                    'ignore bad channels defined by the union of spectrum '
                    f'({specfile})and background ({backfile}) quality',
                    Warning,
                    stacklevel=2,
                )

        # corrfile and corrnorm are not supported yet
        if corrfile or corrnorm:
            warnings.warn(
                'correction to data is not yet supported',
                Warning,
                stacklevel=2,
            )

        # check correction file
        # use poisson=True to bypass stat_err check, which takes no effect
        # if corrfile:
        #     corr = Spectrum(corrfile, True)
        # elif spec.corrfile:
        #     corr = Spectrum(spec.corrfile, True)
        # else:
        #     corr = None

        # check correction scale
        # if corr:
        #     if corrnorm:
        #         spec._corr_scale = corrnorm

        self._spec = spec
        self._resp = resp
        self._back = back
        # self._corr = corr

        self._name = name
        self._erange = erange
        self._good_quality = good_quality
        self._record_channel = bool(record_channel)

        # response attributes
        self._ph_egrid = resp.ph_egrid
        self._channel = None
        self._ch_emin = None
        self._ch_emax = None
        self._ch_emid = None
        self._ch_mean = None
        self._ch_width = None
        self._ch_error = None
        self._resp_matrix = None

        # spectrum attributes
        self._spec_exposure = spec.exposure
        self._spec_poisson = spec.poisson
        self._spec_counts = None
        self._spec_error = None

        self._has_back = True if back else False

        # background attributes
        if self._has_back:
            self._back_exposure = back.exposure
            self._back_poisson = back.poisson
            self._back_ratio = (
                spec.exposure * spec.area_scale * spec.back_scale
            ) / (back.exposure * back.area_scale * back.back_scale)
        else:
            self._back_exposure = None
            self._back_poisson = None
            self._back_ratio = None
        self._back_counts = None
        self._back_error = None

        # net spectrum attributes
        self._net_counts = None
        self._net_error = None
        self._ce = None
        self._ce_error = None

        # other attributes
        self._grouping = None
        self._ch_mask = None

        if group:
            # group spectrum and set the other attributes therein
            self.group(group, scale)
        else:
            # set the other attributes
            self._set_data(spec.grouping)

    def group(self, method: str, scale: float | int):
        """Group the spectrum.

        Parameters
        ----------
        method : str
            Method to group spectrum and background adaptively, these options
            are available so that each channel group has:

            * ``'const'``: `scale` number channels
            * ``'min'``: total (source + background) counts >= `scale`
            * ``'sig'``: source significance >= `scale` sigma
            * ``'opt'``: optimal binning, see Kaastra & Bleeker (2016) [3]_
            * ``'optmin'``: optimal binning with total counts >= `scale`
            * ``'optsig'``: optimal binning with source significance >= `scale`
              sigma
            * ``'bmin'``: background counts >= `scale`, used to avoid bias when
              using ``wstat`` to simultaneously fit the source and background
            * ``'bpos'``: background counts < 0 with probability < `scale`,
              used to avoid bias when using ``pgstat`` to simultaneously fit
              the source and background

        scale : float
            Grouping scale.

        Warns
        -----
        GroupWarning
            Warn if grouping scale is not met for any channel.

        Raises
        ------
        NotImplementedError
            Grouping is not yet implemented for spectrum with ``AREASCAL``
            and/or ``BACKSCAL`` array.

        Notes
        -----
        If there are ignored channels in a channel group, this may cause an
        inconsistency in a spectral plot. That is to say, the error bar of a
        channel group will cover these bad channels, whilst these bad channels
        are never used in fitting.

        References
        ----------
        .. [1] `Kaastra & Bleeker 2016, A&A, 587, A151 <https://doi.org/10.1051/0004-6361/201527395>`__

        """
        ch_emin, ch_emax = self._resp._raw_channel_egrid.T
        ch_mask = self._channel_mask(ch_emin, ch_emax)  # shape = (nchan, 2)
        spec_counts = self._spec._raw_counts
        # spec_error = self._spec._raw_error
        grouping = np.full(len(spec_counts), 1, dtype=np.int64)

        def apply_grouping(group_func, mask, *args):
            """function operating the grouping array defined above."""
            data = (i[mask] * self._good_quality[mask] for i in args)
            grouping_flag, grouping_success = group_func(*data, float(scale))
            grouping[mask] = grouping_flag
            return grouping_success

        def apply_map(func, *args):
            """map the apply function and return a success flag."""
            return all(apply_grouping(func, mask, *args) for mask in ch_mask)

        if method == 'const':
            success = group_const()

        elif method == 'min':
            success = apply_map(group_min, spec_counts)

        elif method == 'sig':
            success = group_sig()

        elif method == 'opt':
            success = group_opt()

        elif method == 'optmin':
            success = group_optmin()

        elif method == 'optsig':
            success = group_optsig()

        elif method == 'bmin':
            if self.has_back and self.back_poisson:
                back_counts = self._back._raw_counts
            else:
                raise ValueError(
                    'Poisson background is required for "bmin" method'
                )
            success = apply_map(group_min, back_counts)

        elif method == 'bpos':
            if self.has_back:
                back_counts = self._back._raw_counts
                back_error = self._back._raw_error
            else:
                raise ValueError(
                    'background data is required for "bpos" method'
                )
            success = apply_map(group_pos, back_counts, back_error)

        else:
            supported = (
                'const',
                'min',
                'sig',
                'opt',
                'optmin',
                'optsig',
                'bmin',
                'bpos',
            )
            raise ValueError(
                f'supported grouping method are: {", ".join(supported)}'
            )

        if not success:
            warnings.warn(
                f'"{method}" grouping failed in some {self._name} channels',
                GroupingWaring,
            )

        self._set_data(grouping)

    def _set_data(self, grouping: NDArray):
        """Set data according to quality, grouping, and energy range."""
        self._spec.group(grouping, self._good_quality)
        self._resp.group(grouping, self._good_quality)
        self._grouping = grouping

        ch = self._spec._header.get('CHANTYPE', 'Ch')
        if self._record_channel:
            groups_channel = np.array(
                [f'{self.name}_{ch}_{"+".join(c)}' for c in self._resp.channel]
            )
        else:
            grp_idx = np.flatnonzero(grouping == 1)  # transform to index
            non_empty = np.add.reduceat(self._good_quality, grp_idx) != 0
            groups_channel = np.array(
                [f'{self.name}_{ch}_{c}' for c in np.flatnonzero(non_empty)]
            )

        ch_emin = self._resp.ch_emin
        ch_emax = self._resp.ch_emax
        ch_mask = self._channel_mask(ch_emin, ch_emax)  # shape = (nchan, 2)
        ch_mask = np.any(ch_mask, axis=0)

        # response attribute
        self._channel = groups_channel[ch_mask]
        self._ch_emin = ch_emin[ch_mask]
        self._ch_emax = ch_emax[ch_mask]
        resp = self._resp
        self._ch_emid = resp.ch_emid[ch_mask]
        self._ch_mean = resp.ch_mean[ch_mask]
        self._ch_width = resp.ch_width[ch_mask]
        self._ch_error = resp.ch_error[:, ch_mask]
        self._resp_matrix = resp.matrix[:, ch_mask]

        # spectrum attribute
        spec = self._spec
        self._spec_counts = spec.counts[ch_mask]
        self._spec_error = spec.error[ch_mask]

        # background attribute
        if self._has_back:
            self._back.group(grouping, self._good_quality)
            back = self._back
            self._back_counts = back.counts[ch_mask]
            self._back_error = back.error[ch_mask]

        # net spectrum attribute
        unit = 1.0 / (self._ch_width * self._spec_exposure)
        if self._has_back:
            net = self._spec_counts - self._back_ratio * self._back_counts
            var = np.square(self._spec_error)
            var += np.square(self._back_ratio * self._back_error)
            net_error = np.sqrt(var)
            ce = net * unit
            ce_error = net_error * unit
            self._net_counts = net
            self._net_error = net_error
            self._ce = ce
            self._ce_error = ce_error

        else:
            self._net_counts = self._spec_counts
            self._net_error = self._spec_error
            self._ce = self._net_counts * unit
            self._ce_error = self._spec_error * unit

        # correction attribute
        # if self._corr:
        #     self._corr.group(grouping, self._good_quality)

    def _channel_mask(self, ch_emin: NDArray, ch_emax: NDArray) -> NDArray:
        """Return channel mask given energy grid and erange of interest."""
        emin = np.expand_dims(self._erange[:, 0], axis=1)
        emax = np.expand_dims(self._erange[:, 1], axis=1)
        mask1 = np.less_equal(emin, ch_emin)
        mask2 = np.less_equal(ch_emax, emax)
        return np.bitwise_and(mask1, mask2)

    def plot_matrix(self, hatch: bool = True) -> None:
        """Plot the response matrix.

        Parameters
        ----------
        hatch : bool, optional
            Whether to hatch the ignored region. The default is True.

        """
        self._resp.plot(self._erange if hatch else None)

    @property
    def name(self) -> str:
        """Data name."""
        return self._name

    @property
    def spec_counts(self) -> NDArray:
        """Spectrum counts in each measuring channel."""
        return self._spec_counts

    @property
    def spec_error(self) -> NDArray:
        """Uncertainty of spectrum counts."""
        return self._spec_error

    @property
    def spec_poisson(self) -> bool:
        """Whether spectrum data follows counting statistics."""
        return self._spec_poisson

    @property
    def spec_exposure(self) -> float:
        """Spectrum exposure."""
        return self._spec_exposure

    @property
    def area_factor(self) -> float | NDArray:
        """Area scaling factor."""
        return self._spec.area_scale

    @property
    def has_back(self) -> bool:
        """Whether spectrum data includes background."""
        return self._has_back

    @property
    def back_counts(self) -> NDArray | None:
        """Background counts in each measuring channel."""
        return self._back_counts

    @property
    def back_error(self) -> NDArray | None:
        """Uncertainty of background counts."""
        return self._back_error

    @property
    def back_poisson(self) -> bool | None:
        """Whether background data follows counting statistics."""
        return self._back_poisson

    @property
    def back_exposure(self) -> float | None:
        """Background exposure."""
        return self._back_exposure

    @property
    def back_ratio(self) -> float | NDArray | None:
        """Ratio of spectrum to background effective exposure."""
        return self._back_ratio

    @property
    def net_counts(self) -> NDArray:
        """Net counts in each measuring channel."""
        return self._net_counts

    @property
    def net_error(self) -> NDArray:
        """Uncertainty of net counts in each measuring channel."""
        return self._net_error

    @property
    def ce(self) -> NDArray:
        """Net counts per second per keV."""
        return self._ce

    @property
    def ce_error(self) -> NDArray:
        """Uncertainty of net counts per second per keV."""
        return self._ce_error

    @property
    def ph_egrid(self) -> NDArray:
        """Photon energy grid of response matrix."""
        return self._ph_egrid

    @property
    def channel(self) -> NDArray:
        """Measured channel information."""
        return self._channel

    @property
    def ch_emin(self) -> NDArray:
        """Left edge of measured energy grid."""
        return self._ch_emin

    @property
    def ch_emax(self) -> NDArray:
        """Right edge of measured energy grid."""
        return self._ch_emax

    @property
    def ch_emid(self) -> NDArray:
        """Middle of measured energy grid."""
        return self._ch_emid

    @property
    def ch_width(self) -> NDArray:
        """Width of measured energy grid."""
        return self._ch_width

    @property
    def ch_mean(self) -> NDArray:
        """Geometric mean of measured energy grid."""
        return self._ch_mean

    @property
    def ch_error(self) -> NDArray:
        """Width between left/right and geometric mean of channel grid."""
        return self._ch_error

    @property
    def resp_matrix(self) -> NDArray:
        """Response matrix."""
        return self._resp_matrix


class Spectrum:
    """Handle spectral data in OGIP standards [1]_.

    Parameters
    ----------
    specfile : str
        Spectrum file path. For type II pha file, the row specifier must be
        given in the end of path, e.g., ``specfile="./spec.pha2{1}"``.
    poisson : bool or None, optional
        Whether the spectrum data follows counting statistics, reading from
        the `specfile` header. This value must be set if ``POISSERR`` is
        undefined in the header.

    References
    ----------
    .. [1] `The OGIP Spectral File Format <https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007/ogip_92_007.html>`__
            and `Addendum: Changes log <https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/docs/spectra/ogip_92_007a/ogip_92_007a.html>`__

    """

    def __init__(self, specfile: str, poisson: bool | None = None):
        # test if file is '/path/to/specfile{n}'
        match = re.compile(r'(.+){(.+)}').match(specfile)
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

        # TODO: more robust way to detect a type II data
        # check if data is type II
        if not type_ii:
            msg = f'row id must be provided for type II spectrum {specfile}'

            nchan = len(data)
            if int(header.get('DETCHANS', nchan)) != nchan:
                raise OSError(msg)

            if header.get('HDUCLAS4', '') == 'TYPE:II':
                raise OSError(msg)

        else:
            data = data[spec_id].array  # set data to the specified row

        # check if COUNTS or RATE exists
        if 'COUNTS' not in data.names and 'RATE' not in data.names:
            raise OSError(f'"COUNTS" or "RATE" not found in {specfile}')

        # get poisson flag
        poisson = header.get('POISSERR', poisson)
        if poisson is None:
            raise ValueError(
                '`poisson` must be set if "POISSERR" is undefined in header'
            )

        # check if STAT_ERR exists for non-Poisson spectrum
        if not poisson and 'STAT_ERR' not in data.names:
            raise OSError(f'"STAT_ERR" not found in {specfile}')

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
                    warnings.warn(
                        f'"STAT_ERR" in {specfile} is assumed for "RATE"',
                        Warning,
                        stacklevel=3,
                    )

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
                warnings.warn(
                    f'spectrum ({specfile}) has non-integer counts, '
                    'which may lead to wrong result',
                    Warning,
                    stacklevel=3,
                )

        # check if statistical errors are positive
        if not poisson and np.any(stat_err <= 0.0):
            raise OSError(f'spectrum ({specfile}) has statistical errors <= 0')

        # check if systematic errors are non-negative
        if np.any(sys_err < 0.0):
            raise OSError(f'spectrum ({specfile}) has systematic errors < 0')

        # total error of counts
        stat_var = np.square(stat_err)
        sys_var = np.square(sys_err * counts)
        error = np.sqrt(stat_var + sys_var)

        # search name in header
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

        # get the background scaling factor
        back_scale = np.float64(get_field('BACKSCAL', 1.0))
        if isinstance(back_scale, NDArray):
            back_scale = np.array(back_scale, dtype=np.float64, order='C')
        else:
            back_scale = np.float64(back_scale)
        self._back_scale = back_scale

        # get the area scaling factor
        area_scale = get_field('AREASCAL', 1.0)
        if isinstance(area_scale, NDArray):
            area_scale = np.array(area_scale, dtype=np.float64, order='C')
        else:
            area_scale = np.float64(area_scale)
        self._area_scale = area_scale

        # get the correction scaling factor
        self._corr_scale = np.float64(get_field('CORRSCAL', 0.0))

        self._header = header
        self._data = data
        self._counts = self._raw_counts = counts
        self._error = self._raw_error = error
        self._grouping = grouping
        self._exposure = exposure
        self._poisson = poisson
        self._quality = quality

    def group(self, grouping: NDArray, noticed: NDArray | None):
        """Group spectrum channel.

        Parameters
        ----------
        grouping : ndarray
            Channel with a grouping flag of 1 with all successive channels
            with grouping flags of -1 are combined.
        noticed : ndarray or None, optional
            Flag indicating which channel to be used in grouping.

        Raises
        ------
        NotImplementedError
            Grouping is not yet implemented for spectrum with ``AREASCAL``
            and/or ``BACKSCAL`` array.

        Notes
        -----
        If there are ignored channels in a channel group, this may cause an
        inconsistency in a spectral plot. That is to say, the error bar of a
        channel group will cover these bad channels, whilst these bad channels
        are never used in fitting.

        """
        # TODO:
        #   * area_scale array grouping info can be hardcode into grouped
        #     response matrix when calculating model
        #   * back_scale array grouping info can be ...? using average?
        #   * net counts and model folding calculation should be re-implemented
        #     in likelihood and helper modules
        if not () == np.shape(self.area_scale) == np.shape(self.back_scale):
            raise NotImplementedError(
                'grouping is not implemented yet for the spectrum with '
                '``AREASCAL`` and/or ``BACKSCAL`` array'
            )

        l0 = len(self._raw_counts)
        if noticed is None:
            noticed = np.full(l0, True)
        else:
            l1 = len(grouping)
            l2 = len(noticed)
            if not l0 == l1 == l2:
                raise ValueError(
                    f'length of grouping ({l1}) and noticed ({l2}) must be '
                    f'matched to spectrum channel ({l0})'
                )

            noticed = np.array(noticed, dtype=bool)

        factor = noticed.astype(np.float64)
        grp_idx = np.flatnonzero(grouping == 1)  # transform to index
        non_empty = np.add.reduceat(factor, grp_idx) != 0

        counts = np.add.reduceat(factor * self._raw_counts, grp_idx)
        counts = counts[non_empty]

        var = factor * self._raw_error * self._raw_error
        var = np.add.reduceat(var, grp_idx)[non_empty]
        error = np.sqrt(var)

        self._counts = counts
        self._error = error

    @property
    def counts(self) -> NDArray:
        """Counts in each measuring channel."""
        return self._counts

    @property
    def error(self) -> NDArray:
        """Uncertainty of counts in each measuring channel."""
        return self._error

    @property
    def grouping(self) -> NDArray:
        """Grouping flag for channel."""
        return self._grouping

    @property
    def quality(self) -> NDArray:
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
    def back_scale(self) -> float | NDArray:
        """Background scaling factor."""
        return self._back_scale

    @property
    def area_scale(self) -> float | NDArray:
        """Area scaling factor."""
        return self._area_scale

    @property
    def corr_scale(self) -> float:
        """Correction scaling factor."""
        return self._corr_scale


class Response:
    """Handle telescope response in OGIP standards [1]_.

    Parameters
    ----------
    respfile : str
        Response file path.
    ancrfile : str or None, optional
        Ancillary response path. The default is None.

    References
    ----------
    .. [1] `The Calibration Requirements for Spectral Analysis (Definition of
            RMF and ARF file formats) <https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html>`___
            and `Addendum: Changes log <https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002a/cal_gen_92_002a.html>`__

    """

    def __init__(self, respfile: str, ancrfile: str | None = None):
        if ancrfile is None:
            ancrfile = ''

        self._respfile = respfile
        self._ancrfile = ancrfile

        # test if file is '/path/to/respfile{n}'
        match = re.compile(r'(.+){(.+)}').match(respfile)
        if match:  # respfile file is of type II
            file = match.group(1)
            resp_id = int(match.group(2))
        else:
            file = respfile
            resp_id = 1

        with fits.open(file) as rsp_hdul:
            if 'MATRIX' in rsp_hdul:
                if ancrfile is None:
                    warnings.warn(
                        f'{file} is probably a rmf, '
                        'ancrfile (arf) maybe needed but not provided',
                        Warning,
                    )

                ext = ('MATRIX', resp_id)

            elif 'SPECRESP MATRIX' in rsp_hdul:
                ext = ('SPECRESP MATRIX', resp_id)

            else:
                raise OSError(
                    f'cannot read response matrix data from {respfile}'
                )

            self._read_ebounds(rsp_hdul['EBOUNDS'].data)
            self._read_resp(rsp_hdul[ext].header, rsp_hdul[ext].data)

        self._read_ancrfile()
        # self._drop_zeros()

    def _read_ebounds(self, ebounds_data):
        ch_emin = ebounds_data['E_MIN']
        ch_emax = ebounds_data['E_MAX']
        if np.any(ch_emin > ch_emax):
            raise OSError(
                f'respfile ({self._respfile}) channel energy grids are not '
                'increasing'
            )
        ch_egrid = np.column_stack((ch_emin, ch_emax))
        ch_egrid = np.asarray(ch_egrid, dtype=np.float64, order='C')
        self._channel_egrid = self._raw_channel_egrid = ch_egrid

    def _read_resp(self, resp_header, resp_data):
        nchan = resp_header.get('DETCHANS', None)
        if nchan is None:
            raise OSError(
                f'keyword "DETCHANS" not found in "{self._respfile}" header'
            )
        else:
            nchan = int(nchan)

        fchan_idx = resp_data.names.index('F_CHAN') + 1
        # set the first channel number to 1 if not found
        first_chan = int(resp_header.get(f'TLMIN{fchan_idx}', 1))

        channel = np.arange(first_chan, first_chan + nchan)
        self._channel = self._raw_channel = channel

        n_grp = resp_data['N_GRP']
        f_chan = resp_data['F_CHAN'] - first_chan
        n_chan = resp_data['N_CHAN']

        # if ndim == 1 and dtype is 'O', it is an array of array
        if f_chan.ndim == 1 and f_chan.dtype != np.dtype('O'):
            # f_chan is scalar in each row, make it an array
            f_chan = f_chan[:, None]

        if n_chan.ndim == 1 and n_chan.dtype != np.dtype('O'):
            # n_chan is scalar in each row, make it an array
            n_chan = n_chan[:, None]

        reduced_matrix = resp_data['MATRIX']
        full_matrix = np.zeros((len(resp_data), nchan))

        for i in range(len(resp_data)):
            n = int(n_grp[i])  # n channel subsets
            if n > 0:
                f = f_chan[i]  # first channel of each subset
                nc = n_chan[i]  # channel number of each subset
                e = f + nc  # end channel of each subset
                idx = np.append(0, nc).cumsum()  # reduced idx of subsets
                reduced_i = reduced_matrix[i]  # row of the reduced matrix
                full_i = full_matrix[i]  # row of the full matrix

                for j in range(n):
                    # reduced matrix of j-th channel subset
                    reduced_ij = reduced_i[idx[j] : idx[j + 1]]

                    # restore to the corresponding position in full matrix
                    full_i[int(f[j]) : int(e[j])] = reduced_ij

        self._matrix = self._raw_matrix = full_matrix

        if not np.allclose(
            resp_data['ENERG_LO'][1:], resp_data['ENERG_HI'][:-1]
        ):
            raise OSError(
                f'respfile ({self._respfile}) photon energy grids exist '
                'discontinuity'
            )

        if np.any(resp_data['ENERG_LO'] > resp_data['ENERG_HI']):
            raise OSError(
                f'respfile ({self._respfile}) photon energy grids are not '
                'increasing'
            )

        ph_egrid = np.append(resp_data['ENERG_LO'], resp_data['ENERG_HI'][-1])
        ph_egrid = np.asarray(ph_egrid, dtype=np.float64, order='C')
        self._ph_egrid = ph_egrid

    def _read_ancrfile(self):
        ancrfile = self._ancrfile

        if ancrfile:
            with fits.open(ancrfile) as arf_hdul:
                arf = arf_hdul['SPECRESP'].data['SPECRESP']

            if len(arf) != len(self._raw_matrix):
                respfile = self._respfile
                raise OSError(
                    f'rmf ({respfile}) and arf ({ancrfile}) are not matched'
                )

            self._raw_matrix *= arf[:, None]
            self._matrix = self._raw_matrix

    def _drop_zeros(self):
        """Drop zero entries at the beginning or end of photon energy grid."""
        matrix = self._raw_matrix
        ph_egrid = self._ph_egrid
        zero_mask = np.all(np.less_equal(matrix, 0.0), axis=1)
        if zero_mask.any():
            n_entries = len(ph_egrid) - 1
            last_idx = len(ph_egrid) - 2
            idx = np.flatnonzero(zero_mask)
            diff = np.diff(idx)
            if len(diff) == 0:  # only one zero entry
                idx = idx[0]
                if idx in (0, last_idx):  # the beginning/end of grid
                    matrix = matrix[~zero_mask]
                    if idx == 0:
                        ph_egrid = ph_egrid[1:]
                    else:
                        ph_egrid = ph_egrid[:-1]
            else:
                splits = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
                zeros_mask2 = np.full(n_entries, False)
                for s in splits:
                    if np.isin(s, (0, last_idx)).any():
                        # drop only if at beginning or ending part of grids
                        zeros_mask2[s] = True

                elo = ph_egrid[:-1][~zeros_mask2]
                ehi = ph_egrid[1:][~zeros_mask2]
                ph_egrid = np.append(elo, ehi[-1])

        self._raw_matrix = self._matrix = matrix[~zero_mask]
        self._ph_egrid = ph_egrid

    def group(self, grouping: NDArray, noticed: NDArray | None = None):
        """Group response matrix.

        Parameters
        ----------
        grouping : ndarray
            Channel with a grouping flag of 1 with all successive channels
            with grouping flags of -1 are combined.
        noticed : ndarray or None, optional
            Flag indicating which channel to be used in grouping.

        """
        l0 = len(self._raw_channel)

        if noticed is None:
            noticed = np.full(l0, True)

        l1 = len(grouping)
        l2 = len(noticed)
        if not l0 == l1 == l2:
            raise ValueError(
                f'length of grouping ({l1}) and good ({l2}) must match to '
                f'original channel ({l0})'
            )

        grp_idx = np.flatnonzero(grouping == 1)  # transform to index

        if len(grp_idx) == l0:  # case of no group, apply good mask
            self._channel = self._raw_channel[noticed]
            self._channel_egrid = self._raw_channel_egrid[noticed]
            self._matrix = self._raw_matrix[:, noticed]

        else:
            non_empty = np.greater(np.add.reduceat(noticed, grp_idx), 0)

            edge_indices = np.append(grp_idx, l0)
            channel = self._raw_channel
            emin, emax = self._raw_channel_egrid.T
            group_channel = []
            group_emin = []
            group_emax = []

            for i in range(len(grp_idx)):
                if not non_empty[i]:
                    continue
                slice_i = slice(edge_indices[i], edge_indices[i + 1])
                quality_slice = noticed[slice_i]
                channel_slice = channel[slice_i]
                group_channel.append(channel_slice[quality_slice].astype(str))
                group_emin.append(min(emin[slice_i]))
                group_emax.append(max(emax[slice_i]))

            self._channel = tuple(tuple(g) for g in group_channel)
            self._channel_egrid = np.column_stack([group_emin, group_emax])

            a = np.where(noticed, 1.0, 0.0)
            matrix = np.add.reduceat(a * self._raw_matrix, grp_idx, axis=1)
            self._matrix = matrix[:, non_empty]

    def plot(self, erange: NDArray | None = None):
        """Plot the response matrix."""
        plt.figure()
        ch_emin, ch_emax = self._raw_channel_egrid.T
        matrix = self._raw_matrix

        idx = np.flatnonzero(ch_emin[1:] - ch_emax[:-1])
        if len(idx) > 0:
            ch_emin = np.insert(ch_emin, idx + 1, ch_emax[idx])
            ch_emax = np.insert(ch_emax, idx + 1, ch_emin[idx + 1])
            matrix = np.insert(matrix, idx + 1, np.nan, axis=1)

        ch_egrid = np.append(ch_emin, ch_emax[-1])
        ch, ph = np.meshgrid(ch_egrid, self._ph_egrid)
        plt.pcolormesh(ch, ph, matrix, cmap='jet')
        plt.loglog()
        plt.xlabel('Measured Energy [keV]')
        plt.ylabel('Photon Energy [keV]')
        plt.colorbar(label='Effective Area [cm$^2$]')

        if erange is not None:
            erange = np.atleast_2d(erange)
            emin = np.expand_dims(erange[:, 0], axis=1)
            emax = np.expand_dims(erange[:, 1], axis=1)
            mask1 = np.less_equal(emin, ch_emin)
            mask2 = np.less_equal(ch_emax, emax)
            idx = [np.flatnonzero(i) for i in np.bitwise_and(mask1, mask2)]

            ignored = []
            if ch_emin[idx[0][0]] > ch_emin[0]:
                ignored.append((ch_emin[0], ch_emin[idx[0][0]]))
            for i in range(len(idx) - 1):
                this_noticed_right = ch_emax[idx[i][-1]]
                next_noticed_left = ch_emin[idx[i + 1][0]]
                ignored.append((this_noticed_right, next_noticed_left))
            if ch_emax[idx[-1][-1]] < ch_emax[-1]:
                ignored.append((ch_emax[idx[-1][-1]], ch_emax[-1]))

            y = (self._ph_egrid[0], self._ph_egrid[-1])
            for i in ignored:
                plt.fill_betweenx(y, *i, alpha=0.4, color='w', hatch='x')

        plt.show()

    @property
    def ph_egrid(self) -> NDArray:
        """Monte Carlo photon energy grid."""
        return self._ph_egrid

    @property
    def channel(self) -> tuple:
        """Measurement channel numbers."""
        return self._channel

    @property
    def ch_emin(self) -> NDArray:
        """Left edge of measured energy grid."""
        return self._channel_egrid[:, 0]

    @property
    def ch_emax(self) -> NDArray:
        """Right edge of measured energy grid."""
        return self._channel_egrid[:, 1]

    @property
    def ch_emid(self) -> NDArray:
        """Middle of measured energy grid."""
        return np.mean(self._channel_egrid, axis=1)

    @property
    def ch_width(self) -> NDArray:
        """Width of measured energy grid."""
        return self._channel_egrid[:, 1] - self._channel_egrid[:, 0]

    @property
    def ch_mean(self) -> NDArray:
        """Geometric mean of measured energy grid."""
        return np.sqrt(np.prod(self._channel_egrid, axis=1))

    @property
    def ch_error(self) -> NDArray:
        """Width between left/right and geometric mean of energy grid."""
        mean = self.ch_mean
        return np.abs([self.ch_emin - mean, self.ch_emax - mean])

    @property
    def matrix(self) -> NDArray:
        """Response matrix."""
        return self._matrix


class GroupingWaring(Warning):
    """Issued by grouping scale not being met for all channels."""
