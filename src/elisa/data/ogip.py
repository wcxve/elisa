"""Handle OGIP/92-007 format data loading."""
from __future__ import annotations

import re
import warnings

import numpy as np
from astropy.io import fits
from .grouping import (
    group_const,
    group_min,
    group_sig,
    group_pos,
    group_opt,
    group_optmin,
    group_optsig
)

__all__ = ['Data']
# TODO: support creating Data object from array

NDArray = np.ndarray


class Data:
    """Class to load observation data stored in OGIP/92-007 format.

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
            * const: `scale` number channels
            * min: counts >= `scale` for src + bkg
            * sig: src significance >= `scale`-sigma
            * opt: optimal binning, see Kaastra & Bleeker (2016, A&A)
            * optmin: opt with counts >= `scale` for src + bkg
            * optsig: opt with src significance >= `scale`-sigma
            * bmin: counts >= `scale` for bkg (useful for W-stat)
            * bpos: bkg < 0 with probability < `scale` (useful for PG-stat)
    scale : float or None, optional
        Grouping scale. Only takes effect if `group` is not None.
    spec_poisson : bool or None, optional
        Whether the spectrum data follows counting statistics, reading from
        the `specfile` header. This value must be set if ``POISSERR`` is
        undefined in the header.
    back_poisson : bool or None, optional
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
        corrfile: str | None = None,
        corrnorm: str | None = None
    ):
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
            msg = f'specfile ({specfile}) and respfile ({respfile}) are not '
            msg += 'matched'
            raise ValueError(msg)

        # check background file
        if backfile:
            back = Spectrum(backfile, back_poisson)
        elif spec.backfile:
            back = Spectrum(spec.backfile, back_poisson)
        else:
            back = None

        if back and len(spec._raw_counts) != len(back._raw_counts):
            msg = f'specfile ({specfile}) and backfile ({backfile}) are not '
            msg += 'matched'
            raise ValueError(msg)

        # bad quality
        bad = (1, 5) if ignore_bad else (1,)

        # check if quality of spectrum and background are matched
        good_quality = ~np.isin(spec.quality, bad)
        if back:
            back_good = ~np.isin(back.quality, bad)
            if not np.all(good_quality == back_good):
                good_quality &= back_good
                msg = 'ignore bad channels defined by the union of spectrum '
                msg += 'and background quality'
                warnings.warn(msg, Warning, stacklevel=2)

        # corrfile and corrnorm not supported yet
        if corrfile or corrnorm:
            msg = 'correction to data is not yet supported.'
            warnings.warn(msg, Warning, stacklevel=2)

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
        self._erange = np.array(erange, dtype=np.float64, order='C', ndmin=2)
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

        # NOTE:
        # grouping of area/background scale is not supported currently,
        # so we hard code effexpo here, but it should be moved into _set_data
        # once grouping of area/background scale is implemented.

        # spectrum attributes
        self._spec_exposure = spec.exposure
        self._spec_effexpo = spec.exposure * spec.area_scale * spec.back_scale
        self._spec_poisson = spec.poisson
        self._spec_counts = None
        self._spec_error = None

        self._has_back = True if back else False

        # background attributes
        if self._has_back:
            self._back_exposure = back.exposure
            self._back_effexpo = back.exposure*back.area_scale*back.back_scale
            self._back_poisson = back.poisson
        else:
            self._back_exposure = None
            self._back_effexpo = None
            self._back_poisson = None
        self._back_counts = None
        self._back_error = None

        # net spectrum attributes
        self._net_counts = None
        self._net_spec = None
        self._net_error = None

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
        """Adaptively group the spectrum.

        Parameters
        ----------
        method : str
            Method to group spectrum and background adaptively, these options
            are available so that each channel group has:
                * const: `scale` number channels
                * min: counts >= `scale` for src + bkg
                * sig: src significance >= `scale`-sigma
                * opt: optimal binning, see Kaastra & Bleeker (2016, A&A)
                * optmin: opt with counts >= `scale` for src + bkg
                * optsig: opt with src significance >= `scale`-sigma
                * bmin: counts >= `scale` for bkg (useful for W-stat)
                * bpos: bkg < 0 with probability < `scale` (useful for PG-stat)
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
        inconsistency in a spectral plot, i.e., the error bar of a channel
        group will cover these bad channels, whilst these bad channels are
        never used in fitting.

        """
        ch_emin, ch_emax = self._resp._raw_channel_egrid.T
        ch_mask = self._channel_mask(ch_emin, ch_emax)  # shape = (nchan, 2)
        spec_counts = self._spec._raw_counts
        # spec_error = self._spec._raw_error
        grouping = np.full(len(spec_counts), 1, dtype=np.int64)

        def apply_grouping(group_func, mask, *args):
            """function operating the grouping array defined above."""
            data = (i[mask] * self._good_quality[mask] for i in args)
            grouping_flag, grouping_success = group_func(*data, scale)
            grouping[mask] = grouping_flag
            return grouping_success

        def apply_map(func, *args):
            """map the apply function and return success flag."""
            return all(
                map(
                    lambda mask: apply_grouping(func, mask, *args),
                    ch_mask
                )
            )

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
                msg = 'Poisson background is required for "bmin" method'
                raise ValueError(msg)
            success = apply_map(group_min, back_counts)

        elif method == 'bpos':
            if self.has_back:
                back_counts = self._back._raw_counts
                back_error = self._back._raw_error
            else:
                msg = 'background data is required for "bpos" method'
                raise ValueError(msg)

            success = apply_map(group_pos, back_counts, back_error)

        else:
            supported = (
                'const', 'min', 'sig', 'opt', 'optmin', 'optsig', 'bmin',
                'bpos'
            )
            msg = f'supported grouping method are: {", ".join(supported)}'
            raise ValueError(msg)

        if not success:
            msg = f'"{method}" grouping failed in some {self._name} channels'
            warnings.warn(msg, GroupWaring)

        self._set_data(grouping)

    def _set_data(self, grouping: NDArray):
        """Set data according to quality, grouping, and energy range."""
        self._spec.group(grouping, self._good_quality)
        self._resp.group(grouping, self._good_quality)
        self._grouping = grouping

        if self._record_channel:
            groups_channel = np.array(
                [f'{self.name}_Ch{"+".join(c)}' for c in self._resp.channel]
            )
        else:
            grp_idx = np.flatnonzero(grouping == 1)  # transform to index
            non_empty = np.add.reduceat(self._good_quality, grp_idx) != 0
            groups_channel = np.array(
                [f'{self.name}_Ch{c}' for c in np.flatnonzero(non_empty)]
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
            ratio = self._spec_effexpo / self._back_effexpo
            net = self._spec_counts - ratio * self._back_counts
            spec = net * unit
            var = np.square(self._spec_error)
            var += np.square(ratio * self._back_error)
            var *= np.square(unit)
            self._net_counts = net
            self._net_spec = spec
            self._net_error = np.sqrt(var)

        else:
            self._net_counts = self._spec_counts
            self._net_spec = self._net_counts * unit
            self._net_error = self._spec_error * unit

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
    def spec_effexpo(self) -> float | NDArray:
        """Effective exposure of spectrum."""
        return self._spec_effexpo

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
    def back_effexpo(self) -> float | NDArray | None:
        """Effective exposure of background."""
        return self._back_effexpo

    @property
    def net_counts(self) -> NDArray:
        """Net counts in each measuring channel."""
        return self._net_counts

    @property
    def net_spec(self) -> NDArray:
        """Net counts per second per keV."""
        return self._net_spec

    @property
    def net_error(self) -> NDArray:
        """Uncertainty of net spectrum."""
        return self._net_error

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
    """Class to handle spectrum data in OGIP standard.

    Parameters
    ----------
    specfile : str
        Spectrum file path. For type II pha file, the row specifier must be
        given in the end of path, e.g., ``specfile="./spec.pha2{1}"``.
    poisson : bool or None, optional
        Whether the spectrum data follows counting statistics, reading from
        the `specfile` header. This value must be set if ``POISSERR`` is
        undefined in the header.

    """

    def __init__(
        self,
        specfile: str,
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

        # TODO: more robust way to detect a type II data
        # check if data is type II
        if not type_ii:
            msg = f'row id must be provided for type II spectrum {specfile}'

            nchan = len(data)
            if int(header.get('DETCHANS', nchan)) != nchan:
                raise ValueError(msg)

            if header.get('HDUCLAS4', '') == 'TYPE:II':
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
                    warnings.warn(msg, Warning, stacklevel=3)

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
                warnings.warn(msg, Warning, stacklevel=3)

        # check if statistical error are positive
        mask = stat_err < 0.0
        if np.any(mask):
            stat_err[mask] = 0.0
            msg = f'spectrum ({specfile}) has some statistical errors < 0, '
            msg += 'which are reset to 0'
            warnings.warn(msg, Warning, stacklevel=3)

        # check if systematic error are positive
        mask = sys_err < 0.0
        if np.any(mask):
            sys_err[mask] = 0.0
            msg = f'spectrum ({specfile}) has some systematic errors < 0, '
            msg += 'which are reset to 0'
            warnings.warn(msg, Warning, stacklevel=3)

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

        # get background scaling factor
        back_scale = np.float64(get_field('BACKSCAL', 1.0))
        if isinstance(back_scale, NDArray):
            back_scale = np.array(back_scale, dtype=np.float64, order='C')
        else:
            back_scale = np.float64(back_scale)
        self._back_scale = back_scale

        # get area scaling factor
        area_scale = get_field('AREASCAL', 1.0)
        if isinstance(area_scale, NDArray):
            area_scale = np.array(area_scale, dtype=np.float64, order='C')
        else:
            area_scale = np.float64(area_scale)
        self._area_scale = area_scale

        # get correction scaling factor
        self._corr_scale = np.float64(get_field('CORRSCAL', 0.0))

        self._header = header
        self._data = data
        self._counts = self._raw_counts = counts
        self._error = self._raw_error = error
        self._grouping = grouping
        self._exposure = exposure
        self._eff_exposure = exposure * area_scale * back_scale
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
        inconsistency in a spectral plot, i.e., the error bar of a channel
        group will cover these bad channels, whilst these bad channels are
        never used in fitting.

        """
        if not () == np.shape(self.area_scale) == np.shape(self.back_scale):
            msg = 'grouping is not implemented yet for the spectrum with '
            msg += '``AREASCAL`` and/or ``BACKSCAL`` array'
            raise NotImplementedError(msg)

        l0 = len(self._raw_counts)
        if noticed is None:
            noticed = np.full(l0, True)
        else:
            l1 = len(grouping)
            l2 = len(noticed)
            if not l0 == l1 == l2:
                msg = f'length of grouping ({l1}) and noticed ({l2}) must be '
                msg += f'matched to spectrum channel ({l0})'
                raise ValueError(msg)

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
    def eff_exposure(self) -> float | NDArray:
        """Effective exposure, corrected with area and background scaling."""
        return self._eff_exposure

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

        channel = ebounds['CHANNEL']

        # assume ph_egrid is continuous
        ph_egrid = np.append(resp['ENERG_LO'], resp['ENERG_HI'][-1])
        ph_egrid = np.asarray(ph_egrid, dtype=np.float64, order='C')
        ch_egrid = np.column_stack((ebounds['E_MIN'], ebounds['E_MAX']))
        ch_egrid = np.asarray(ch_egrid, dtype=np.float64, order='C')

        # extract response matrix
        matrix = resp['MATRIX']

        # wrap around N/A of matrix
        nch = len(ch_egrid)
        nch_matrix = np.array([len(i) for i in matrix])
        if np.any(nch_matrix != nch):
            # inhomogeneous matrix is due to zero elements being discarded
            # here we put zeros back in
            mask = np.less(np.arange(nch), nch_matrix[:, None])
            matrix_flatten = np.concatenate(matrix, dtype=np.float64)
            matrix = np.zeros(mask.shape)
            matrix[mask] = matrix_flatten

        # read in ancrfile if exists
        if ancrfile:
            with fits.open(ancrfile) as arf_hdul:
                arf = arf_hdul['SPECRESP'].data['SPECRESP']

            if len(arf) != len(matrix):
                msg = f'rmf ({respfile}) and arf ({ancrfile}) are not matched'
                raise ValueError(msg)

            matrix *= arf[:, None]

        # drop the zero entries at the beginning or end of photon energy grid
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
                        # drop only if at beginning or ending part of the grid
                        zeros_mask2[s] = True

                elo = ph_egrid[:-1][~zeros_mask2]
                ehi = ph_egrid[1:][~zeros_mask2]
                ph_egrid = np.append(elo, ehi[-1])

        self._ph_egrid = ph_egrid
        self._channel = self._raw_channel = channel
        self._channel_egrid = self._raw_channel_egrid = ch_egrid
        self._matrix = self._raw_matrix = matrix

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
            msg = f'length of grouping ({l1}) and good ({l2}) must match to '
            msg += f'original channel ({l0})'
            raise ValueError(msg)

        grp_idx = np.flatnonzero(grouping == 1)  # transform to index

        if len(grp_idx) == l0:  # case of no group, apply good mask
            self._channel = self._raw_channel[noticed]
            self._channel_egrid = self._raw_channel_egrid[noticed]
            self._matrix = self._raw_matrix[:, noticed]

        else:
            non_empty = np.add.reduceat(noticed, grp_idx) != 0

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

            self._channel = tuple(map(lambda g: tuple(g), group_channel))
            self._channel_egrid = np.column_stack([group_emin, group_emax])

            a = np.where(noticed, 1.0, 0.0)
            matrix = np.add.reduceat(a * self._raw_matrix, grp_idx, axis=1)
            self._matrix = matrix[:, non_empty]

    @property
    def ph_egrid(self) -> NDArray:
        """Photon energy grid."""
        return self._ph_egrid

    @property
    def channel(self) -> tuple:
        """Measured channel numbers."""
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


class GroupWaring(Warning):
    """Issued by grouping scale not being met for all channels."""
