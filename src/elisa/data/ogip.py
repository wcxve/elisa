"""Containers of OGIP/92-007 format data."""

from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.sparse import coo_array, csc_array

from elisa.data.grouping import (
    group_const,
    group_min,
    group_opt,
    group_optsig_gv,
    group_optsig_lima,
    group_optsig_normal,
    group_sig_gv,
    group_sig_lima,
    group_sig_normal,
    significance_gv,
    significance_lima,
)
from elisa.plot.misc import get_colors

if TYPE_CHECKING:
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
        given in the end of path, e.g., ``specfile='spec.pha2{1}'``.
    backfile : str or None, optional
        Background file path. Read from the `specfile` header if None.
        For type II pha file, the row specifier must be given in the end of
        path, e.g., ``backfile='back.pha2{1}'``.
    respfile : str or None, optional
        Response file path. Read from the `specfile` header if None.
        The path must be given if ``RESPFILE`` is undefined in the header.
    ancrfile : str or None, optional
        Ancillary response path. Read from the `specfile` header if None.

    Other Parameters
    ----------------
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
            * ``'bmin'``: background counts >= `scale`, used to avoid bias when
              using ``wstat`` to simultaneously fit the source and background
            * ``'bsig'``: background significance >= `scale` sigma, used to
              avoid bias when using ``pgstat`` to simultaneously fit the source
              and background
            * ``'opt'``: optimal binning, see Kaastra & Bleeker (2016) [3]_
            * ``'optmin'``: optimal binning with total counts >= `scale`
            * ``'optsig'``: optimal binning with source significance >= `scale`
              sigma
            * ``'optbmin'``: optimal binning with background counts >= `scale`
            * ``'optbsig'``: optimal binning with background significance
              >= `scale` sigma

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
    resp_sparse : bool, optional
        Whether the response matrix is sparse. The default is False.

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
        resp_sparse: bool = False,
        # corrfile: bool | None = None,
        # corrnorm: bool | None = None,
    ):
        erange = np.array(erange, dtype=np.float64, order='C', ndmin=2)

        # check if erange is increasing
        if np.any(np.diff(erange, axis=1) <= 0.0):
            raise ValueError('erange must be increasing')

        # check if erange is overlapped
        erange = erange[erange[:, 0].argsort()]
        if np.any(np.diff(np.hstack(erange)) <= 0.0):
            raise ValueError('erange must not be overlapped')

        try:
            spec = Spectrum(specfile, spec_poisson)
        except PoissonFlagNotFoundError as err:
            raise PoissonFlagNotFoundError(
                '"POISSERR" is undefined in spectrum header, `spec_poisson` '
                'must be set in Data(..., spec_poisson=True/False)'
            ) from err

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
            raise ValueError(
                f'specfile ({specfile}) and respfile ({respfile}) are not '
                'matched'
            )

        # check background file
        try:
            if backfile:
                back = Spectrum(backfile, back_poisson)
            elif spec.backfile:
                back = Spectrum(spec.backfile, back_poisson)
            else:
                back = None
        except PoissonFlagNotFoundError as err:
            raise PoissonFlagNotFoundError(
                '"POISSERR" is undefined in background header, `back_poisson` '
                'must be set in Data(..., back_poisson=True/False)'
            ) from err

        if back and len(spec._raw_counts) != len(back._raw_counts):
            raise ValueError(
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
        if not np.any(good_quality):
            raise RuntimeError(f'no good channel is found for {name} data')

        # corrfile and corrnorm are not supported yet
        # if corrfile or corrnorm:
        #     warnings.warn(
        #         'correction to data is not yet supported',
        #         Warning,
        #         stacklevel=2,
        #     )
        #
        # check correction file
        # use poisson=True to bypass stat_err check, which takes no effect
        # if corrfile:
        #     corr = Spectrum(corrfile, True)
        # elif spec.corrfile:
        #     corr = Spectrum(spec.corrfile, True)
        # else:
        #     corr = None
        #
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
        self._resp_sparse = bool(resp_sparse)
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

    def group(self, method: str, scale: float | int | None):
        """Group the spectrum.

        Parameters
        ----------
        method : str
            Method to group spectrum and background adaptively, these options
            are available so that each channel group has:

            * ``'const'``: `scale` number channels
            * ``'min'``: total (source + background) counts >= `scale`
            * ``'sig'``: source significance >= `scale` sigma
            * ``'bmin'``: background counts >= `scale`, used to avoid bias when
              using ``wstat`` to simultaneously fit the source and background
            * ``'bsig'``: background significance >= `scale` sigma, used to
              avoid bias when using ``pgstat`` to simultaneously fit the source
              and background
            * ``'opt'``: optimal binning, see Kaastra & Bleeker (2016) [1]_
            * ``'optmin'``: optimal binning with total counts >= `scale`
            * ``'optsig'``: optimal binning with source significance >= `scale`
              sigma
            * ``'optbmin'``: optimal binning with background counts >= `scale`
            * ``'optbsig'``: optimal binning with background significance
              >= `scale` sigma

        scale : float, int or None
            Grouping scale.

        Raises
        ------
        NotImplementedError
            Grouping is not yet implemented for spectrum with ``AREASCAL``
            and/or ``BACKSCAL`` array.

        Warns
        -----
        GroupWarning
            Warn if grouping scale is not met for any channel.

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
        method = str(method)
        scale = float(scale) if scale is not None else None

        if method != 'opt' and scale is None:
            raise ValueError(f'scale must be given for {method} grouping')

        if method.startswith('opt'):
            fwhm = self._resp.ch_fwhm
        else:
            fwhm = None

        ch_emin, ch_emax = self._resp._raw_channel_egrid.T
        ch_mask = self._channel_mask(ch_emin, ch_emax)  # shape = (nchan, 2)
        spec_counts = self._spec._raw_counts
        spec_errors = self._spec._raw_error
        if self.has_back:
            back_ratio = self.back_ratio
            back_counts = self._back._raw_counts
            back_errors = self._back._raw_error
            berr = back_ratio * back_errors
            net_counts = spec_counts - back_ratio * back_counts
            net_errors = np.sqrt(spec_errors * spec_errors + berr * berr)
        else:
            back_ratio = None
            back_counts = None
            back_errors = None
            net_counts = spec_counts
            net_errors = spec_errors

        grouping = np.full(len(spec_counts), 1, dtype=np.int64)

        def apply_grouping(group_func, mask, args, all_good=()):
            """Apply the grouping array defined above."""
            data = []
            for i, j in enumerate(args):
                if np.shape(j) == ():  # scalar
                    data.append(j)
                else:
                    if i not in all_good:
                        data.append(j[mask] * self._good_quality[mask])
                    else:
                        data.append(j[mask])
            grouping_flag, grouping_success = group_func(*data, scale)
            grouping[mask] = grouping_flag
            return grouping_success

        def apply_map(func, *args, all_good=()):
            """Map the apply function and return a success flag."""
            return all(
                apply_grouping(func, mask, args, all_good) for mask in ch_mask
            )

        if method == 'const':
            success = apply_map(group_const, len(spec_counts))

        elif method == 'min':
            success = apply_map(group_min, spec_counts)

        elif method == 'sig':
            if self.spec_poisson:
                if self.back_poisson:
                    fn = group_sig_lima
                    args = (spec_counts, back_counts, back_ratio)
                    all_good = (2,)
                else:
                    fn = group_sig_gv
                    args = (spec_counts, back_counts, back_errors, back_ratio)
                    all_good = (3,)
            else:
                fn = group_sig_normal
                args = (net_counts, net_errors)
                all_good = ()
            success = apply_map(fn, *args, all_good=all_good)

        elif method == 'bmin':
            if not (self.has_back and self.back_poisson):
                raise ValueError(
                    'Poisson background is required for "bmin" method'
                )
            success = apply_map(group_min, back_counts)

        elif method == 'bsig':
            if not self.has_back:
                raise ValueError(
                    'background data is required for "bsig" method'
                )
            success = apply_map(group_sig_normal, back_counts, back_errors)

        elif method == 'opt':
            success = apply_map(group_opt, fwhm, net_counts, all_good=(0, 1))

        elif method == 'optmin':
            success = apply_map(
                group_opt, fwhm, net_counts, spec_counts, all_good=(0, 1)
            )

        elif method == 'optsig':
            if self.spec_poisson:
                if self.back_poisson:
                    fn = group_optsig_lima
                    args = (
                        fwhm,
                        net_counts,
                        spec_counts,
                        back_counts,
                        back_ratio,
                    )
                else:
                    fn = group_optsig_gv
                    args = (
                        fwhm,
                        net_counts,
                        spec_counts,
                        back_counts,
                        back_errors,
                        back_ratio,
                    )
            else:
                fn = group_optsig_normal
                args = (fwhm, net_counts, net_counts, net_errors)
            success = apply_map(fn, *args, all_good=(0, 1))

        elif method == 'optbmin':
            if not (self.has_back and self.back_poisson):
                raise ValueError(
                    'Poisson background is required for "optbmin" method'
                )
            success = apply_map(
                group_opt, fwhm, net_counts, back_counts, all_good=(0, 1)
            )

        elif method == 'optbsig':
            if not self.has_back:
                raise ValueError(
                    'background data is required for "optbsig" method'
                )
            success = apply_map(
                group_optsig_normal,
                fwhm,
                net_counts,
                back_counts,
                back_errors,
                all_good=(0, 1),
            )

        else:
            supported = (
                'const',
                'min',
                'bmin',
                'bsig',
                'sig',
                'opt',
                'optmin',
                'optsig',
                'optbmin',
                'optbsig',
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
            grp_idx = np.flatnonzero(grouping != -1)  # transform to index
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
        self._resp_matrix = resp.sparse_matrix.tocsc()[:, ch_mask]

        # spectrum attribute
        spec = self._spec
        self._spec_counts = spec.counts[ch_mask]
        self._spec_error = spec.error[ch_mask]

        # background attribute
        if self._has_back:
            self._back.group(grouping, self._good_quality)
            self._back_counts = self._back.counts[ch_mask]
            self._back_error = self._back.error[ch_mask]

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

    def plot_spec(self, xlog: bool = True, ylog: bool = False):
        """Plot the spectrum.

        .. warning::
            The significance plot is accurate only if the spectrum data has
            enough count statistics.

        Parameters
        ----------
        xlog : bool, optional
            Whether to use log scale on x-axis. The default is True.
        ylog : bool, optional
            Whether to use log scale on y-axis. The default is False.
        """
        fig, axs = plt.subplots(
            nrows=2,
            ncols=1,
            sharex=True,
            height_ratios=[1.618, 1.0],
            gridspec_kw={'hspace': 0.05},
        )
        fig.align_ylabels(axs)

        for ax in axs:
            ax.tick_params(
                axis='both',
                which='both',
                direction='in',
                bottom=True,
                top=True,
                left=True,
                right=True,
            )

        plt.rcParams['axes.formatter.min_exponent'] = 3

        axs[0].set_ylabel(r'$C_E\ \mathrm{[s^{-1}\ keV^{-1}]}$')
        axs[1].set_ylabel(r'Significance [$\mathrm{\sigma}$]')
        axs[1].set_xlabel(r'$\mathrm{Energy\ [keV]}$')

        if self.has_back:
            colors = get_colors(2, 'colorblind')
        else:
            colors = get_colors(1, 'colorblind')

        factor = 1.0 / (self.spec_exposure * self.ch_width)
        x = self.ch_mean if xlog else self.ch_emid
        xerr = self.ch_error if xlog else 0.5 * self.ch_width
        axs[0].errorbar(
            x=x,
            xerr=xerr,
            y=self.spec_counts * factor,
            yerr=self.spec_error * factor,
            fmt='o',
            color=colors[0],
            alpha=0.8,
            label='Observation',
            ms=3,
            mfc='#FFFFFFCC',
        )

        if self.has_back:
            factor = self.back_ratio / (self.spec_exposure * self.ch_width)
            axs[0].errorbar(
                x=x,
                xerr=xerr,
                y=self.back_counts * factor,
                yerr=self.back_error * factor,
                fmt='s',
                color=colors[1],
                alpha=0.8,
                label='Background',
                ms=3,
                mfc='#FFFFFFCC',
            )

        if self.spec_poisson and self.has_back:
            if self.back_poisson:
                sig = significance_lima(
                    self.spec_counts, self.back_counts, self.back_ratio
                )
            else:
                sig = significance_gv(
                    self.spec_counts,
                    self.back_counts,
                    self.back_error,
                    self.back_ratio,
                )
            sig *= np.sign(self.net_counts)
        else:
            sig = self.net_counts / self.net_error

        axs[1].errorbar(
            x=x,
            xerr=xerr,
            y=sig,
            yerr=1,
            fmt='o',
            color=colors[0],
            alpha=0.8,
            ms=3,
            mfc='#FFFFFFCC',
        )
        axs[1].axhline(0, color='k', ls=':', lw=0.5)

        if xlog:
            axs[0].set_xscale('log')
        if ylog:
            axs[0].set_yscale('log')

        axs[0].legend()
        axs[0].set_title(self.name)

    def plot_effective_area(self, hatch: bool = True):
        """Plot the effective area.

        Parameters
        ----------
        hatch : bool, optional
            Whether to add hatches in the ignored region. The default is True.
        """
        self._resp.plot_effective_area(self._erange if hatch else None)

    def plot_matrix(self, hatch: bool = True) -> None:
        """Plot the response matrix.

        Parameters
        ----------
        hatch : bool, optional
            Whether to add hatches in the ignored region. The default is True.
        """
        self._resp.plot_matrix(self._erange if hatch else None)

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
    def spec_exposure(self) -> np.float64:
        """Spectrum exposure."""
        return self._spec_exposure

    @property
    def area_factor(self) -> np.float64 | NDArray:
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
    def back_exposure(self) -> np.float64 | None:
        """Background exposure."""
        return self._back_exposure

    @property
    def back_ratio(self) -> np.float64 | NDArray | None:
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
        """Measurement channel information."""
        return self._channel

    @property
    def ch_emin(self) -> NDArray:
        """Left edge of measurement energy grid."""
        return self._ch_emin

    @property
    def ch_emax(self) -> NDArray:
        """Right edge of measurement energy grid."""
        return self._ch_emax

    @property
    def ch_emid(self) -> NDArray:
        """Middle of measurement energy grid."""
        return self._ch_emid

    @property
    def ch_width(self) -> NDArray:
        """Width of measurement energy grid."""
        return self._ch_width

    @property
    def ch_mean(self) -> NDArray:
        """Geometric mean of measurement energy grid."""
        return self._ch_mean

    @property
    def ch_error(self) -> NDArray:
        """Width between left/right and geometric mean of channel grid."""
        return self._ch_error

    @property
    def resp_matrix(self) -> NDArray:
        """Response matrix."""
        return self._resp_matrix.todense()

    @property
    def sparse_resp_matrix(self) -> coo_array:
        """Sparse response matrix."""
        return self._resp_matrix

    @property
    def resp_sparse(self) -> bool:
        """Whether the response matrix is sparse."""
        return self._resp_sparse


class Spectrum:
    """Handle spectral data in OGIP standards [1]_.

    Parameters
    ----------
    specfile : str
        Spectrum file path. For type II pha file, the row specifier must be
        given in the end of path, e.g., ``specfile='spec.pha2{1}'``.
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
            msg = 'row id must be provided for type II spectrum, e.g., '
            msg += f"'{specfile}{{1}}'"

            nchan = len(data)
            if int(header.get('DETCHANS', nchan)) != nchan:
                raise ValueError(msg)

            if header.get('HDUCLAS4', '') == 'TYPE:II':
                raise ValueError(msg)

        else:
            data = data[spec_id : spec_id + 1]  # set data to the specified row

        # check if COUNTS or RATE exists
        if 'COUNTS' not in data.names and 'RATE' not in data.names:
            raise ValueError(f'"COUNTS" or "RATE" not found in {specfile}')

        # get poisson flag
        poisson = header.get('POISSERR', poisson)
        if poisson is None:
            raise PoissonFlagNotFoundError(
                '"POISSERR" is undefined in header, `poisson` must be set in '
                'Spectrum(..., poisson=True/False)'
            )

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

        # check data
        if poisson:
            # check if counts are integers
            diff = np.abs(counts - np.round(counts))
            if np.any(diff > 1e-8 * counts):
                warnings.warn(
                    f'poisson spectrum ({specfile}) has non-integer counts, '
                    'which may lead to wrong result',
                    Warning,
                    stacklevel=3,
                )
        else:
            # check if statistical errors are positive
            if np.any(stat_err < 0.0):
                raise ValueError(
                    f'spectrum ({specfile}) has negative statistical errors'
                )

            if np.any(stat_err == 0.0):
                warnings.warn(
                    f'spectrum ({specfile}) has zero statistical errors, '
                    'which may lead to wrong result under Gaussian statistics,'
                    ' consider to group the spectrum',
                    Warning,
                    stacklevel=3,
                )

            # check if systematic errors are non-negative
            if np.any(sys_err < 0.0):
                raise ValueError(
                    f'spectrum ({specfile}) has systematic errors < 0'
                )

        # total error of counts
        if not poisson:
            stat_var = np.square(stat_err)
            sys_var = np.square(sys_err * counts)
            error = np.sqrt(stat_var + sys_var)
        else:
            error = stat_err

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
        # self._corrfile = get_field('CORRFILE', '', excluded_file)

        # get the background scaling factor
        back_scale = np.float64(get_field('BACKSCAL', 1.0))
        if isinstance(back_scale, np.ndarray):
            back_scale = np.array(back_scale, dtype=np.float64, order='C')
        else:
            back_scale = np.float64(back_scale)
        self._back_scale = back_scale

        # get the area scaling factor
        area_scale = get_field('AREASCAL', 1.0)
        if isinstance(area_scale, np.ndarray):
            area_scale = np.array(area_scale, dtype=np.float64, order='C')
        else:
            area_scale = np.float64(area_scale)
        self._area_scale = area_scale

        # get the correction scaling factor
        # self._corr_scale = np.float64(get_field('CORRSCAL', 0.0))

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
        grp_idx = np.flatnonzero(grouping != -1)  # transform to index
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
    def exposure(self) -> np.float64:
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

    # @property
    # def corrfile(self) -> str:
    #     """Correction file."""
    #     return self._corrfile

    @property
    def back_scale(self) -> np.float64 | NDArray:
        """Background scaling factor."""
        return self._back_scale

    @property
    def area_scale(self) -> np.float64 | NDArray:
        """Area scaling factor."""
        return self._area_scale

    # @property
    # def corr_scale(self) -> np.float64:
    #     """Correction scaling factor."""
    #     return self._corr_scale


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
            RMF and ARF file formats) <https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html>`__
            and `Addendum: Changes log <https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002a/cal_gen_92_002a.html>`__
    """

    _fwhm: NDArray | None = None
    _ch_fwhm: NDArray | None = None

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
                raise ValueError(
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
            raise ValueError(
                f'respfile ({self._respfile}) channel energy grids are not '
                'increasing'
            )
        ch_egrid = np.column_stack((ch_emin, ch_emax))
        ch_egrid = np.asarray(ch_egrid, dtype=np.float64, order='C')
        self._channel_egrid = self._raw_channel_egrid = ch_egrid

    def _read_resp(self, resp_header, resp_data):
        # check and read photon energy grid
        if not np.allclose(
            resp_data['ENERG_LO'][1:], resp_data['ENERG_HI'][:-1]
        ):
            raise ValueError(
                f'respfile ({self._respfile}) photon energy grids exist '
                'discontinuity'
            )

        if np.any(resp_data['ENERG_LO'] > resp_data['ENERG_HI']):
            raise ValueError(
                f'respfile ({self._respfile}) photon energy grids are not '
                'increasing'
            )

        ph_egrid = np.append(resp_data['ENERG_LO'], resp_data['ENERG_HI'][-1])
        ph_egrid = np.asarray(ph_egrid, dtype=np.float64, order='C')
        self._ph_egrid = ph_egrid

        # check and read response matrix
        nchan = resp_header.get('DETCHANS', None)
        if nchan is None:
            raise ValueError(
                f'keyword "DETCHANS" not found in "{self._respfile}" header'
            )
        else:
            nchan = int(nchan)

        fchan_idx = resp_data.names.index('F_CHAN') + 1
        # set the first channel number to 1 if not found
        first_chan = int(resp_header.get(f'TLMIN{fchan_idx}', 1))

        channel = tuple(str(c) for c in range(first_chan, first_chan + nchan))
        self._raw_channel = channel
        self._channel = tuple((c,) for c in channel)

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

        # the last channel numbers
        e_chan = f_chan + n_chan

        rows = np.hstack(
            tuple(np.full(round(n.sum()), i) for i, n in enumerate(n_chan))
        )
        cols = []
        for i in range(len(resp_data)):
            n = int(n_grp[i])  # n channel subsets
            if n > 0:
                f = f_chan[i].astype(int)  # first channel numbers of subsets
                e = e_chan[i].astype(int)  # last channel numbers of subsets
                cols.extend(map(np.arange, f, e))
        cols = np.hstack(cols)

        matrix = resp_data['MATRIX'].ravel()
        if matrix.dtype != np.dtype('O'):
            reduced_matrix = matrix
        else:
            reduced_matrix = np.hstack(matrix)
        self._sparse_matrix = coo_array(
            (reduced_matrix, (rows, cols)), shape=(len(resp_data), nchan)
        )
        self._sparse_matrix.eliminate_zeros()
        self._matrix = self._sparse_matrix

    def _read_ancrfile(self):
        ancrfile = self._ancrfile

        if ancrfile:
            with fits.open(ancrfile) as arf_hdul:
                arf = arf_hdul['SPECRESP'].data['SPECRESP']

            if len(arf) != self._sparse_matrix.shape[0]:
                respfile = self._respfile
                raise ValueError(
                    f'rmf ({respfile}) and arf ({ancrfile}) are not matched'
                )

            self._sparse_matrix *= arf[:, None]
            self._matrix = self._sparse_matrix

    def _drop_zeros(self):
        """Remove leading or trailing rows filled with 0 from the matrix."""
        matrix = self._sparse_matrix
        ph_egrid = self._ph_egrid
        nonzero_rows = np.unique(matrix.nonzero()[0])
        nonzero_mask = np.isin(range(matrix.shape[0]), nonzero_rows)
        zero_mask = np.bitwise_not(nonzero_mask)
        if zero_mask.any():
            n_entries = len(ph_egrid) - 1
            last_idx = len(ph_egrid) - 2
            idx = np.flatnonzero(zero_mask)
            diff = np.diff(idx)
            if len(diff) == 0:  # only one zero entry
                idx = idx[0]
                if idx in (0, last_idx):  # check if idx is leading or trailing
                    if idx == 0:
                        ph_egrid = ph_egrid[1:]
                    else:
                        ph_egrid = ph_egrid[:-1]
            else:
                splits = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
                zeros_mask2 = np.full(n_entries, False)
                for s in splits:
                    if np.isin(s, (0, last_idx)).any():
                        # only drop leading or trailing part of grids
                        zeros_mask2[s] = True

                elo = ph_egrid[:-1][~zeros_mask2]
                ehi = ph_egrid[1:][~zeros_mask2]
                ph_egrid = np.append(elo, ehi[-1])

        self._sparse_matrix = self._matrix = matrix.tocsr()[~zero_mask]
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

        grp_idx = np.flatnonzero(grouping != -1)  # transform to index

        if len(grp_idx) == l0:  # case of no group, apply good mask
            if np.count_nonzero(noticed) != noticed.size:
                channel = np.array(self._raw_channel)[noticed]
                self._channel = tuple((c,) for c in channel)
                self._channel_egrid = self._raw_channel_egrid[noticed]
                self._matrix = self._sparse_matrix.tocsc()[:, noticed]

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
                channel_slice = np.array(channel[slice_i])[quality_slice]
                group_channel.append(tuple(channel_slice))
                group_emin.append(min(emin[slice_i]))
                group_emax.append(max(emax[slice_i]))

            self._channel = tuple(group_channel)
            self._channel_egrid = np.column_stack([group_emin, group_emax])

            a = np.where(noticed, 1.0, 0.0)
            n = self._sparse_matrix.shape[1]
            idx = np.arange(n)
            ptr = np.append(grp_idx, n)
            grouping_matrix = csc_array((a, idx, ptr))
            matrix = self._sparse_matrix.dot(grouping_matrix)
            self._matrix = matrix.tocsc()[:, non_empty]

    def plot_effective_area(self, noticed_range: NDArray | None = None):
        """Plot the response matrix.

        Parameters
        ----------
        noticed_range : ndarray, optional
            Energy range to show. Other energy ranges will be hatched.
        """
        eff_area = self._sparse_matrix.sum(axis=1)
        eff_area = np.clip(eff_area, a_min=0.0, a_max=None)

        plt.figure()
        plt.rcParams['axes.formatter.min_exponent'] = 3
        plt.step(self.ph_egrid, np.append(eff_area, eff_area[-1]))
        plt.xlim(self.ph_egrid[0], self.ph_egrid[-1])
        plt.xlabel('Photon Energy [keV]')
        plt.ylabel('Effective Area [cm$^2$]')
        plt.xscale('log')

        if noticed_range is not None:
            ph_emin = self.ph_egrid[:-1]
            ph_emax = self.ph_egrid[1:]
            noticed_range = np.atleast_2d(noticed_range)
            emin = np.expand_dims(noticed_range[:, 0], axis=1)
            emax = np.expand_dims(noticed_range[:, 1], axis=1)
            mask1 = np.less_equal(emin, ph_emin)
            mask2 = np.less_equal(ph_emax, emax)
            idx = [np.flatnonzero(i) for i in np.bitwise_and(mask1, mask2)]

            ignored = []
            if ph_emin[idx[0][0]] > ph_emin[0]:
                ignored.append((ph_emin[0], ph_emin[idx[0][0]]))
            for i in range(len(idx) - 1):
                this_noticed_right = ph_emax[idx[i][-1]]
                next_noticed_left = ph_emin[idx[i + 1][0]]
                ignored.append((this_noticed_right, next_noticed_left))
            if ph_emax[idx[-1][-1]] < ph_emax[-1]:
                ignored.append((ph_emax[idx[-1][-1]], ph_emax[-1]))

            ylim = plt.gca().get_ylim()
            for i in ignored:
                plt.fill_betweenx(
                    ylim, *i, alpha=0.8, color='gray', hatch='x', zorder=10
                )
            plt.ylim(ylim)

    def plot_matrix(self, noticed_range: NDArray | None = None):
        """Plot the response matrix.

        Parameters
        ----------
        noticed_range : ndarray, optional
            Energy range to show. Other energy ranges will be hatched.
        """
        ch_emin, ch_emax = self._raw_channel_egrid.T
        matrix = np.clip(self._sparse_matrix.todense(), a_min=0.0, a_max=None)

        # some response matrix has discontinuity in channel energy grid,
        # insert np.nan to handle this
        idx = np.flatnonzero(ch_emin[1:] - ch_emax[:-1])
        if len(idx) > 0:
            ch_emin = np.insert(ch_emin, idx + 1, ch_emax[idx])
            ch_emax = np.insert(ch_emax, idx + 1, ch_emin[idx + 1])
            matrix = np.insert(matrix, idx + 1, np.nan, axis=1)

        ch_egrid = np.append(ch_emin, ch_emax[-1])
        ch, ph = np.meshgrid(ch_egrid, self._ph_egrid)
        plt.figure()
        plt.rcParams['axes.formatter.min_exponent'] = 3
        plt.pcolormesh(ch, ph, matrix, cmap='magma')
        plt.xlabel('Measurement Energy [keV]')
        plt.ylabel('Photon Energy [keV]')
        plt.colorbar(label='Effective Area [cm$^2$]')
        plt.xscale('log')
        plt.yscale('log')

        if noticed_range is not None:
            noticed_range = np.atleast_2d(noticed_range)
            emin = np.expand_dims(noticed_range[:, 0], axis=1)
            emax = np.expand_dims(noticed_range[:, 1], axis=1)
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
        """Left edge of measurement energy grid."""
        return self._channel_egrid[:, 0]

    @property
    def ch_emax(self) -> NDArray:
        """Right edge of measurement energy grid."""
        return self._channel_egrid[:, 1]

    @property
    def ch_emid(self) -> NDArray:
        """Middle of measurement energy grid."""
        return np.mean(self._channel_egrid, axis=1)

    @property
    def ch_width(self) -> NDArray:
        """Width of measurement energy grid."""
        return self._channel_egrid[:, 1] - self._channel_egrid[:, 0]

    @property
    def ch_mean(self) -> NDArray:
        """Geometric mean of measurement energy grid."""
        return np.sqrt(np.prod(self._channel_egrid, axis=1))

    @property
    def ch_error(self) -> NDArray:
        """Width between left/right and geometric mean of energy grid."""
        mean = self.ch_mean
        return np.abs([self.ch_emin - mean, self.ch_emax - mean])

    @property
    def matrix(self) -> NDArray:
        """Response matrix."""
        return self._matrix.todense()

    @property
    def sparse_matrix(self):
        """Sparse representation of the response matrix."""
        return self._matrix

    @property
    def fwhm(self) -> NDArray:
        """Estimated Full Width at Half Maximum (FWHM) in photon energy space.

        .. note::
            The calculation code is translated from ``heasp``. This does assume
            that the response has a well-defined main peak and operates by the
            simple method of stepping out from the peak in both directions till
            the response falls below half the maximum. A better solution would
            obviously be to fit a gaussian.
        """
        if self._fwhm is not None:
            return self._fwhm

        matrix = self._sparse_matrix
        csr_matrix = matrix.tocsr()
        nE, nC = matrix.shape
        row_idx = np.arange(nE)
        argmax = np.squeeze(matrix.argmax(axis=1))
        max_value = csr_matrix[row_idx, argmax]
        half_max = 0.5 * max_value
        i, j = np.nonzero(matrix <= half_max[:, None])
        idx_low_high = np.column_stack([argmax - 1, argmax + 1])
        for iE in range(nE):
            imax = argmax[iE]
            idx = j[i == iE]  # rsp elements of iE row at idx are <= half_max
            if idx.size:
                mask = idx < imax
                # find the right-most lower index
                k = np.flatnonzero(mask)
                lower = idx[k[-1]] if k.size else imax - 1
                # find the left-most upper index
                k = np.flatnonzero(~mask)
                upper = idx[k[0]] if k.size else imax + 1
                idx_low_high[iE] = [lower, upper]
        ilow, ihigh = np.clip(idx_low_high, 0, nC - 1).T
        good_low = csr_matrix[row_idx, [0] * nE] <= half_max
        good_high = csr_matrix[row_idx, [-1] * nE] <= half_max
        fwhm = np.full(nE, 0)
        fwhm[good_high] += ihigh[good_high] - argmax[good_high]
        fwhm[good_low] += argmax[good_low] - ilow[good_low]
        fwhm[(good_high & (~good_low)) | ((~good_high) & good_low)] *= 2
        fwhm = np.clip(fwhm, 1, None)  # set minimum FWHM to 1
        fwhm[~(good_high | good_low)] = -1
        self._fwhm = fwhm
        return self._fwhm

    @property
    def ch_fwhm(self) -> NDArray:
        """Estimated Full Width at Half Maximum (FWHM) in channel energy space.

        .. note::
            The calculation code is translated from ``heasp``. The calculation
            interpolates the `estimated_fwhm` using the nominal channel energy
            to give the FWHM for each channel. Assuming that FWHM does not
            change significantly over the channel, so just find the FWHM at the
            center energy of the channel.
        """
        if self._ch_fwhm is not None:
            return self._ch_fwhm

        fwhm = self.fwhm
        ch_emid = self._raw_channel_egrid.mean(1)
        idx = np.searchsorted(self.ph_egrid, ch_emid) - 1
        idx = np.clip(idx, 0, len(fwhm) - 1)
        self._ch_fwhm = fwhm[idx]
        return self._ch_fwhm


class GroupingWaring(Warning):
    """Issued by grouping scale not being met for all channel groups."""


class PoissonFlagNotFoundError(RuntimeError):
    """Issued by ``POISSERR`` not found in spectrum header."""
