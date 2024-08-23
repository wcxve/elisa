from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_array, csc_array, sparray

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
    NDArray = np.ndarray


# TODO: support multiple response in a single data object
# TODO: support area and background scale arrays
class ObservationData:
    """Observation data.

    Parameters
    ----------
    name : str
        Name of the observation data.
    erange : array_like
        Energy range of interest in keV, e.g., ``erange=[(0.5, 2), (5, 200)]``.
    spec_data : SpectrumData
        Spectrum data of the observation.
    resp_data : ResponseData
        Response matrix data of the observation.
    back_data : SpectrumData
        Background data of the observation.
    ignore_bad : bool, optional
        Whether to ignore bad channels whose quality flags are 2 or 5.
        The default is True.
    keep_channel_info : bool, optional
        Whether to keep channel information when grouping the data.
        The default is False.
    """

    def __init__(
        self,
        name: str,
        erange: list | tuple,
        spec_data: SpectrumData,
        resp_data: ResponseData,
        back_data: SpectrumData | None = None,
        ignore_bad: bool = True,
        keep_channel_info: bool = False,
    ):
        if not isinstance(spec_data, SpectrumData):
            raise TypeError('spec_data must be a SpectrumData instance')

        if not isinstance(resp_data, ResponseData):
            raise TypeError('resp_data must be a ResponseData instance')

        if len(spec_data.counts) != resp_data.channel_number:
            raise ValueError(
                f'channels number of spec_data ({len(spec_data.counts)}) '
                f'and resp_data ({resp_data.channel_number}) are not matched'
            )

        if back_data is not None:
            if not isinstance(back_data, SpectrumData):
                raise TypeError('back_data must be a SpectrumData instance')

            if len(back_data.counts) != resp_data.channel_number:
                raise ValueError(
                    f'channels number of back_data ({len(back_data.counts)}) '
                    f'and resp_data ({resp_data.channel_number}) are not '
                    'matched'
                )

        self.name = name
        self._spec_data = spec_data
        self._resp_data = resp_data
        self._back_data = back_data
        self._has_back = back_data is not None
        if self.has_back:
            spec = self.spec_data
            back = self.back_data
            self._back_ratio = (
                spec.exposure * spec.area_scale * spec.back_scale
            ) / (back.exposure * back.area_scale * back.back_scale)
        else:
            self._back_ratio = None

        # bad quality flags of the spectrum
        bad_quality = (1, 2, 5) if ignore_bad else (1,)
        bad_flag = np.isin(self.spec_data.quality, bad_quality)
        good_quality = np.bitwise_not(bad_flag)

        # check if the quality of spectrum and background are matched
        if self.has_back:
            back_bad_flag = np.isin(self.back_data.quality, bad_quality)
            back_good_quality = np.bitwise_not(back_bad_flag)
            if not np.all(good_quality == back_good_quality):
                warnings.warn(
                    f'bad channels of {self.name} data are defined by the '
                    'union of spectrum and background bad channels',
                    Warning,
                    stacklevel=2,
                )
                good_quality = np.bitwise_and(good_quality, back_good_quality)

        if not np.any(good_quality):
            raise RuntimeError(f'no good channel is found for {name} data')

        self._good_quality = good_quality
        self._keep_channel_info = bool(keep_channel_info)

        self._initialized = False
        self.set_erange(erange)
        self.set_grouping(self.spec_data.grouping)
        self._initialized = True

    def _get_channel_mask(
        self, channel_emin: NDArray, channel_emax: NDArray
    ) -> NDArray:
        """Return channel mask given channel energy grids."""
        emin = np.expand_dims(self._erange[:, 0], axis=1)
        emax = np.expand_dims(self._erange[:, 1], axis=1)
        mask1 = np.less_equal(emin, channel_emin)
        mask2 = np.less_equal(channel_emax, emax)
        return np.bitwise_and(mask1, mask2)

    def set_erange(self, erange: list | tuple):
        """Set energy range of interest.

        Parameters
        ----------
        erange : array_like
            Energy range of interest in keV, e.g.,
            ``erange=[(0.5, 2), (5, 200)]``.
        """
        erange = np.array(erange, dtype=np.float64, order='C', ndmin=2)

        # check if erange is increasing
        if np.any(np.diff(erange, axis=1) <= 0.0):
            raise ValueError('erange must be increasing')

        # check if erange is overlapped
        erange = erange[erange[:, 0].argsort()]
        if np.any(np.diff(np.hstack(erange)) <= 0.0):
            raise ValueError('erange must not be overlapped')

        self._erange = erange

        if self._initialized:
            self.set_grouping(self.grouping)

    def set_grouping(self, grouping: NDArray | None):
        """Set grouping flags.

        First group the spectrum and background accordind to the grouping
        flags, then ignore the channel groups falling outside the energy
        range of interest.

        Parameters
        ----------
        grouping : ndarray
            The grouping flags. If None, clear current grouping flags.
        """
        if grouping is None:
            grouping = np.ones(self.resp_data.channel_number, dtype=np.int64)
        else:
            grouping = np.asarray(grouping)

        if grouping.shape != (self.resp_data.channel_number,):
            raise ValueError(
                'grouping must have the same size as the number of channels'
            )

        self._grouping = grouping
        quality = self.good_quality

        spec_counts, spec_errors = self.spec_data.group(grouping, quality)

        channel_emin, channel_emax, matrix, channels = self.resp_data.group(
            grouping, quality, self.keep_channel_info
        )

        ch = self.resp_data.channel_type
        prefix = f'{self.name}_{ch}'
        groups_channel = np.array([f'{prefix}_{i}' for i in channels])
        channel_mask = self._get_channel_mask(channel_emin, channel_emax)
        channel_mask = np.any(channel_mask, axis=0)

        # response attribute
        self._channel = groups_channel[channel_mask]
        self._channel_emin = channel_emin[channel_mask]
        self._channel_emax = channel_emax[channel_mask]
        self._channel_emid = 0.5 * (self.channel_emin + self.channel_emax)
        self._channel_emean = np.sqrt(self.channel_emin * self.channel_emax)
        self._channel_width = self.channel_emax - self.channel_emin
        emean = self.channel_emean
        self._channel_errors = np.abs(
            [self.channel_emin - emean, self.channel_emax - emean]
        )
        self._resp_matrix = matrix.tocsc()[:, channel_mask]

        # spectrum attribute
        self._spec_counts = spec_counts[channel_mask]
        self._spec_errors = spec_errors[channel_mask]

        # background attribute
        if self.has_back:
            back_counts, back_errors = self.back_data.group(grouping, quality)
            self._back_counts = back_counts[channel_mask]
            self._back_errors = back_errors[channel_mask]
        else:
            self._back_counts = None
            self._back_errors = None

        # net spectrum attribute
        unit = 1.0 / (self.channel_width * self.spec_data.exposure)
        if self.has_back:
            net_counts = self.spec_counts - self.back_ratio * self.back_counts
            net_vars = np.square(self.spec_errors)
            net_vars += np.square(self.back_ratio * self.back_errors)
            net_errors = np.sqrt(net_vars)
            ce = net_counts * unit
            ce_errors = net_errors * unit
            self._net_counts = net_counts
            self._net_errors = net_errors
            self._ce = ce
            self._ce_errors = ce_errors

        else:
            self._net_counts = self.spec_counts
            self._net_errors = self.spec_errors
            self._ce = self.net_counts * unit
            self._ce_errors = self.spec_errors * unit

    def group(self, method: str, scale: float | None = None):
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
            fwhm = self.resp_data.channel_fwhm
        else:
            fwhm = None

        channel_emin = self.resp_data.channel_emin
        channel_emax = self.resp_data.channel_emax
        channel_mask = self._get_channel_mask(channel_emin, channel_emax)
        spec_counts = self.spec_data.counts
        spec_errors = self.spec_data.errors
        if self.has_back:
            back_ratio = self.back_ratio
            back_counts = self.back_data.counts
            back_errors = self.back_data.errors
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
                    d = j[mask]
                    if i not in all_good:
                        d = d * self._good_quality[mask]
                        if method == 'const':
                            d = d.sum()
                    data.append(d)
            grouping_flag, grouping_success = group_func(*data, scale)
            grouping[mask] = grouping_flag
            return grouping_success

        def apply_map(func, *args, all_good=()):
            """Map the apply function and return a success flag."""
            flags = [
                apply_grouping(func, mask, args, all_good)
                for mask in channel_mask
            ]
            return all(flags)

        if method == 'const':
            success = apply_map(group_const, np.ones(spec_counts.size, int))

        elif method == 'min':
            success = apply_map(group_min, spec_counts)

        elif method == 'sig':
            if self.spec_data.poisson:
                if self.back_data.poisson:
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
            if not (self.has_back and self.back_data.poisson):
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
            if self.spec_data.poisson:
                if self.back_data.poisson:
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
            if not (self.has_back and self.back_data.poisson):
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
                GroupingWarning,
            )

        self.set_grouping(grouping)

    def plot_spec(
        self,
        xlog: bool = True,
        data_ylog: bool = True,
        sig_ylog: bool = False,
    ) -> plt.Figure:
        """Plot the spectrum.

        .. warning::
            The significance plot is accurate only if the spectrum data has
            enough count statistics.

        Parameters
        ----------
        xlog : bool, optional
            Whether to use log scale on x-axis. The default is True.
        data_ylog : bool, optional
            Whether to use log scale on y-axis in spectral plot.
            The default is True.
        sig_ylog : bool, optional
            Whether to use log scale on y-axis in significance plot.
            The default is False.

        Returns
        -------
        plt.Figure
            The figure object.
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

        axs[0].set_ylabel(r'$C_E\ \mathrm{[count\ s^{-1}\ keV^{-1}]}$')
        axs[1].set_ylabel(r'Significance [$\mathrm{\sigma}$]')
        axs[1].set_xlabel(r'$\mathrm{Energy\ [keV]}$')

        if self.has_back:
            colors = get_colors(2, 'colorblind')
        else:
            colors = get_colors(1, 'colorblind')

        if xlog:
            x = self.channel_emean
            xerr = self.channel_errors
        else:
            x = self.channel_emid
            xerr = 0.5 * self.channel_width

        factor = 1.0 / (self.spec_data.exposure * self.channel_width)

        axs[0].errorbar(
            x=x,
            xerr=xerr,
            y=self.spec_counts * factor,
            yerr=self.spec_errors * factor,
            fmt='o',
            color=colors[0],
            alpha=0.8,
            label='Total',
            ms=3,
            mfc='#FFFFFFCC',
        )

        if self.has_back:
            back_factor = self.back_ratio * factor
            axs[0].errorbar(
                x=x,
                xerr=xerr,
                y=self.back_counts * back_factor,
                yerr=self.back_errors * back_factor,
                fmt='s',
                color=colors[1],
                alpha=0.8,
                label='Background',
                ms=3,
                mfc='#FFFFFFCC',
            )

        if self.spec_data.poisson:
            if self.has_back:
                if self.back_data.poisson:
                    sig = significance_lima(
                        self.spec_counts, self.back_counts, self.back_ratio
                    )
                else:
                    sig = significance_gv(
                        self.spec_counts,
                        self.back_counts,
                        self.back_errors,
                        self.back_ratio,
                    )
                sig *= np.sign(self.net_counts)
            else:
                sig = np.zeros_like(self.net_counts, dtype=np.float64)
                mask = self.net_counts > 0
                sig[mask] = self.net_counts[mask] / self.net_errors[mask]
        else:
            sig = self.net_counts / self.net_errors

        axs[1].errorbar(
            x=x,
            xerr=xerr,
            y=sig,
            yerr=np.where(sig != 0.0, 1.0, 0.0),
            fmt='o',
            color=colors[0],
            alpha=0.8,
            ms=3,
            mfc='#FFFFFFCC',
        )
        axs[1].axhline(0, color='k', ls=':', lw=0.5)

        if xlog:
            axs[0].set_xscale('log')
        if data_ylog:
            axs[0].set_yscale('log')
        if sig_ylog:
            axs[1].set_yscale('log')

        axs[0].legend()
        axs[0].set_title(self.name)

        return fig

    def plot_effective_area(
        self, hatch: bool = True, ylog: bool = True
    ) -> plt.Figure:
        """Plot the effective area.

        Parameters
        ----------
        hatch : bool, optional
            Whether to add hatches in the ignored region. The default is True.
        ylog : bool, optional
            Whether to use log scale on y-axis. The default is True.

        Returns
        -------
        plt.Figure
            The figure object.
        """
        return self.resp_data.plot_effective_area(
            noticed_range=self._erange if hatch else None,
            good_quality=self._good_quality,
            ylog=ylog,
        )

    def plot_matrix(self, hatch: bool = True, norm: str = 'log') -> plt.Figure:
        """Plot the response matrix.

        Parameters
        ----------
        hatch : bool, optional
            Whether to add hatches in the ignored region. The default is True.
        norm : str, optional
            Colorbar normalization method. The default is ``'log'``.

        Returns
        -------
        plt.Figure
            The figure object.
        """
        return self.resp_data.plot_matrix(
            noticed_range=self._erange if hatch else None,
            good_quality=self._good_quality,
            norm=norm,
        )

    def get_fixed_data(self) -> FixedData:
        """Return a fixed data object."""

        def copy(x: NDArray) -> NDArray:
            return np.copy(x, order='C')

        return FixedData(
            name=self.name,
            spec_counts=copy(self.spec_counts),
            spec_errors=copy(self.spec_errors),
            spec_poisson=self.spec_data.poisson,
            spec_exposure=self.spec_data.exposure,
            area_scale=self.spec_data.area_scale,
            has_back=self.has_back,
            back_counts=copy(self.back_counts) if self.has_back else None,
            back_errors=copy(self.back_errors) if self.has_back else None,
            back_poisson=self.back_data.poisson if self.has_back else None,
            back_exposure=self.back_data.exposure if self.has_back else None,
            back_ratio=self.back_ratio if self.has_back else None,
            net_counts=copy(self.net_counts),
            net_errors=copy(self.net_errors),
            ce=copy(self.ce),
            ce_errors=copy(self.ce_errors),
            photon_egrid=copy(self.resp_data.photon_egrid),
            channel=copy(self.channel),
            channel_emin=copy(self.channel_emin),
            channel_emax=copy(self.channel_emax),
            channel_emid=copy(self.channel_emid),
            channel_width=copy(self.channel_width),
            channel_emean=copy(self.channel_emean),
            channel_errors=copy(self.channel_errors),
            response_matrix=copy(self.response_matrix),
            sparse_matrix=self.sparse_matrix.tocoo(copy=True),
            response_sparse=self.resp_data.sparse,
        )

    @property
    def name(self) -> str:
        """Name of the observation data."""
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = str(name)

    @property
    def erange(self) -> list[list[float]]:
        """Energy range of interest."""
        return self._erange.tolist()

    @property
    def spec_data(self) -> SpectrumData:
        """The spectrum data of the observation."""
        return self._spec_data

    @property
    def resp_data(self) -> ResponseData:
        """The response matrix data of the observation."""
        return self._resp_data

    @property
    def has_back(self) -> bool:
        """Whether the observation has background."""
        return self._has_back

    @property
    def back_data(self) -> SpectrumData | None:
        """The background data of the observation."""
        return self._back_data

    @property
    def spec_exposure(self) -> float:
        """Spectrum exposure."""
        return self.spec_data.exposure

    @property
    def spec_poisson(self) -> bool:
        """Whether spectrum data follows Poisson counting statistics."""
        return self.spec_data.poisson

    @property
    def area_scale(self) -> float:
        """Area scaling factor."""
        return self.spec_data.area_scale

    @property
    def back_ratio(self) -> NDArray | None:
        """Ratio of spectrum to background effective exposure."""
        return self._back_ratio

    @property
    def back_exposure(self) -> float | None:
        """Background exposure."""
        return self.back_data.exposure if self.has_back else None

    @property
    def back_poisson(self) -> bool | None:
        """Whether background data follows Poisson counting statistics."""
        return self.back_data.poisson if self.has_back else None

    @property
    def good_quality(self) -> NDArray:
        """Flags indicating which measurement channel to be used."""
        return self._good_quality

    @property
    def grouping(self) -> NDArray:
        """Current grouping flags of the observation data."""
        return self._grouping

    @property
    def keep_channel_info(self) -> bool:
        """Whether to keep channel information when grouping the data."""
        return self._keep_channel_info

    @property
    def spec_counts(self) -> NDArray:
        """Spectrum counts of grouped measuring channels."""
        return self._spec_counts

    @property
    def spec_errors(self) -> NDArray:
        """Uncertainty of grouped spectrum counts."""
        return self._spec_errors

    @property
    def net_counts(self) -> NDArray:
        """Net counts of grouped measuring channels."""
        return self._net_counts

    @property
    def net_errors(self) -> NDArray:
        """Uncertainty of net counts of grouped measuring channels."""
        return self._net_errors

    @property
    def ce(self) -> NDArray:
        """Grouped net counts per second per keV."""
        return self._ce

    @property
    def ce_errors(self) -> NDArray:
        """Uncertainty of grouped net counts per second per keV."""
        return self._ce_errors

    @property
    def back_counts(self) -> NDArray | None:
        """Background counts of grouped measuring channels."""
        return self._back_counts

    @property
    def back_errors(self) -> NDArray | None:
        """Uncertainty of background counts of grouped measuring channels."""
        return self._back_errors

    @property
    def channel_emin(self) -> NDArray:
        """Left edge of the grouped measurement channel energy grid."""
        return self._channel_emin

    @property
    def channel_emax(self) -> NDArray:
        """Right edge of the grouped measurement channel energy grid."""
        return self._channel_emax

    @property
    def channel(self) -> NDArray:
        """Grouped channel information."""
        return self._channel

    @property
    def channel_emid(self) -> NDArray:
        """Midpoint of the grouped measurement channel energy grid."""
        return self._channel_emid

    @property
    def channel_width(self) -> NDArray:
        """Width of the grouped measurement channel energy grid."""
        return self._channel_width

    @property
    def channel_emean(self) -> NDArray:
        """Geometric mean of the grouped measurement channel energy grid."""
        return self._channel_emean

    @property
    def channel_errors(self) -> NDArray:
        """Width between left/right edge and geometric mean of channel grid."""
        return self._channel_errors

    @property
    def sparse_matrix(self) -> csc_array:
        """Grouped response matrix in sparse representation."""
        return self._resp_matrix

    @property
    def response_matrix(self) -> NDArray:
        """Grouped response matrix."""
        return self._resp_matrix.todense()


class SpectrumData:
    """Spectrum data.

    Parameters
    ----------
    counts : array-like
        Spectrum counts in each measuring channel.
    errors : array-like
        Uncertainty of spectrum counts.
    poisson : bool
        Whether spectrum data follows counting statistics.
    exposure : float
        Spectrum exposure.
    quality : array-like, optional
        Data quality of each spectrum channel. The default is ``0`` for all
        channels. The possible values are:

            * ``0``: good
            * ``1``: defined bad by software
            * ``2``: defined dubious by software
            * ``5``: defined bad by user
            * ``-1``: reason for bad flag unknown

    grouping : array-like, optional
        The grouping flag. When grouping the spectrum, channel with a
        grouping flag of 1 with all successive channels with grouping
        flags of -1 will be combined. The default is ``1`` for all channels.
    area_scale : float or array-like, optional
        Area scaling factor. The default is 1.0.
    back_scale : float or array-like, optional
        Background scaling factor. The default is 1.0.
    sys_errors : float or array-like, optional
        Systematic errors. If scalar, it will be applied to all channels.
        The default is 0.0.
    zero_errors_warning : bool, optional
        Whether to warn about zero errors when `poisson` is False.
        The default is True.
    non_int_warning : bool, optional
        Whether to warn about non-integer counts when `poisson` is True.
        The default is True.
    sys_errors_warning : bool, optional
        Whether to warn about non-zero systematic errors when `poisson` is
        True. The default is True.
    """

    def __init__(
        self,
        counts: NDArray,
        errors: NDArray,
        poisson: bool,
        exposure: float,
        quality: NDArray | None = None,
        grouping: NDArray | None = None,
        area_scale: float | NDArray = 1.0,
        back_scale: float | NDArray = 1.0,
        sys_errors: float | NDArray = 0.0,
        zero_errors_warning: bool = True,
        non_int_warning: bool = True,
        sys_errors_warning: bool = True,
    ):
        counts_shape = np.shape(counts)
        errors_shape = np.shape(errors)

        if not (len(counts_shape) == len(errors_shape) == 1):
            raise ValueError('spec_counts and spec_errors must be 1D arrays')

        if counts_shape != errors_shape:
            raise ValueError(
                'spec_counts and spec_errors must have the same shape'
            )

        if quality is None:
            quality = np.zeros(len(counts), dtype=np.int64)

        if np.shape(quality) != counts_shape:
            raise ValueError('quality must have the same size as counts')

        if grouping is None:
            grouping = np.ones(len(counts), dtype=np.int64)

        if np.shape(grouping) != counts_shape:
            raise ValueError('grouping must have the same size as counts')

        if np.shape(area_scale) != ():
            raise NotImplementedError('area scale array not supported yet')

        if np.shape(back_scale) != ():
            raise NotImplementedError('area scale array not supported yet')

        sys_errors_shape = np.shape(sys_errors)
        if sys_errors_shape != () and sys_errors_shape != counts_shape:
            raise ValueError(
                'sys_errors must be a scalar or have the same size as counts'
            )

        counts = np.array(counts, dtype=np.float64, order='C')
        errors = np.array(errors, dtype=np.float64, order='C')
        quality = np.array(quality, dtype=np.int64, order='C')
        grouping = np.array(grouping, dtype=np.int64, order='C')

        poisson = bool(poisson)
        exposure = float(exposure)
        area_scale = float(area_scale)
        back_scale = float(back_scale)
        sys_errors = np.full(counts_shape, sys_errors, dtype=np.float64)

        if not np.all(np.isin(quality, [0, 1, 2, 5, -1])):
            raise ValueError('quality must be 0, 1, 2, 5, or -1')

        if poisson:
            if np.any(counts < 0):
                raise ValueError('Poisson counts must be non-negative')

            if non_int_warning:
                diff = np.abs(counts - np.round(counts))
                if np.any(diff > 1e-8 * counts):
                    warnings.warn(
                        'Poisson counts has non-integer data, which will '
                        'lead to incorrect result',
                        Warning,
                        stacklevel=2,
                    )

            if sys_errors_warning and np.any(sys_errors > 0):
                warnings.warn(
                    'systematic errors will be ignored for Poisson data',
                    Warning,
                    stacklevel=2,
                )

        if np.any(errors < 0):
            raise ValueError('errors must be non-negative')

        if np.any(sys_errors < 0):
            raise ValueError('sys_errors must be non-negative')

        if exposure <= 0:
            raise ValueError('exposure must be positive')

        if area_scale <= 0:
            raise ValueError('area_scale must be positive')

        if back_scale <= 0:
            raise ValueError('back_scale must be positive')

        if not poisson and zero_errors_warning and np.any(errors == 0):
            warnings.warn(
                'zero errors will lead to incorrect result under '
                'Gaussian statistics, consider grouping the spectrum',
                Warning,
                stacklevel=2,
            )

        if not poisson:
            vars = np.square(errors)
            sys_vars = np.square(sys_errors * counts)
            errors = np.sqrt(vars + sys_vars)

        self._counts = counts
        self._errors = errors
        self._quality = quality
        self._grouping = grouping
        self._poisson = poisson
        self._exposure = exposure
        self._area_scale = area_scale
        self._back_scale = back_scale

    def group(
        self,
        grouping: NDArray,
        quality: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Group the spectrum.

        Parameters
        ----------
        grouping : ndarray
            Channel with a grouping flag of 1 with all successive channels
            with grouping flags of -1 are combined.
        quality : ndarray, optional
            Flag indicating which channel to be used in grouping.

        Returns
        -------
        counts : ndarray
            Grouped spectrum counts. The values of groups filled with bad
            channels are automatically dropped.
        errors : ndarray
            Uncertainty of grouped spectrum counts. The values of groups
            filled with bad channels are automatically dropped.
        """

        if np.shape(grouping) != np.shape(self.counts):
            raise ValueError('grouping must have the same size as counts')

        if quality is None:
            quality = np.ones(self.counts.shape)

        if np.size(quality) != np.size(self.counts):
            raise ValueError('quality must have the same size as counts')

        factor = quality.astype(np.float64)
        group_index = np.flatnonzero(grouping != -1)  # transform to index

        counts = np.add.reduceat(factor * self.counts, group_index)
        vars = np.add.reduceat(factor * self.errors * self.errors, group_index)
        errors = np.sqrt(vars)

        mask = np.add.reduceat(factor, group_index) > 0.0

        return counts[mask], errors[mask]

    @property
    def counts(self) -> NDArray:
        """Spectrum counts in each measuring channel."""
        return self._counts

    @property
    def errors(self) -> NDArray:
        """Uncertainty of spectrum counts."""
        return self._errors

    @property
    def quality(self) -> NDArray:
        """Data quality of each spectrum channel."""
        return self._quality

    @property
    def grouping(self) -> NDArray:
        """Grouping flags of the spectrum."""
        return self._grouping

    @property
    def poisson(self) -> bool:
        """Whether spectrum data follows counting statistics."""
        return self._poisson

    @property
    def exposure(self) -> float:
        """Spectrum exposure."""
        return self._exposure

    @property
    def area_scale(self) -> float:
        """Area scaling factor."""
        return self._area_scale

    @property
    def back_scale(self) -> float:
        """Background scaling factor."""
        return self._back_scale


class ResponseData:
    """Response matrix data.

    Parameters
    ----------
    photon_egrid : array-like
        Photon energy array of response matrix, must be increasing.
    channel_emin : array-like
        Left edge of the measurement channel energy array, must be increasing.
    channel_emax : array-like
        Right edge of the measurement channel energy array, must be increasing.
    response_matrix : array-like or sparse matrix
        Response matrix, the shape is (len(photon_egrid), len(channel)). This
        can be a sparse matrix.
    channel : array-like
        Measurement channel information.
    channel_type : str, optional
        Measurement channel type, e.g. `'PI'`. The default is 'Ch'.
    sparse : bool, optional
        Whether the response matrix is sparse. The default is False.
    """

    def __init__(
        self,
        photon_egrid: NDArray,
        channel_emin: NDArray,
        channel_emax: NDArray,
        response_matrix: NDArray | sparray,
        channel: NDArray,
        channel_type: str = 'Ch',
        sparse: bool = False,
    ):
        photon_egrid = np.array(photon_egrid, dtype=np.float64, order='C')
        channel_emin = np.array(channel_emin, dtype=np.float64, order='C')
        channel_emax = np.array(channel_emax, dtype=np.float64, order='C')
        response_matrix = coo_array(response_matrix)
        response_matrix.eliminate_zeros()
        channel = np.array(channel, dtype=str, order='C')

        photon_egrid_shape = np.shape(photon_egrid)
        emin_shape = np.shape(channel_emin)
        emax_shape = np.shape(channel_emax)
        channel_shape = np.shape(channel)
        response_shape = np.shape(response_matrix)

        if len(photon_egrid_shape) != 1:
            raise ValueError('photon_egrid must be a 1D array')

        if len(emin_shape) != 1:
            raise ValueError('channel_emin must be a 1D array')

        if len(emax_shape) != 1:
            raise ValueError('channel_emax must be a 1D array')

        if len(channel_shape) != 1:
            raise ValueError('channel must be a 1D array')

        if not (emin_shape == emax_shape == channel_shape):
            raise ValueError(
                f'size of channel_emin ({emin_shape[0]}), channel_emax '
                f'({emax_shape[0]}), and channel ({channel_shape[0]}) are'
                'not equal'
            )

        if len(response_shape) != 2:
            raise ValueError('resp_matrix must be a 2D array')

        if (
            photon_egrid_shape[0] - 1 != response_shape[0]
            or channel_shape[0] != response_shape[1]
        ):
            raise ValueError(
                f'shape of response_matrix {response_shape} is not matched to '
                f'photon energy channel ({photon_egrid_shape[0] - 1}) and '
                f'measurement channel ({channel_shape[0]})'
            )

        if np.any(channel_emin > channel_emax):
            raise ValueError('channel energy grids are not increasing')

        if np.any(photon_egrid[:-1] >= photon_egrid[1:]):
            raise ValueError('photon energy grids are not increasing')

        self._photon_egrid = photon_egrid
        self._channel_egrid = np.column_stack([channel_emin, channel_emax])
        self._channel = channel
        self._channel_type = str(channel_type)
        self._response_matrix = response_matrix
        self._sparse = bool(sparse)
        self._fwhm: NDArray | None = None
        self._channel_fwhm: NDArray | None = None

    def group(
        self,
        grouping: NDArray,
        quality: NDArray | None = None,
        keep_channel_info: bool = False,
    ) -> tuple[NDArray, NDArray, coo_array, NDArray]:
        """Group the response matrix.

        Parameters
        ----------
        grouping : ndarray
            Channel with a grouping flag of 1 with all successive channels
            with grouping flags of -1 are combined.
        quality : ndarray, optional
            Flag indicating which channel to be used in grouping.

        Returns
        -------
        group_emin : ndarray
            Left edge of the grouped channel energy array. The values of groups
            filled with bad channels are automatically dropped.
        group_emax : ndarray
            Right edge of the grouped channel energy array. The values of
            groups filled with bad channels are automatically dropped.
        matrix : coo_array
            Grouped response matrix. The values of groups filled with bad
            channels are automatically dropped.
        channel : ndarray
            Grouped channel information. The values of groups filled with bad
            channels are automatically dropped.
        keep_channel_info : bool, optional
            Whether to keep channel information when grouping the response.
            The default is False.
        """
        channel_number = self.channel_number
        if np.shape(grouping) != (channel_number,):
            raise ValueError('grouping must have the same size as channel')

        if quality is None:
            quality = np.ones(channel_number)

        if np.shape(quality) != (channel_number,):
            raise ValueError('quality must have the same size as channel')

        group_channels = np.array(self.channel, dtype=str)
        group_emin = self.channel_emin
        group_emax = self.channel_emax
        matrix = self.sparse_matrix

        group_index = np.flatnonzero(grouping != -1)  # transform to index

        if len(group_index) == channel_number:  # apply good mask only
            if np.count_nonzero(quality) != quality.size:
                group_channels = np.array(group_channels[quality], dtype=str)
                group_emin = group_emin[quality]
                group_emax = group_emax[quality]
                matrix = self.sparse_matrix.tocsc()[:, quality]

        else:
            mask = np.add.reduceat(quality, group_index) > 0

            edge_indices = np.append(group_index, channel_number)
            raw_channel = self.channel.tolist()
            emin = self.channel_emin
            emax = self.channel_emax
            group_channels = []
            group_emin = []
            group_emax = []

            for i in range(len(group_index)):
                if not mask[i]:
                    continue
                slice_i = slice(edge_indices[i], edge_indices[i + 1])
                quality_slice = quality[slice_i]
                channel_slice = np.array(raw_channel[slice_i])[quality_slice]
                if keep_channel_info:
                    group_channels.append(np.array('+'.join(channel_slice)))
                else:
                    group_channels.append(str(i))
                group_emin.append(min(emin[slice_i]))
                group_emax.append(max(emax[slice_i]))

            group_channels = np.array(group_channels)
            group_emin = np.array(group_emin)
            group_emax = np.array(group_emax)

            a = np.where(quality, 1.0, 0.0)
            idx = np.arange(channel_number)
            ptr = np.append(group_index, channel_number)
            grouping_matrix = csc_array((a, idx, ptr))
            matrix = self.sparse_matrix.dot(grouping_matrix)
            matrix = matrix.tocsc()[:, mask]

        return group_emin, group_emax, coo_array(matrix), group_channels

    def plot_effective_area(
        self,
        noticed_range: NDArray | None = None,
        good_quality: NDArray | None = None,
        ylog: bool = True,
    ) -> plt.Figure:
        """Plot the response matrix.

        Parameters
        ----------
        noticed_range : ndarray, optional
            Energy range to show. Other energy ranges will be hatched.
        good_quality : ndarray, optional
            Flags indicating which measurement channel to be used in plotting.
            It Must be the same length as the number of channels.
        ylog : bool, optional
            Whether to use log scale on y-axis. The default is True.

        Returns
        -------
        fig : plt.Figure
            The figure object.
        """
        if good_quality is None:
            eff_area = self.sparse_matrix.sum(axis=1)
        else:
            if len(good_quality) != self.sparse_matrix.shape[1]:
                raise ValueError(
                    'length of good_quality must match to number of channels'
                )
            factor = np.array(good_quality, dtype=bool)
            eff_area = (self.sparse_matrix * factor).sum(axis=1)
        eff_area = np.clip(eff_area, a_min=0.0, a_max=None)

        fig = plt.figure()
        plt.rcParams['axes.formatter.min_exponent'] = 3
        plt.step(self.photon_egrid, np.append(eff_area, eff_area[-1]))
        plt.xlim(self.photon_egrid[0], self.photon_egrid[-1])
        plt.xlabel('Photon Energy [keV]')
        plt.ylabel('Effective Area [cm$^2$]')
        plt.xscale('log')
        if ylog:
            plt.yscale('log')

        if noticed_range is not None:
            ph_emin = self.photon_egrid[:-1]
            ph_emax = self.photon_egrid[1:]
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

        return fig

    def plot_matrix(
        self,
        noticed_range: NDArray | None = None,
        good_quality: NDArray | None = None,
        norm: str = 'log',
    ) -> plt.Figure:
        """Plot the response matrix.

        Parameters
        ----------
        noticed_range : ndarray, optional
            Energy range to show. Other energy ranges will be hatched.
        good_quality : ndarray, optional
            Flags indicating which measurement channel to be used in plotting.
            It Must be the same length as the number of channels.
        norm : str, optional
            Colorbar normalization method. The default is ``'log'``.

        Returns
        -------
        fig : plt.Figure
            The figure object.
        """
        channel_emin = self.channel_emin
        channel_emax = self.channel_emax
        matrix = self.sparse_matrix

        if good_quality is not None:
            if len(good_quality) != matrix.shape[1]:
                raise ValueError(
                    'length of good_quality must match to number of channels'
                )

            good_quality = np.array(good_quality, dtype=bool)
            channel_emin = channel_emin[good_quality]
            channel_emax = channel_emax[good_quality]
            matrix = matrix.tocsc()[:, good_quality]

        if norm == 'log':
            a_min = matrix.max() * 1e-5
        else:
            a_min = 0.0
        matrix = np.clip(matrix.todense(), a_min=a_min, a_max=None)

        # some response matrix has discontinuity in channel energy grid,
        # insert np.nan to handle this
        idx = np.flatnonzero(channel_emin[1:] - channel_emax[:-1])
        if len(idx) > 0:
            idx2 = idx + 1
            channel_emin = np.insert(channel_emin, idx2, channel_emax[idx])
            channel_emax = np.insert(channel_emax, idx2, channel_emin[idx2])
            matrix = np.insert(matrix, idx2, np.nan, axis=1)

        channel_egrid = np.append(channel_emin, channel_emax[-1])
        ch, ph = np.meshgrid(channel_egrid, self.photon_egrid)
        fig = plt.figure()
        plt.rcParams['axes.formatter.min_exponent'] = 3
        plt.pcolormesh(ch, ph, matrix, cmap='magma', norm=norm)
        plt.xlabel('Measurement Energy [keV]')
        plt.ylabel('Photon Energy [keV]')
        plt.colorbar(label='Effective Area [cm$^2$]')
        plt.xscale('log')
        plt.yscale('log')

        if noticed_range is not None:
            noticed_range = np.atleast_2d(noticed_range)
            emin = np.expand_dims(noticed_range[:, 0], axis=1)
            emax = np.expand_dims(noticed_range[:, 1], axis=1)
            mask1 = np.less_equal(emin, channel_emin)
            mask2 = np.less_equal(channel_emax, emax)
            idx = [np.flatnonzero(i) for i in np.bitwise_and(mask1, mask2)]

            ignored = []
            if channel_emin[idx[0][0]] > channel_emin[0]:
                ignored.append((channel_emin[0], channel_emin[idx[0][0]]))
            for i in range(len(idx) - 1):
                this_noticed_right = channel_emax[idx[i][-1]]
                next_noticed_left = channel_emin[idx[i + 1][0]]
                ignored.append((this_noticed_right, next_noticed_left))
            if channel_emax[idx[-1][-1]] < channel_emax[-1]:
                ignored.append((channel_emax[idx[-1][-1]], channel_emax[-1]))

            y = (self.photon_egrid[0], self.photon_egrid[-1])
            for i in ignored:
                plt.fill_betweenx(y, *i, alpha=0.4, color='w', hatch='x')

        return fig

    @property
    def photon_egrid(self) -> NDArray:
        """Photon energy grid of response matrix."""
        return self._photon_egrid

    @property
    def channel_emin(self) -> NDArray:
        """Left edge of measurement energy grid."""
        return self._channel_egrid[:, 0]

    @property
    def channel_emax(self) -> NDArray:
        """Right edge of measurement energy grid."""
        return self._channel_egrid[:, 1]

    @property
    def channel(self) -> NDArray:
        """Measurement channel information."""
        return self._channel

    @property
    def channel_type(self) -> str:
        """Measurement channel type."""
        return self._channel_type

    @property
    def channel_number(self) -> int:
        """Number of channels."""
        return len(self.channel)

    @property
    def matrix(self) -> NDArray:
        """Response matrix."""
        return self._response_matrix.todense()

    @property
    def sparse(self) -> bool:
        """Whether the response matrix is sparse."""
        return self._sparse

    @property
    def sparse_matrix(self) -> coo_array:
        """Sparse response matrix."""
        return self._response_matrix

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

        matrix = self.sparse_matrix
        csr_matrix = matrix.tocsr()
        nE, nC = matrix.shape
        row_idx = np.arange(nE)
        argmax = np.squeeze(matrix.argmax(axis=1))
        max_value = csr_matrix[row_idx, argmax]
        half_max = 0.5 * max_value

        # Initialize left and right index arrays
        idx_low_high = np.column_stack([argmax - 1, argmax + 1])

        # Now find the left and right indices where the response falls below
        # half_max
        dense_matrix = matrix.todense()
        for iE in range(nE):
            # elements of iE row and idx cols are <= half_max
            idx = np.flatnonzero(dense_matrix[iE] <= half_max[iE])
            if idx.size:
                k = np.searchsorted(idx, argmax[iE])
                if k > 0:
                    idx_low_high[iE][0] = idx[k - 1]
                if k < idx.size:
                    idx_low_high[iE][1] = idx[k]
        ilow, ihigh = np.clip(idx_low_high, 0, nC - 1).T

        good_low = csr_matrix[row_idx, [0] * nE] <= half_max
        good_high = csr_matrix[row_idx, [-1] * nE] <= half_max
        fwhm = np.full(nE, 0)
        fwhm[good_high] += ihigh[good_high] - argmax[good_high]
        fwhm[good_low] += argmax[good_low] - ilow[good_low]
        fwhm[(good_high & (~good_low)) | ((~good_high) & good_low)] *= 2

        # Ensure minimum FWHM of 1
        fwhm = np.clip(fwhm, 1, None)

        # Set FWHM to -1 for rows without any valid half max
        fwhm[~(good_high | good_low)] = -1

        self._fwhm = fwhm
        return self._fwhm

    @property
    def channel_fwhm(self) -> NDArray:
        """Estimated Full Width at Half Maximum (FWHM) in channel energy space.

        .. note::
            The calculation code is translated from ``heasp``. The calculation
            interpolates the `estimated_fwhm` using the nominal channel energy
            to give the FWHM for each channel. Assuming that FWHM does not
            change significantly over the channel, so just find the FWHM at the
            center energy of the channel.
        """
        if self._channel_fwhm is not None:
            return self._channel_fwhm

        fwhm = self.fwhm
        channel_emid = self._channel_egrid.mean(axis=1)
        idx = np.searchsorted(self.photon_egrid, channel_emid) - 1
        idx = np.clip(idx, 0, len(fwhm) - 1)
        self._channel_fwhm = fwhm[idx]
        return self._channel_fwhm


class FixedData(NamedTuple):
    """Data to fit."""

    name: str
    """Name of the observation data."""

    spec_counts: NDArray
    """Spectrum counts in each measuring channel."""

    spec_errors: NDArray
    """Uncertainty of spectrum counts."""

    spec_poisson: bool
    """Whether spectrum data follows counting statistics."""

    spec_exposure: float
    """Spectrum exposure."""

    area_scale: float | NDArray
    """Area scaling factor."""

    has_back: bool
    """Whether spectrum data includes background."""

    back_counts: NDArray | None
    """Background counts in each measuring channel."""

    back_errors: NDArray | None
    """Uncertainty of background counts."""

    back_poisson: bool | None
    """Whether background data follows counting statistics."""

    back_exposure: np.float64 | None
    """Background exposure."""

    back_ratio: np.float64 | NDArray | None
    """Ratio of spectrum to background effective exposure."""

    net_counts: NDArray
    """Net counts in each measuring channel."""

    net_errors: NDArray
    """Uncertainty of net counts in each measuring channel."""

    ce: NDArray
    """Net counts per second per keV."""

    ce_errors: NDArray
    """Uncertainty of net counts per second per keV."""

    photon_egrid: NDArray
    """Photon energy grid of response matrix."""

    channel: NDArray
    """Measurement channel information."""

    channel_emin: NDArray
    """Left edge of measurement energy grid."""

    channel_emax: NDArray
    """Right edge of measurement energy grid."""

    channel_emid: NDArray
    """Middle of measurement energy grid."""

    channel_width: NDArray
    """Width of measurement energy grid."""

    channel_emean: NDArray
    """Geometric mean of measurement energy grid."""

    channel_errors: NDArray
    """Width between left/right and geometric mean of channel grid."""

    response_matrix: NDArray
    """Response matrix."""

    sparse_matrix: coo_array
    """Sparse response matrix."""

    response_sparse: bool
    """Whether the response matrix is sparse."""


class GroupingWarning(Warning):
    """Issued by grouping scale not being met for all channel groups."""
