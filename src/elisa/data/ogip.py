from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING

import numpy as np
from astropy.io import fits
from scipy.sparse import coo_array

from elisa.data.base import ObservationData, ResponseData, SpectrumData

if TYPE_CHECKING:
    NDArray = np.ndarray


class Data(ObservationData):
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
        Whether to ignore channels with ``QUALITY`` being 2 or 5.
        The default is True. The possible values for spectral ``QUALITY`` are

            * ``0``: good
            * ``1``: defined bad by software
            * ``2``: defined dubious by software
            * ``5``: defined bad by user
            * ``-1``: reason for bad flag unknown

    keep_channel_info : bool, optional
        Whether to keep channel information in the label of grouped
        channel. Takes effect only if `group` is not None or spectral data
        has ``GROUPING`` defined. The default is False.
    sparse_response : bool, optional
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
        scale: float | None = None,
        spec_poisson: bool | None = None,
        back_poisson: bool | None = None,
        ignore_bad: bool = True,
        keep_channel_info: bool = False,
        sparse_response: bool = False,
        # corrfile: bool | None = None,
        # corrnorm: bool | None = None,
    ):
        try:
            spec_data = Spectrum(specfile, spec_poisson)
        except PoissonFlagNotFoundError as err:
            raise PoissonFlagNotFoundError(
                f'"POISSERR" is undefined in header of {specfile}, '
                'spec_poisson must be set manually, i.e., '
                'Data(..., spec_poisson=True/False)'
            ) from err

        # check data name
        if name:
            name = str(name)
        elif spec_data.name:
            name = spec_data.name
        else:
            raise ValueError(
                f'name must be set manually for {specfile} data, i.e., '
                "Data(..., name='NAME')"
            )

        # check ancillary response file
        if not ancrfile:
            ancrfile = spec_data.ancrfile

        # check response file
        sparse_response = bool(sparse_response)
        if respfile:
            resp_data = Response(respfile, ancrfile, sparse_response)
        elif spec_data.respfile:
            resp_data = Response(spec_data.respfile, ancrfile, sparse_response)
        else:
            raise ValueError(
                f'response file must be set manually for {specfile} data, '
                "i.e., Data(..., respfile='/path/to/rsp.fits')"
            )

        if len(spec_data.counts) != resp_data.channel_number:
            raise ValueError(
                f'specfile ({specfile}) and respfile ({respfile}) are not '
                'matched'
            )

        # check background file
        try:
            if backfile:
                back_data = Spectrum(backfile, back_poisson)
            elif spec_data.backfile:
                back_data = Spectrum(spec_data.backfile, back_poisson)
            else:
                back_data = None
        except PoissonFlagNotFoundError as err:
            raise PoissonFlagNotFoundError(
                '"POISSERR" is undefined in header of background spectrum of '
                f'{specfile}, back_poisson must be set manually, i.e., '
                'Data(..., back_poisson=True/False)'
            ) from err

        if back_data and len(back_data.counts) != resp_data.channel_number:
            raise ValueError(
                f'specfile ({specfile}) and backfile ({backfile}) are not '
                'matched'
            )

        super().__init__(
            name=name,
            erange=erange,
            spec_data=spec_data,
            resp_data=resp_data,
            back_data=back_data,
            ignore_bad=bool(ignore_bad),
            keep_channel_info=bool(keep_channel_info),
        )

        if group is not None:
            self.group(group, scale)


class Spectrum(SpectrumData):
    """Load spectrum data in OGIP standards [1]_.

    Parameters
    ----------
    specfile : str
        Spectrum file path. For type II pha file, the row specifier must be
        given at the end of path, e.g., ``specfile='spec.pha2{1}'``.
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
            msg = 'row id must be provided for type II spectrum, i.e., '
            msg += f"'{specfile}{{N}}'"

            channel_number = len(data)
            if int(header.get('DETCHANS', channel_number)) != channel_number:
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
        if np.shape(sys_err) == ():
            sys_err = np.zeros(len(counts))
        else:
            sys_err = np.array(sys_err, dtype=np.float64, order='C')

        # get quality flag
        quality = get_field('QUALITY', 0)
        if np.shape(quality) == ():
            quality = np.zeros(len(counts), dtype=np.int64)
        else:
            quality = np.array(quality, dtype=np.int64, order='C')

        # get grouping flag
        grouping = get_field('GROUPING', 0)
        if np.shape(grouping) == ():
            grouping = np.ones(len(counts), np.int64)
        else:
            grouping = np.array(grouping, dtype=np.int64, order='C')

        # check data
        if poisson:
            # check if counts are integers
            diff = np.abs(counts - np.round(counts))
            if np.any(diff > 1e-8 * counts):
                warnings.warn(
                    f'Poisson spectrum {specfile} has non-integer counts, '
                    'which may lead to wrong result',
                    Warning,
                    stacklevel=3,
                )
        else:
            # check if statistical errors are positive
            if np.any(stat_err < 0.0):
                raise ValueError(
                    f'spectrum {specfile} has negative statistical errors'
                )

            if np.any(stat_err == 0.0):
                warnings.warn(
                    f'spectrum {specfile} has zero statistical errors, '
                    'which may lead to wrong result under Gaussian statistics,'
                    ' consider grouping the spectrum',
                    Warning,
                    stacklevel=3,
                )

            # check if systematic errors are non-negative
            if np.any(sys_err < 0.0):
                raise ValueError(
                    f'spectrum {specfile} has systematic errors < 0'
                )

        # total error of counts
        if not poisson:
            stat_var = np.square(stat_err)
            sys_var = np.square(sys_err * counts)
            errors = np.sqrt(stat_var + sys_var)
        else:
            if np.any(sys_err > 0.0):
                warnings.warn(
                    'systematic errors are ignored for Poisson spectrum '
                    f'{specfile}',
                    Warning,
                    stacklevel=3,
                )
            errors = stat_err

        # search name in header
        excluded_name = ('', 'none', 'unknown')
        for key in ('DETNAM', 'INSTRUME', 'TELESCOP'):
            name = str(header.get(key, ''))
            if name.lower() not in excluded_name:
                break
            else:
                name = ''
        self._name = str(name)

        # get the area scaling factor
        area_scale = get_field('AREASCAL', 1.0)
        if isinstance(area_scale, np.ndarray):
            area_scale = np.array(area_scale, dtype=np.float64, order='C')
        else:
            area_scale = float(area_scale)

        # get the background scaling factor
        back_scale = get_field('BACKSCAL', 1.0)
        if isinstance(back_scale, np.ndarray):
            back_scale = np.array(back_scale, dtype=np.float64, order='C')
        else:
            back_scale = float(back_scale)

        # get the correction scaling factor
        # self._corr_scale = np.float64(get_field('CORRSCAL', 0.0))

        self._header = header
        self._specfile = specfile
        excluded_file = ('none', 'None', 'NONE')

        # get backfile
        self._backfile = get_field('BACKFILE', '', excluded_file)

        # get respfile
        self._respfile = get_field('RESPFILE', '', excluded_file)

        # get ancrfile
        self._ancrfile = get_field('ANCRFILE', '', excluded_file)

        # get corrfile
        # self._corrfile = get_field('CORRFILE', '', excluded_file)

        super().__init__(
            counts=counts,
            errors=errors,
            poisson=poisson,
            exposure=exposure,
            quality=quality,
            grouping=grouping,
            area_scale=area_scale,
            back_scale=back_scale,
            sys_errors=sys_err,
            zero_errors_warning=False,
            non_int_warning=False,
            sys_errors_warning=False,
        )

    @property
    def name(self) -> str:
        """The name of the observation instrument."""
        return self._name

    @property
    def header(self) -> fits.Header:
        """The spectrum header."""
        return self._header

    @property
    def specfile(self) -> str:
        """The spectrum file path."""
        return self._specfile

    @property
    def backfile(self) -> str:
        """The background file path."""
        return self._backfile

    @property
    def respfile(self) -> str:
        """The response file path."""
        return self._respfile

    @property
    def ancrfile(self) -> str:
        """The ancillary response file path."""
        return self._ancrfile


class Response(ResponseData):
    """Load telescope response data in OGIP standards [1]_.

    Parameters
    ----------
    respfile : str
        Response file path.
    ancrfile : str or None, optional
        Ancillary response path. The default is None.
    sparse : bool, optional
        Whether the response matrix is sparse. The default is False.

    References
    ----------
    .. [1] `The Calibration Requirements for Spectral Analysis (Definition of
            RMF and ARF file formats) <https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002/cal_gen_92_002.html>`__
            and `Addendum: Changes log <https://heasarc.gsfc.nasa.gov/docs/heasarc/caldb/docs/memos/cal_gen_92_002a/cal_gen_92_002a.html>`__
    """

    def __init__(
        self,
        respfile: str,
        ancrfile: str | None = None,
        sparse: bool = False,
    ):
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

        response_data = self._read_response(file, resp_id)

        arf = self._read_arf()
        if arf is not None:
            if len(arf) != response_data['response_matrix'].shape[0]:
                raise ValueError(
                    f'rmf ({respfile}) and arf ({ancrfile}) are not matched'
                )
            response_data['response_matrix'] *= arf[:, None]

        # photon_egrid, sparse_matrix = self._drop_zeros(
        #     response_data['photon_egrid'],
        #     response_data['sparse_matrix']
        # )
        # response_data['photon_egrid'] = photon_egrid
        # response_data['sparse_matrix'] = sparse_matrix

        super().__init__(**response_data, sparse=sparse)

    def _read_response(self, file: str, response_id: int) -> dict:
        respfile = self.respfile
        with fits.open(file) as response_hdu:
            if 'MATRIX' in response_hdu:
                if self.ancrfile is None:
                    warnings.warn(
                        f'{file} is probably a rmf, '
                        'ancrfile (arf) maybe needed but not provided',
                        Warning,
                    )

                ext = ('MATRIX', response_id)

            elif 'SPECRESP MATRIX' in response_hdu:
                ext = ('SPECRESP MATRIX', response_id)

            else:
                raise ValueError(
                    f'cannot read response matrix data from {respfile}'
                )

            ebounds_data = response_hdu['EBOUNDS'].data
            response_header = response_hdu[ext].header
            response_data = response_hdu[ext].data

        channel_type = response_header.get('CHANTYPE', 'Ch')

        channel_emin = ebounds_data['E_MIN']
        channel_emax = ebounds_data['E_MAX']

        photon_emin = response_data['ENERG_LO']
        photon_emax = response_data['ENERG_HI']
        photon_egrid = np.append(photon_emin, photon_emax[-1])
        photon_egrid = np.asarray(photon_egrid, dtype=np.float64, order='C')

        # check and read response matrix
        channel_number = response_header.get('DETCHANS', None)
        if channel_number is None:
            raise ValueError(
                f'keyword "DETCHANS" is not found in "{respfile}" header'
            )
        else:
            channel_number = int(channel_number)

        fchan_idx = response_data.names.index('F_CHAN') + 1
        # set the first channel number to 1 if not found
        first_chan = int(response_header.get(f'TLMIN{fchan_idx}', 1))

        channel = tuple(
            str(c) for c in range(first_chan, first_chan + channel_number)
        )

        n_grp = response_data['N_GRP']
        f_chan = response_data['F_CHAN'] - first_chan
        n_chan = response_data['N_CHAN']

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
        for i in range(len(response_data)):
            n = int(n_grp[i])  # n channel subsets
            if n > 0:
                f = f_chan[i].astype(int)  # first channel numbers of subsets
                e = e_chan[i].astype(int)  # last channel numbers of subsets
                cols.extend(map(np.arange, f, e))
        cols = np.hstack(cols)

        matrix = response_data['MATRIX'].ravel()
        if matrix.dtype != np.dtype('O'):
            reduced_matrix = matrix
        else:
            reduced_matrix = np.hstack(matrix)

        sparse_matrix = coo_array(
            (reduced_matrix, (rows, cols)),
            shape=(len(response_data), channel_number),
        )
        sparse_matrix.eliminate_zeros()

        return {
            'photon_egrid': photon_egrid,
            'channel_emin': channel_emin,
            'channel_emax': channel_emax,
            'channel': channel,
            'channel_type': channel_type,
            'response_matrix': sparse_matrix,
        }

    def _read_arf(self) -> NDArray | None:
        if self.ancrfile:
            with fits.open(self.ancrfile) as arf_hdul:
                return arf_hdul['SPECRESP'].data['SPECRESP']

    @staticmethod
    def _drop_zeros(
        photon_egrid: NDArray, sparse_matrix: coo_array
    ) -> tuple[NDArray, coo_array]:
        """Remove leading or trailing rows filled with 0 from the matrix."""
        nonzero_rows = np.unique(sparse_matrix.nonzero()[0])
        nonzero_mask = np.isin(range(sparse_matrix.shape[0]), nonzero_rows)
        zero_mask = np.bitwise_not(nonzero_mask)
        if zero_mask.any():
            n_entries = len(photon_egrid) - 1
            last_idx = len(photon_egrid) - 2
            idx = np.flatnonzero(zero_mask)
            diff = np.diff(idx)
            if len(diff) == 0:  # only one zero entry
                idx = idx[0]
                if idx in (0, last_idx):  # check if idx is leading or trailing
                    if idx == 0:
                        photon_egrid = photon_egrid[1:]
                    else:
                        photon_egrid = photon_egrid[:-1]
            else:
                splits = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
                zeros_mask2 = np.full(n_entries, False)
                for s in splits:
                    if np.isin(s, (0, last_idx)).any():
                        # only drop leading or trailing part of grids
                        zeros_mask2[s] = True

                elo = photon_egrid[:-1][~zeros_mask2]
                ehi = photon_egrid[1:][~zeros_mask2]
                photon_egrid = np.append(elo, ehi[-1])

        return photon_egrid, sparse_matrix.tocsr()[~zero_mask]

    @property
    def respfile(self) -> str:
        """The response file path."""
        return self._respfile

    @property
    def ancrfile(self) -> str:
        """The ancillary response file path."""
        return self._ancrfile


class PoissonFlagNotFoundError(RuntimeError):
    """Issued by ``POISSERR`` not found in spectrum header."""
