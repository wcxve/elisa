import os
import warnings

import numpy as np
import xarray as xr
from astropy.io import fits

__all__ = ['Data']

class Data:
    # TODO: extract rsp2 file
    # Currently need to manually specified in respfile and ancrfile.
    def __init__(
        self, erange, specfile,
        backfile=None, respfile=None, ancrfile=None, name=None,
        is_spec_poisson=None, is_back_poisson=None,
        ignore_bad=True, keep_channel_info=False,
        group_type=None, group_scale=None
    ):
        self._extract_spec(specfile, is_spec_poisson)
        self._set_name(name)
        self._extract_back(backfile, is_back_poisson)
        self._extract_resp(respfile, ancrfile)
        self._filter_channel(
            erange, ignore_bad, keep_channel_info, group_type, group_scale
        )

    def _extract_spec(self, specfile, is_spec_poisson):
        if '{' in specfile and specfile[-1] == '}':
            self._spec_num = int(specfile.split('{')[1].split('}')[0]) - 1
            self._spec_typeii = True
            specfile = specfile.split('{')[0]
        else:
            self._spec_typeii = False

        if not os.path.exists(specfile):
            raise FileNotFoundError(f'spectrum file "{specfile}" not found')

        with fits.open(specfile) as spec_hdul:
            spec_header = spec_hdul['SPECTRUM'].header
            spec_data = spec_hdul['SPECTRUM'].data

        try:
            spec_poisson = spec_header['POISSERR']
        except Exception as e:
            print('SPECTRUM: ' + str(e))
            if is_spec_poisson is None:
                raise ValueError(
                    '`is_spec_poisson` must be provided when no "POISSERR" in'
                    'spectrum header'
                )
            spec_poisson = is_spec_poisson

        if self._spec_typeii:
            spec_data = spec_data[self._spec_num : self._spec_num + 1]
            spec_exposure = spec_data['EXPOSURE'][0]

            if 'COUNTS' in spec_data.names:
                spec_counts = spec_data['COUNTS'][0]
                if spec_poisson:
                    spec_error = np.sqrt(spec_counts)
                else:
                    spec_error = spec_data['STAT_ERR'][0]
            elif 'RATE' in spec_data.names:
                spec_counts = spec_data['RATE'][0] * spec_exposure
                if spec_poisson:
                    spec_error = np.sqrt(spec_counts)
                else:
                    spec_error = spec_data['STAT_ERR'][0] * spec_exposure
            else:
                raise ValueError(
                    f'Spectrum ({specfile}) is not stored in COUNTS or RATE'
                )

            if 'QUALITY' in spec_data.names:
                spec_quality = spec_data['QUALITY'][0]
            else:
                spec_quality = np.zeros(len(spec_counts))

            if 'GROUPING' in spec_data.names:
                grouping = np.flatnonzero(spec_data['GROUPING'][0] == 1)
            else:
                grouping = np.arange(len(spec_counts))

        else:
            spec_exposure = spec_header['EXPOSURE']

            if 'COUNTS' in spec_data.names:
                spec_counts = spec_data['COUNTS']
                if spec_poisson:
                    spec_error = np.sqrt(spec_counts)
                else:
                    spec_error = spec_data['STAT_ERR']
            elif 'RATE' in spec_data.names:
                spec_counts = spec_data['RATE'] * spec_exposure
                if spec_poisson:
                    spec_error = np.sqrt(spec_counts)
                else:
                    spec_error = spec_data['STAT_ERR'] * spec_exposure
            else:
                raise ValueError(
                    f'Spectrum ({specfile}) is not stored in COUNTS or RATE'
                )

            if 'QUALITY' in spec_data.names:
                spec_quality = spec_data['QUALITY']
            else:
                spec_quality = np.zeros(len(spec_counts))

            if 'GROUPING' in spec_data.names:
                grouping = np.flatnonzero(spec_data['GROUPING'] == 1)
            else:
                grouping = np.arange(len(spec_counts))


        if spec_poisson:  # check if counts are integers
            diff = np.abs(spec_counts - np.round(spec_counts))
            if np.any(diff > 1e-8 * spec_counts):
                warnings.warn(
                    f'spectrum ({specfile}) counts are not integers, which'
                    ' could lead to wrong result',
                    stacklevel=2
                )

        self.spec_exposure = spec_exposure
        self.spec_poisson = spec_poisson
        self._spec_header = spec_header
        self._spec_data = spec_data
        self._spec_counts = np.array(spec_counts, dtype=np.float64)
        self._spec_error = np.array(spec_error, dtype=np.float64)
        self._spec_quality = np.array(spec_quality, dtype=np.int8)
        self._grouping = np.array(grouping, dtype=np.int64)


    def _extract_back(self, backfile, is_back_poisson):
        if self._spec_typeii:
            backfile = backfile or self._spec_data['BACKFILE'][0]
        else:
            backfile = backfile or self._spec_header['BACKFILE']

        if backfile.lower() in ['none', '']:
            warnings.warn(
                f'assumes {self.name} has no background',
                stacklevel=2
            )
            self.has_back = False
            return None

        if '{' in backfile and backfile[-1] == '}':
            self._back_num = int(backfile.split('{')[1].split('}')[0]) - 1
            self._back_typeii = True
            backfile = backfile.split('{')[0]
        else:
            self._back_typeii = False

        if not os.path.exists(backfile):
            raise FileNotFoundError(f'background file "{backfile}" not found')

        with fits.open(backfile) as back_hdul:
            back_header = back_hdul['SPECTRUM'].header
            back_data = back_hdul['SPECTRUM'].data

        try:
            back_poisson = back_header['POISSERR']
        except Exception as e:
            print('BACKGROUND: ' + str(e))
            if is_back_poisson is None:
                raise ValueError(
                    '`is_back_poisson` must be provided when no "POISSERR" in '
                    'background header'
                )
            back_poisson = is_back_poisson

        if self._back_typeii:
            back_data = back_data[self._back_num : self._back_num + 1]

            back_exposure = back_data['EXPOSURE'][0]

            if 'COUNTS' in back_data.names:
                back_counts = back_data['COUNTS'][0]
                if back_poisson:
                    back_error = np.sqrt(back_counts)
                else:
                    back_error = back_data['STAT_ERR'][0]
            elif 'RATE' in back_data.names:
                back_counts = back_data['RATE'][0] * back_exposure
                if back_poisson:
                    back_error = np.sqrt(back_counts)
                else:
                    back_error = back_data['STAT_ERR'][0] * back_exposure
            else:
                raise ValueError(
                    f'Background ({backfile}) is not stored in COUNTS or RATE'
                )

            if 'QUALITY' in back_data.names:
                back_quality = back_data['QUALITY'][0]
            else:
                back_quality = np.zeros(len(back_counts))

        else:
            back_exposure = back_header['EXPOSURE']

            if 'COUNTS' in back_data.names:
                back_counts = back_data['COUNTS']
                if back_poisson:
                    back_error = np.sqrt(back_counts)
                else:
                    back_error = back_data['STAT_ERR']
            elif 'RATE' in back_data.names:
                back_counts = back_data['RATE'] * back_exposure
                if back_poisson:
                    back_error = np.sqrt(back_counts)
                else:
                    back_error = back_data['STAT_ERR'] * back_exposure
            else:
                raise ValueError(
                    f'Background ({backfile}) is not stored in COUNTS or RATE'
                )

            if 'QUALITY' in back_data.names:
                back_quality = back_data['QUALITY']
            else:
                back_quality = np.zeros(len(back_counts))


        if back_poisson:  # check if counts are integers
            diff = np.abs(back_counts - np.round(back_counts))
            if np.any(diff > 1e-8 * back_counts):
                warnings.warn(
                    f'background ({backfile}) counts are not integers, '
                    'which could lead to wrong result',
                    stacklevel=2
                )

        self.has_back = True
        self.back_exposure = back_exposure
        self.back_poisson = back_poisson
        self._back_header = back_header
        self._back_data = back_data
        self._back_counts = np.array(back_counts, dtype=np.float64)
        self._back_error = np.array(back_error, dtype=np.float64)
        self._back_quality = np.array(back_quality, dtype=np.int8)


    def _extract_resp(self, respfile, ancrfile):
        if self._spec_typeii:
            respfile = respfile or self._spec_data['RESPFILE'][0]
            if ancrfile:
                ancrfile = ancrfile
            else:
                if 'ANCRFILE' in self._spec_data.names:
                    ancrfile = self._spec_data['ANCRFILE'][0]
                else:
                    ancrfile = self._spec_header['ANCRFILE']
        else:
            respfile = respfile or self._spec_header['RESPFILE']
            ancrfile = ancrfile or self._spec_header['ANCRFILE']

        if respfile.lower() in ['none', '']:
            raise FileNotFoundError('response file is required')
        elif not os.path.exists(respfile):
            raise FileNotFoundError(f'response file "{respfile}" not found')
        else:
            with fits.open(respfile) as resp_hdul:
                ebounds = resp_hdul['EBOUNDS'].data
                if 'MATRIX' in [hdu.name for hdu in resp_hdul]:
                    resp = resp_hdul['MATRIX'].data
                else:
                    resp = resp_hdul['SPECRESP MATRIX'].data

                if len(ebounds) != len(self._spec_counts):
                    raise ValueError('response is not match with spectrum')

                self._channel = ebounds['CHANNEL']

        # a simple wrap around for zero elements for some response files
        mask = [np.any(i['MATRIX'] > 0.0) for i in resp]
        resp_ = resp[mask]

        # assumes ph_ebins is continuous
        ph_ebins = np.append(resp_['ENERG_LO'], resp_['ENERG_HI'][-1])
        ch_ebins = np.column_stack((ebounds['E_MIN'], ebounds['E_MAX']))

        # extract response matrix
        resp_matrix = resp_['MATRIX']
        if resp_matrix.dtype == np.dtype('O'):
            resp_matrix = np.array(resp_matrix.tolist(), dtype=np.float64)

        if ancrfile.lower() in ['none', '']:
            pass
        elif not os.path.exists(ancrfile):
            raise FileNotFoundError(f'arf file "{ancrfile}" not found')
        else:
            with fits.open(ancrfile) as arf_hdul:
                arf_data = arf_hdul['SPECRESP'].data['SPECRESP']

            if len(arf_data) != len(resp.data):
                raise ValueError(
                    f'arf ({ancrfile}) is not matched with rmf ({respfile})'
                )

            resp_matrix = np.expand_dims(arf_data[mask], axis=1) * resp_matrix

        ichannel = [f'{self.name}_In{c}' for c in range(len(resp_matrix))]
        self.ichannel = ichannel
        self.ph_ebins = ph_ebins
        self._ch_ebins = ch_ebins
        self._resp_matrix = resp_matrix


    def _set_name(self, name):
        excluded = ['', 'none', 'unknown']
        if name:
            self.name = name
        elif (_:=self._spec_header['DETNAM']) and _.lower() not in excluded:
            self.name = self._spec_header['DETNAM']
        elif (_:=self._spec_header['INSTRUME']) and _.lower() not in excluded:
            self.name = self._spec_header['INSTRUME']
        elif (_:=self._spec_header['TELESCOP']) and _.lower() not in excluded:
            self.name = self._spec_header['TELESCOP']
        else:
            raise ValueError('input for `name` is required')


    def _filter_channel(
        self, erange, ignore_bad, keep_channel_info, group_type, group_scale
    ):
        if ignore_bad:
            bad_quality = [1, 5]
        else:
            bad_quality = [1]

        good_quality = ~np.isin(self._spec_quality, bad_quality)
        if self.has_back:
            good_quality &= ~np.isin(self._back_quality, bad_quality)
        factor = np.where(good_quality, 1.0, 0.0)

        if group_type != None:
            if group_type not in ['bmin', 'nmin']:
                raise ValueError(
                    'current supported grouping method is "bmin"'
                )
            else:
                if group_type == 'bmin' and \
                        not (self.has_back and self.back_poisson):
                    raise ValueError(
                        'Poisson background data is required for "bmin" '
                        'grouping method'
                    )

            # initialize grouping
            grouping = np.zeros(len(self._channel), np.int64)

            # calculate channel mask according to erange information
            ch_emin = self._ch_ebins[:, 0]
            ch_emax = self._ch_ebins[:, 1]
            erange = np.atleast_2d(erange)
            emin = np.expand_dims(erange[:, 0], axis=1)
            emax = np.expand_dims(erange[:, 1], axis=1)
            chmask = (emin <= ch_emin) & (ch_emax <= emax)
            noticed = np.any(chmask, axis=0)
            # set grouping of ignored channel to 1
            grouping[~noticed] = 1

            # now group the noticed channel
            if group_type == 'bmin':
                filtered_back = factor * self._back_counts
                flag = False
                for mask in chmask:
                    back_i = filtered_back[mask]
                    grp_flag = _counts_grouping_flag(back_i, group_scale)
                    grouping[mask] = grp_flag

                    idx = np.flatnonzero(grp_flag)
                    if np.any(np.add.reduceat(back_i, idx) < group_scale):
                        flag = True

                if flag:
                    warnings.warn(
                        f'some grouped {self.name} channel has background '
                        f'counts less than {group_scale}'
                    )

            # transform grouping flag to index
            self._grouping = np.flatnonzero(grouping)

        any_good_in_group = np.add.reduceat(good_quality, self._grouping) != 0

        resp_matrix = np.add.reduceat(
            self._resp_matrix * factor,
            self._grouping,
            axis=1
        )
        resp_matrix = resp_matrix[:, any_good_in_group]

        if len(self._grouping) == len(self._spec_counts):  # case of no group
            any_good_in_group = good_quality
            good_channel = self._channel[good_quality]
            groups_channel = np.expand_dims(good_channel, axis=1).astype(str)
            emin, emax = self._ch_ebins.T
            groups_emin = emin[good_quality]
            groups_emax = emax[good_quality]
        else:
            # NOTE: if there are some bad channels within a group, this will
            # cause some inconsistency of a spectral plot, i.e., the error bar
            # of energy of a channel group will cover the energy band of bad
            # channels within the group, while these bad channels are never
            # used in fitting.
            any_good_in_group = np.add.reduceat(good_quality, self._grouping) != 0
            groups_edge_indices = np.append(self._grouping, len(self._spec_counts))
            channel = self._channel
            emin, emax = self._ch_ebins.T
            groups_channel = []
            groups_emin = []
            groups_emax = []

            for i in range(len(self._grouping)):
                if not any_good_in_group[i]:
                    continue
                slice_i = slice(groups_edge_indices[i], groups_edge_indices[i + 1])
                quality_slice = good_quality[slice_i]
                channel_slice = channel[slice_i]
                groups_channel.append(channel_slice[quality_slice].astype(str))
                groups_emin.append(min(emin[slice_i]))
                groups_emax.append(max(emax[slice_i]))

        if keep_channel_info:
            groups_channel = np.array([
                self.name+'_Ch' + '+'.join(c)
                for c in groups_channel
            ])
        else:
            groups_channel = np.array([
                self.name+f'_Ch{c}'
                for c in np.flatnonzero(any_good_in_group)
            ])

        groups_ch_ebins = np.column_stack((groups_emin, groups_emax))
        groups_emin = np.array(groups_emin)
        groups_emax = np.array(groups_emax)

        erange = np.atleast_2d(erange)
        emin = np.expand_dims(erange[:, 0], axis=1)
        emax = np.expand_dims(erange[:, 1], axis=1)
        chmask = (emin <= groups_emin) & (groups_emax <= emax)
        chmask = np.any(chmask, axis=0)

        self.channel = groups_channel[chmask]
        self.ch_emin = groups_ch_ebins[chmask, 0]
        self.ch_emax = groups_ch_ebins[chmask, 1]
        self.ch_emid = (self.ch_emin + self.ch_emax) / 2.0
        self.ch_emid_geom = np.sqrt(self.ch_emin * self.ch_emax)
        self.ch_width = self.ch_emax - self.ch_emin
        self.ch_error = np.array([
            self.ch_emin - self.ch_emid_geom,
            self.ch_emax - self.ch_emid_geom
        ])
        self.resp_matrix = resp_matrix[:, chmask]

        spec_counts = np.where(good_quality, self._spec_counts, 0)
        spec_counts = np.add.reduceat(spec_counts, self._grouping)
        spec_counts = spec_counts[any_good_in_group]

        spec_error = np.where(good_quality, self._spec_error, 0)
        spec_error = np.sqrt(
            np.add.reduceat(spec_error*spec_error, self._grouping)
        )
        spec_error = spec_error[any_good_in_group]

        self.spec_counts = spec_counts[chmask]
        self.spec_error = spec_error[chmask]

        self.data = xr.Dataset(
            data_vars={
                'name': self.name,
                'spec_counts': ('channel', self.spec_counts),
                'spec_error': ('channel', self.spec_error),
                'spec_poisson': self.spec_poisson,
                'spec_exposure': self.spec_exposure,
                'ph_ebins': self.ph_ebins,
                'ch_emin': ('channel', self.ch_emin),
                'ch_emax': ('channel', self.ch_emax),
                'ch_emid': ('channel', self.ch_emid),
                'ch_emid_geom': ('channel', self.ch_emid_geom),
                'ch_width': ('channel', self.ch_width),
                'ch_error': (('edge', 'channel'), self.ch_error),
                'resp_matrix': (['channel_in', 'channel'], self.resp_matrix),
            },
            coords={
                'channel_in': self.ichannel,
                'channel': self.channel,
                'edge': ['left', 'right']
            }
        )

        if self.has_back:
            back_counts = np.where(good_quality, self._back_counts, 0)
            back_counts = np.add.reduceat(back_counts, self._grouping)
            back_counts = back_counts[any_good_in_group]

            back_error = np.where(good_quality, self._back_error, 0)
            back_error = np.sqrt(
                np.add.reduceat(back_error * back_error, self._grouping)
            )
            back_error = back_error[any_good_in_group]

            self.back_counts = back_counts[chmask]
            self.back_error = back_error[chmask]

            self.data['back_counts'] = ('channel', self.back_counts)
            self.data['back_error'] = ('channel', self.back_error)
            self.data['back_poisson'] = self.back_poisson
            self.data['back_exposure'] = self.back_exposure

        delta = self.ch_emax - self.ch_emin
        self.net_spec = self.spec_counts / self.spec_exposure / delta
        self.net_error = self.spec_error / self.spec_exposure / delta
        error_gehrels = _gehrels_error1(self.spec_counts)
        self.net_error_gehrels = error_gehrels / self.spec_exposure / delta
        self.net_counts = self.spec_counts

        if self.has_back:
            back_rate = self.back_counts / self.back_exposure
            self.net_spec -= back_rate / delta

            err1 = self.net_error
            err2 = self.back_error / self.back_exposure / delta
            self.net_error = np.sqrt(err1*err1 + err2*err2)

            err1 = self.net_error_gehrels
            err2 = _gehrels_error1(self.back_counts)
            err2 /= self.back_exposure * delta
            self.net_error_gehrels = np.sqrt(err1*err1 + err2*err2)

            self.net_counts = self.net_counts - back_rate * self.spec_exposure

        self.data['net_counts'] = ('channel', self.net_counts)
        self.data['net_spec'] = ('channel', self.net_spec)
        self.data['net_error'] = ('channel', self.net_error)
        self.data['net_error_gehrels'] = ('channel', self.net_error_gehrels)


def _counts_grouping_idx(counts, group_scale):
    n = len(counts)
    n_minus_1 = n - 1
    grouping = np.empty(n, np.int64)
    grouping[0] = 0

    group_counts = 0
    ngroup = 1

    for i, ci in enumerate(counts):
        group_counts += ci

        if i == n_minus_1:
            if group_counts < group_scale:
                # if the last grousp does not have enough counts,
                # then combine the last two groups to ensure all
                # groups meet the scale requirement
                ngroup -= 1

            break

        if group_counts >= group_scale:
            grouping[ngroup] = i + 1

            group_counts = 0
            ngroup += 1

    return grouping[:ngroup]


def _counts_grouping_flag(counts, group_scale):
    idx = _counts_grouping_idx(counts, group_scale)
    flag = np.zeros(len(counts), dtype=np.int64)
    flag[idx] = 1

    return flag


def __gehrels_lower(n):
    if n == 0.0:
        return 0.0
    tmp = (1.0 - 1.0/(9.0*n) - 1.0/(3.0*np.sqrt(n)))
    return n*tmp*tmp*tmp
_gehrels_lower = np.vectorize(__gehrels_lower, otypes=[np.float64])


def _gehrels_upper(n):
    np1 = np.asarray(n) + 1.0
    tmp = 1.0 - 1.0/(9.0*np1) + 1.0/(3.0*np.sqrt(np1))
    return np1*tmp*tmp*tmp


def _gehrels_error1(n):
    return _gehrels_upper(n) - n


def _gehrels_error2(n):
    return n - _gehrels_lower(n)

def _gehrels_error3(n):
    return (_gehrels_upper(n) - _gehrels_lower(n)) / 2.0
