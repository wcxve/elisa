import numpy as np
import pymc as pm
import pytensor.tensor as pt

__all__ = ['chi', 'cstat', 'pstat', 'pgstat', 'wstat']


def normal_logp(value, mu, sigma):
    resd = (value - mu) / sigma
    return -0.5 * resd*resd


def normal_random(mu, sigma, rng=None, size=None):
    return rng.normal(loc=mu, scale=sigma, size=size)


def poisson_logp(value, mu):
    gof = pt.xlogx.xlogx(value) - value
    # logp = pt.xlogx.xlogy0(value, mu) - mu
    # logp = value*pt.log(mu) - mu
    logp = pt.switch(pt.eq(value, 0.0), -mu, value * pt.log(mu) - mu)
    return logp - gof


def poisson_random(mu, rng=None, size=None):
    return rng.poisson(lam=mu, size=size)


def pgstat_background(s, n, b_est, sigma, a):
    sigma2 = sigma*sigma
    e = b_est - a*sigma2
    f = a*sigma2*n + e*s
    c = a*e - s
    d = pt.sqrt(c*c + 4.0*a*f)
    b = pt.switch(
        pt.or_(pt.ge(e, 0.0), pt.ge(f, 0.0)),
        pt.switch(
            pt.gt(n, 0.0),
            (c + d)/(2*a),
            e
        ),
        0.0
    )
    # b = pt.switch(
    #     pt.gt(n, 0.0),
    #     (c + d) / (2 * a),
    #     e
    # )
    return b


def wstat_background(s, n_on, n_off, a):
    c = a*(n_on + n_off) - (a + 1)*s
    d = pt.sqrt(c*c + 4*a*(a + 1)*n_off*s)
    b = pt.switch(
        pt.eq(n_on, 0),
        n_off/(1 + a),
        pt.switch(
            pt.eq(n_off, 0),
            pt.switch(
                pt.le(s, a/(a + 1)*n_on),
                n_on/(1 + a) - s/a,
                0.0
            ),
            (c + d) / (2*a*(a + 1))
        )
    )
    return b


def chi(data, model, context):
    name = data.name
    chdim = f'{name}_channel'

    context.add_coord(chdim, data.channel)
    with context:
        counts = data.spec_counts
        error = data.spec_error
        if data.has_back:
            back_counts = data.back_counts
            back_error = data.back_error
            ratio = data.spec_exposure / data.back_exposure
            counts = counts - ratio * back_counts
            error = np.sqrt(error*error + ratio*ratio * back_error*back_error)

        counts = pm.MutableData(f'{name}_spec_counts', counts)
        error = pm.ConstantData(f'{name}_spec_error', error)
        spec_exposure = pm.ConstantData(f'{name}_spec_exposure',
                                        data.spec_exposure)
        ph_ebins = pm.ConstantData(f'{name}_ph_ebins', data.ph_ebins)
        resp_matrix = pm.ConstantData(f'{name}_resp_matrix', data.resp_matrix)

        NEdE = model(ph_ebins)
        CEdE = pt.dot(NEdE, resp_matrix)

        s = CEdE * spec_exposure
        mu_on = pm.Deterministic(f'{name}_TOTAL', s, dims=chdim)

        pm.CustomDist(
            f'{name}_Non', mu_on, error,
            logp=normal_logp, random=normal_random,
            observed=counts, dims=chdim
        )

def cstat(data, model, context):
    name = data.name
    chdim = f'{name}_channel'

    if not data.spec_poisson:
        raise ValueError(
            f'Poisson data is required for using C-statistics, but {name} data'
            'is Gaussian distributed'
        )

    if data.has_back:
        back_type = 'Poisson' if data.back_poisson else 'Gaussian'
        stat_type = 'W' if data.back_poisson else 'PG'
        stat = 'w' if data.back_poisson else 'pg'
        raise ValueError(
            f'C-statistics is not valid for Poisson data with {back_type} '
            f'background, use {stat_type}-statistics ({stat}stat) for '
            f'{name} instead'
        )

    context.add_coord(chdim, data.channel)
    with context:
        spec_counts = pm.MutableData(f'{name}_spec_counts', data.spec_counts)
        spec_exposure = pm.ConstantData(f'{name}_spec_exposure',
                                        data.spec_exposure)
        ph_ebins = pm.ConstantData(f'{name}_ph_ebins', data.ph_ebins)
        resp_matrix = pm.ConstantData(f'{name}_resp_matrix', data.resp_matrix)

        NEdE = model(ph_ebins)
        CEdE = pt.dot(NEdE, resp_matrix)

        s = CEdE * spec_exposure

        mu_on = pm.Deterministic(f'{name}_TOTAL', s, dims=chdim)

        pm.CustomDist(
            f'{name}_Non', mu_on,
            logp=poisson_logp, random=poisson_random,
            observed=spec_counts, dims=chdim
        )

def pstat(data, model, context):
    if not data.spec_poisson:
        raise ValueError(
            'Poisson data is required for using P-statistics'
        )
    if not data.has_back:
        raise ValueError(
            'Background is required for using P-statistics'
        )

    name = data.name
    chdim = f'{name}_channel'

    context.add_coord(chdim, data.channel)
    with context:
        spec_counts = pm.MutableData(f'{name}_spec_counts', data.spec_counts)
        back_counts = pm.ConstantData(f'{name}_back_counts', data.back_counts)
        spec_exposure = pm.ConstantData(f'{name}_spec_exposure',
                                        data.spec_exposure)
        back_exposure = pm.ConstantData(f'{name}_back_exposure',
                                        data.back_exposure)
        ph_ebins = pm.ConstantData(f'{name}_ph_ebins', data.ph_ebins)
        resp_matrix = pm.ConstantData(f'{name}_resp_matrix', data.resp_matrix)

        NEdE = model(ph_ebins)
        CEdE = pt.dot(NEdE, resp_matrix)

        s = CEdE * spec_exposure
        a = spec_exposure / back_exposure
        b = back_counts

        mu_on = pm.Deterministic(f'{name}_TOTAL', s + a*b, dims=chdim)
        # pm.Deterministic(f'{name}_BKG', b, dims=chdim)

        pm.CustomDist(
            f'{name}_Non', mu_on,
            logp=poisson_logp, random=poisson_random,
            observed=spec_counts, dims=chdim
        )

def pgstat(data, model, context):
    name = data.name
    chdim = f'{name}_channel'

    if not data.spec_poisson:
        raise ValueError(
            'Poisson data is required for using PG-statistics'
        )
    if not data.has_back:
        raise ValueError(
            'Background is required for using PG-statistics'
        )

    context.add_coord(chdim, data.channel)
    with context:
        spec_counts = pm.MutableData(f'{name}_spec_counts', data.spec_counts)
        back_counts = pm.MutableData(f'{name}_back_counts', data.back_counts)
        back_error = pm.ConstantData(f'{name}_back_error', data.back_error)
        spec_exposure = pm.ConstantData(f'{name}_spec_exposure',
                                       data.spec_exposure)
        back_exposure = pm.ConstantData(f'{name}_back_exposure',
                                       data.back_exposure)
        ph_ebins = pm.ConstantData(f'{name}_ph_ebins', data.ph_ebins)
        resp_matrix = pm.ConstantData(f'{name}_resp_matrix', data.resp_matrix)

        NEdE = model(ph_ebins)
        CEdE = pt.dot(NEdE, resp_matrix)

        s = CEdE * spec_exposure
        a = spec_exposure / back_exposure
        b = pgstat_background(s, spec_counts, back_counts, back_error, a)

        mu_on = pm.Deterministic(f'{name}_TOTAL', s + a * b, dims=chdim)
        mu_off = pm.Deterministic(f'{name}_BKG', b, dims=chdim)

        pm.CustomDist(
            f'{name}_Non', mu_on,
            logp=poisson_logp, random=poisson_random,
            observed=spec_counts, dims=chdim
        )

        pm.CustomDist(
            f'{name}_Noff', mu_off, back_error,
            logp=normal_logp, random=normal_random,
            observed=back_counts, dims=chdim
        )

def wstat(data, model, context):
    name = data.name
    chdim = f'{name}_channel'

    if not data.spec_poisson:
        raise ValueError(
            'Poisson data is required for using W-statistics'
        )
    if not (data.has_back and data.back_poisson):
        raise ValueError(
            'Poisson background is required for using W-statistics'
        )

    context.add_coord(chdim, data.channel)
    with context:
        spec_counts = pm.MutableData(f'{name}_spec_counts', data.spec_counts)
        back_counts = pm.MutableData(f'{name}_back_counts', data.back_counts)
        spec_exposure = pm.ConstantData(f'{name}_spec_exposure',
                                       data.spec_exposure)
        back_exposure = pm.ConstantData(f'{name}_back_exposure',
                                       data.back_exposure)
        ph_ebins = pm.ConstantData(f'{name}_ph_ebins', data.ph_ebins)
        resp_matrix = pm.ConstantData(f'{name}_resp_matrix', data.resp_matrix)

        NE_dEph = model(ph_ebins)
        CE_dEch = pt.dot(NE_dEph, resp_matrix)
        s = CE_dEch * spec_exposure
        a = spec_exposure / back_exposure
        b = wstat_background(s, spec_counts, back_counts, a)

        mu_on = pm.Deterministic(f'{name}_TOTAL', s + a * b, dims=chdim)
        mu_off = pm.Deterministic(f'{name}_BKG', b, dims=chdim)

        pm.CustomDist(
            f'{name}_Non', mu_on,
            logp=poisson_logp, random=poisson_random,
            observed=spec_counts, dims=chdim
        )

        pm.CustomDist(
            f'{name}_Noff', mu_off,
            logp=poisson_logp, random=poisson_random,
            observed=back_counts, dims=chdim
        )

def fpstat(data, model, context):
    name = data.name
    chdim = f'{name}_channel'

    if not data.spec_poisson:
        raise ValueError(
            'Poisson data is required for using full Poisson statistics'
        )
    if not (data.has_back and data.back_poisson):
        raise ValueError(
            'Poisson background is required for using full Poisson statistics'
        )

    context.add_coord(chdim, data.channel)
    with context:
        spec_counts = pm.MutableData(f'{name}_spec_counts', data.spec_counts)
        back_counts = pm.MutableData(f'{name}_back_counts', data.back_counts)
        spec_exposure = pm.ConstantData(f'{name}_spec_exposure',
                                       data.spec_exposure)
        back_exposure = pm.ConstantData(f'{name}_back_exposure',
                                       data.back_exposure)
        ph_ebins = pm.ConstantData(f'{name}_ph_ebins', data.ph_ebins)
        resp_matrix = pm.ConstantData(f'{name}_resp_matrix', data.resp_matrix)

        NE_dEph = model(ph_ebins)
        CE_dEch = pt.dot(NE_dEph, resp_matrix)
        s = CE_dEch * spec_exposure
        a = spec_exposure / back_exposure
        # b = wstat_background(s, spec_counts, back_counts, a)
        b = pt.exp(pm.Uniform(f'ln({name}_BKG)', -15, 15, dims=chdim))

        mu_on = pm.Deterministic(f'{name}_TOTAL', s + a * b, dims=chdim)
        mu_off = pm.Deterministic(f'{name}_BKG', b, dims=chdim)

        pm.CustomDist(
            f'{name}_Non', mu_on,
            logp=poisson_logp, random=poisson_random,
            observed=spec_counts, dims=chdim
        )

        pm.CustomDist(
            f'{name}_Noff', mu_off,
            logp=poisson_logp, random=poisson_random,
            observed=back_counts, dims=chdim
        )
