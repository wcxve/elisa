def test_mle_result(simulation, mle_result):
    data = simulation
    result = mle_result

    nbins = len(data.channel)
    expo = data.spec_exposure

    assert result.ndata['total'] == nbins
    assert result.dof == nbins - 1

    # Check various methods of mle result
    result.boot(1009)
    result.ci(method='boot')
    result.flux(1, 2)
    result.lumin(1, 10000, z=1)
    result.eiso(1, 10000, z=1, duration=expo)
    result.summary()
    result.plot()
    result.plot('data ne ene eene Fv vFv rq pit corner')
    _ = result.deviance
    _ = result.aic
    _ = result.bic
    assert all(i > 0.05 for i in result.gof.values())

    plotter = result.plot
    plotter.plot_qq('rd')
    plotter.plot_qq('rp')
    plotter.plot_qq('rq')

    result.save('mle.pkl.gz', 'gzip')
    result.load('mle.pkl.gz', 'gzip')
    result.save('mle.pkl.bz2', 'bz2')
    result.load('mle.pkl.bz2', 'bz2')
    result.save('mle.pkl.xz', 'lzma')
    result.load('mle.pkl.xz', 'lzma')


def test_posterior_result(simulation, posterior_result):
    data = simulation
    result = posterior_result

    nbins = len(data.channel)
    expo = data.spec_exposure

    # Check various methods of posterior result
    assert result.ndata['total'] == nbins
    assert result.dof == nbins - 2
    result.ppc(1009)
    result.flux(1, 2)
    result.lumin(1, 10000, z=1)
    result.eiso(1, 10000, z=1, duration=expo)
    result.summary()
    result.plot()
    result.plot('data ne ene eene Fv vFv rq pit corner khat trace')
    _ = result.deviance
    _ = result.loo
    _ = result.waic
    assert all(i > 0.05 for i in result.gof.values())

    plotter = result.plot
    plotter.plot_qq('rd')
    plotter.plot_qq('rp')
    plotter.plot_qq('rq')

    result.save('posterior.pkl.gz', 'gzip')
    result.load('posterior.pkl.gz', 'gzip')
    result.save('posterior.pkl.bz2', 'bz2')
    result.load('posterior.pkl.bz2', 'bz2')
    result.save('posterior.pkl.xz', 'lzma')
    result.load('posterior.pkl.xz', 'lzma')
