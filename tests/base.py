if __name__ == '__main__':
    from elisa.model.base import UniformParameter, generate_parameter
    from elisa.model.add import BlackBody, Powerlaw
    from numpyro.distributions import Normal
    a = UniformParameter('a', 'a', 1.0, 0, 2)
    b = UniformParameter('b', 'b', 1.0, 0, 2)
    c = UniformParameter('c', 'c', 1.0, 0, 2)
    d = a+b
    e = c*d
    f = generate_parameter('f', 'f', 2.0, Normal())
    g = e*f

    m1 = BlackBody(fmt='BB', method='simpson')
    m2 = BlackBody(fmt='BB', method='simpson')
    m2.kT = m1.kT * UniformParameter('f', 'f', 0.5, 0.001, 1, log=True)
    m3 = m1 + m2
    # m1.kT
    # m2['PhoIndex']
    # m3.blackbody['kT']
    # m3['powerlaw'].PhoIndex
    # m1(); m1(1, 2); m1(kT=1.0, norm=2.0)

    # model = Model()
    # model() -> model._eval() ->
    # model._generate_params_dict() -> params -> wrapped func

    # super_model = SuperModel()
    # super_model() ->

