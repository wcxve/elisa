from elisa.inference.fit import MaxLikeFit, BayesianFit
from elisa.model.add import Powerlaw
from elisa.data.data import Data


m = Powerlaw()

MaxLikeFit([Data()], m, ['chi2'])
