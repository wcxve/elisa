from elisa.inference.fit import LikelihoodFit, BayesianFit
from elisa.model.add import Powerlaw
from elisa.data.ogip import Data


m = Powerlaw()

LikelihoodFit([Data([1, 20], '')], m, ['chi2'])
