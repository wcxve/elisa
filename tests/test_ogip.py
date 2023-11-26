import numpy as np
from astropy.io import fits
from elisa.data.ogip import Data

def test_data():
    data = Data([0,np.inf],
                '/Users/xuewc/ObsData/GRB221009A/01126853000/spec/BAT_3373-3500s.pha',
                respfile='/Users/xuewc/ObsData/GRB221009A/01126853000/spec/BAT_3373-3500s.rsp')