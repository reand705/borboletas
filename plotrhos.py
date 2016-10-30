# coding: utf-8
#Plota diferença nos phis médios para diferentes rhos
#get_ipython().magic(u'matplotlib inline')
import scipy.integrate
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc ## desnecessário
matplotlib.rcParams['text.usetex'] = True

from borboletas3.py import solver
