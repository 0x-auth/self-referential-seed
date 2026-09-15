"""
The universe: one Mobius map, f(x) = 1 + 1/x.
Two fixed points exist (solve x = (x+1)/x -> x^2 - x - 1 = 0):
    phi     = (1+sqrt(5))/2   ~ 1.618  (attracting)
    psi     = (1-sqrt(5))/2   ~ -0.618 (repelling)
"""
import math

PHI = (1 + math.sqrt(5)) / 2
PSI = (1 - math.sqrt(5)) / 2

def f(x):   return 1 + 1/x          # forward time step ("becoming")
def g(x):   return 1/(x - 1)        # f's exact inverse -- backward time step
def C(x):   return 1 - x            # charge conjugation: the involution swapping phi <-> psi
                                     # since phi + psi = 1  ->  C(x) = phi+psi-x = 1-x
