"""
Created on Oct 11, 2017

@author: Siyuan Huang

Markov Chain Monte Carlo

"""
import random
import numpy as np


# Simulated Annealing of Metropolis-Hasting Algorithm
def metropolis_hasting(Ex, Ey, Q_xy=1, Q_yx=1, T=1):
    proposal = (Q_yx / Q_xy) * np.exp(-(Ey - Ex) / T)
    # print proposal, Ey, Ex
    accept_prob = np.min(np.array([1, proposal]))
    r = random.random()
    # print 'accept prob is :{}, random number is :{}'.format(accept_prob, r)
    if r < accept_prob:
        return True
    else:
        return False


def main():
    print metropolis_hasting(0.6, 0.1, T=1)


if __name__ == '__main__':
    main()
