#/bin/python3

import time
import pyjet
import numpy as np

from BranchClass import *

from itertools import combinations


def CalcDeltaR(j1, j2):
    eta1 = j1.eta
    phi1 = j1.phi
    eta2 = j2.eta
    phi2 = j2.phi

    dEta = eta1-eta2
    dPhi = abs(phi1-phi2)
    if dPhi > np.pi:
        dPhi = 2*np.pi - dPhi

    dR = (dPhi**2 + dEta**2)**0.5

    return dR

def CalcEECorr(jet, n=1, beta=1.0):

    assert n == 2 or n == 3, 'fn must be in [2, 3] but is n'

    jet_particles = jet.constituents()

    if len(jet_particles) < n:
        return -1

    currentSum = 0

    if n == 2:
        for p1, p2 in combinations(jet_particles, 2):
            # get the terms of the triplet at hand
            pt1 = p1.pt
            pt2 = p2.pt
            dr12 = CalcDeltaR(p1, p2)

            # calculate the partial contribution
            thisterm = pt1*pt2 * (dr12)**beta

            # sum it up
            currentSum += thisterm

        eec = currentSum/(jet.pt)**2

    elif n == 3:
        dr = {(p1, p2): CalcDeltaR(p1, p2)
              for p1, p2 in combinations(jet_particles, 2)}
        for p1, p2, p3 in combinations(jet_particles, 3):
            # get the terms of the triplet at hand
            dr12 = dr[(p1, p2)]
            dr13 = dr[(p1, p3)]
            dr23 = dr[(p2, p3)]

            # calculate the partial contribution
            thisterm = p1.pt*p2.pt*p3.pt * (dr12*dr13*dr23)**beta

            # sum it up
            currentSum += thisterm

        eec = currentSum/(jet.pt)**3
    return eec

def CalcDeltaRArray(p, a):
    dEta = p['eta'] - \
        a['eta'].repeat(p.shape[0]).reshape(a.shape[0], p.shape[0])
    dPhi = np.abs(p['phi'] - a['phi'].repeat(p.shape[0]
                                             ).reshape(a.shape[0], p.shape[0]))
    mask = dPhi > np.pi
    dPhi[mask] *= -1
    dPhi[mask] += 2 * np.pi
    return (dPhi**2 + dEta**2)**0.5

def t0(jet):
    return sum(p.pt * CalcDeltaR(p, jet) for p in jet.constituents())

def tn(jet, n): #t1 t2 t3 t21 t32
    assert n >= 0
    if n == 0:
        return t0(jet)
    particles = jet.constituents_array()
    if len(particles) < n:
        return -1
    subjets = pyjet.cluster(particles, R=1.0, p=1).exclusive_jets(n)
    subjets_array = [subjet.constituents_array() for subjet in subjets]
    wta_axes = [a[np.argmax(a['pT'])] for a in subjets_array]
    wta_axes = np.array(wta_axes, dtype=subjets_array[0].dtype)
    return np.sum(particles['pT']*CalcDeltaRArray(particles, wta_axes).min(axis=0)) / t0(jet)

