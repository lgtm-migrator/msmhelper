# -*- coding: utf-8 -*-
"""
Create Markov State Model.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

Authors: Daniel Nagel
         Georg Diez

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numba
import numpy as np
from pyemma import msm as emsm

from msmhelper import tools


# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def build_MSM(trajs, lag_time, reversible=False, **kwargs):
    """Wrapps pyemma.msm.estimate_markov_model.

    Based on the choice of reversibility it either calls pyemma for a
    reversible matrix or it creates a transition count matrix.

    Parameters
    ----------
    trajs : list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.

    lag_time : int
        Lag time for estimating the markov model given in [frames].

    reversible : bool, optional
        If `True` it will uses pyemma.msm.estimate_markov_model which does not
        guarantee that the matrix is of full dimension. In case of `False` or
        if not statedm the local function based on a simple transitition count
        matrix will be used instead.

    kwargs
        For passing values to `pyemma.msm.estimate_markov_model`.

    Returns
    -------
    transmat : ndarray
        Transition rate matrix.

    """
    if reversible:
        MSM = emsm.estimate_markov_model(trajs, lag_time, **kwargs)
        transmat = MSM.transition_matrix
    else:
        transmat = estimate_markov_model(trajs, lag_time)

    return transmat


def estimate_markov_model(trajs, lag_time):
    """Estimates Markov State Model.

    This method estimates the MSM based on the transition count matrix.

    Parameters
    ----------
    trajs : list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.

    lag_time : int
        Lag time for estimating the markov model given in [frames].

    Returns
    -------
    T : ndarray
        Transition rate matrix.

    """
    # format input
    trajs = tools.format_state_traj(trajs)

    # get number of states
    nstates = np.unique(np.concatenate(trajs)).shape[0]

    # convert trajs to numba list
    if not numba.config.DISABLE_JIT:
        trajs = numba.typed.List(trajs)

    T_count = _generate_transition_count_matrix(trajs, lag_time, nstates)
    return _row_normalize_matrix(T_count)


@numba.njit
def _generate_transition_count_matrix(trajs, lag_time, nstates):
    """Generate a simple transition count matrix from multiple trajectories."""
    # initialize matrix
    T_count = np.zeros((nstates, nstates), dtype=np.int64)

    for traj in trajs:
        for idx in range(len(traj) - lag_time):  # due to sliding window
            T_count[traj[idx], traj[idx + lag_time]] += 1

    return T_count


@numba.njit
def _row_normalize_matrix(matrix):
    """Row normalize the given 2d matrix."""
    row_sum = np.sum(matrix, axis=1)
    if not row_sum.all():
        raise ValueError('Row sum of 0 can not be normalized.')
    # due to missing np.newaxis row_sum[:, np.newaxis] becomes
    return matrix / row_sum.reshape(matrix.shape[0], 1)


def left_eigenvectors(matrix):
    """Estimate left eigenvectors.

    Estimates the left eigenvectors and corresponding eigenvalues of a
    quadratic matrix.

    Parameters
    ----------
    matrix : ndarray
        Quadratic 2d matrix eigenvectors and eigenvalues or determined of.

    Returns
    -------
    eigenvalues : ndarray
        N eigenvalues sorted by their value (descending).

    eigenvectors : ndarray
        N eigenvectors sorted by descending eigenvalues.

    """
    matrix = np.asarray(matrix)
    tools._check_quadratic(matrix)

    # Transpose matrix and therefore determine eigenvalues and left
    # eigenvectors
    matrix_T = np.matrix.transpose(matrix)
    eigenvalues, eigenvectors = np.linalg.eig(matrix_T)

    # Transpose eigenvectors, since v[:,i] is eigenvector
    eigenvectors = eigenvectors.T

    # Sort them by descending eigenvalues
    idx_eigenvalues = eigenvalues.argsort()[::-1]
    eigenvalues_sorted = eigenvalues[idx_eigenvalues]
    eigenvectors_sorted = eigenvectors[idx_eigenvalues]

    return eigenvalues_sorted, eigenvectors_sorted


def implied_timescales(matrix, lagtime):
    """
    Calculate implied timescales.

    .. todo::
        - Clearify usage. Better passing trajs to calculate matrix?
        - Check if lagtime is valid parameter.
        - Filter 0th EV.

    Parameters
    ----------
    matrix : n x n matrix
        Transition matrix

    lagtime: int
        lagtime specified in the desired unit


    Returns
    -------
    timescales: ndarray
        N implied timescales in [unit]. The first entry corresponds to the
        stationary distribution.

    """
    # try to cast to quadratic matrix
    matrix = tools._asquadratic(matrix)

    eigenvalues, eigenvectors = left_eigenvectors(matrix)
    timescales = - (lagtime / np.log(eigenvalues))

    return timescales
