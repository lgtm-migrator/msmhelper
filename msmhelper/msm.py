# -*- coding: utf-8 -*-
"""Create Markov State Model.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

Authors: Daniel Nagel
         Georg Diez

"""
import numba
import numpy as np
from pyemma import msm as emsm

from msmhelper import linalg, tests
from msmhelper.statetraj import StateTraj, LumpedStateTraj
from msmhelper.decorators import shortcut


def build_MSM(trajs, lagtime, reversible=False, **kwargs):
    """Wrapps pyemma.msm.estimate_markov_model.

    Based on the choice of reversibility it either calls pyemma for a
    reversible matrix or it creates a transition count matrix.

    Parameters
    ----------
    trajs : list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.

    lagtime : int
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
        if isinstance(trajs, StateTraj):
            trajs = trajs.state_trajs
        MSM = emsm.estimate_markov_model(trajs, lagtime, **kwargs)
        transmat, states = MSM.transition_matrix, MSM.active_set
    else:
        transmat, states = estimate_markov_model(trajs, lagtime)

    return transmat, states


def estimate_markov_model(trajs, lagtime):
    """Estimates Markov State Model.

    This method estimates the MSM based on the transition count matrix.

    Parameters
    ----------
    trajs : statetraj or list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.

    lagtime : int
        Lag time for estimating the markov model given in [frames].

    Returns
    -------
    T : ndarray
        Transition rate matrix.

    permutation : ndarray
        Array with corresponding states.

    """
    trajs = StateTraj(trajs)

    if isinstance(trajs, LumpedStateTraj):
        return _estimate_markov_model(
            trajs.trajs,
            lagtime,
            trajs.nmicrostates,
            trajs.microstates,
        )
    return _estimate_markov_model(
        trajs.trajs,
        lagtime,
        trajs.nstates,
        trajs.states,
    )


def _estimate_markov_model(trajs, lagtime, nstates, perm=None):
    """Estimate MSM based on the transition count matrix."""
    # convert trajs to numba list # noqa: SC100
    if not numba.config.DISABLE_JIT:
        trajs = numba.typed.List(trajs)

    if perm is None:
        perm = np.arange(nstates)

    Tcount = _generate_transition_count_matrix(trajs, lagtime, nstates)
    return _row_normalize_matrix(Tcount), perm


@numba.njit
def _generate_transition_count_matrix(trajs, lagtime, nstates):
    """Generate a simple transition count matrix from multiple trajectories."""
    # initialize matrix
    T_count = np.zeros((nstates, nstates), dtype=np.int64)

    for traj in trajs:
        for stateFrom, stateTo in zip(traj[:-lagtime], traj[lagtime:]):
            T_count[stateFrom, stateTo] += 1

    return T_count


@numba.njit
def _row_normalize_matrix(matrix):
    """Row normalize the given 2d matrix."""
    row_sum = np.sum(matrix, axis=1)
    if not row_sum.all():
        row_sum[row_sum == 0] = 1

    # due to missing np.newaxis row_sum[:, np.newaxis] becomes # noqa: SC100
    return matrix / row_sum.reshape(matrix.shape[0], 1)


def _implied_timescales(tmat, lagtime):
    """
    Calculate implied timescales.

    .. todo::
        - Clearify usage. Better passing trajs to calculate matrix?

    Parameters
    ----------
    tmat : ndarray
        Quadratic transition matrix.

    lagtime: int
        Lagtime for estimating the markov model given in [frames].

    Returns
    -------
    timescales: ndarray
        Implied timescale given in frames.

    """
    tmat = np.asarray(tmat)

    eigenvalues = linalg.left_eigenvalues(tmat)
    # for negative eigenvalues no timescale is defined
    eigenvalues[eigenvalues < 0] = np.nan
    return - lagtime / np.log(eigenvalues[1:])


def implied_timescales(trajs, lagtimes, reversible=False):
    """Calculate the implied timescales.

    Calculate the implied timescales for the given values.
    .. todo:: catch if for higher lagtimes the dimensionality changes

    Parameters
    ----------
    trajs : StateTraj or list or ndarray or list of ndarray
        State trajectory/trajectories. The states should start from zero and
        need to be integers.

    lagtimes : list or ndarray int
        Lagtimes for estimating the markov model given in [frames].

    reversible : bool
        If reversibility should be enforced for the markov state model.

    Returns
    -------
    T : ndarray
        Transition rate matrix.

    """
    # format input
    trajs = StateTraj(trajs)
    lagtimes = np.atleast_1d(lagtimes)

    # check that lag times are array of integers
    if not np.issubdtype(lagtimes.dtype, np.integer):
        raise TypeError(
            'Lagtimes needs to be integers but are {0}'.format(lagtimes.dtype),
        )
    if not (lagtimes > 0).all():
        raise TypeError('Lagtimes needs to be positive integers')

    # initialize result
    impl_timescales = np.zeros((len(lagtimes), trajs.nstates - 1))

    for idx, lagtime in enumerate(lagtimes):
        transmat, _ = build_MSM(trajs, lagtime, reversible=reversible)
        impl_timescales[idx] = _implied_timescales(transmat, lagtime)

    return impl_timescales


@shortcut('peq')
def equilibrium_population(tmat):
    """Calculate equilibirum population.

    Parameters
    ----------
    tmat : ndarray
        Quadratic transition matrix, needs to be ergodic.

    Returns
    -------
    peq : ndarray
        Equilibrium population of input matrix.

    """
    tmat = np.asarray(tmat)
    if not tests.is_ergodic(tmat):
        raise TypeError('tmat needs to be ergodic transition matrix.')

    _, eigenvectors = linalg.left_eigenvectors(tmat)
    return eigenvectors[0] / np.sum(eigenvectors[0])
