"""
Set of helpful functions.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

Author: Daniel Nagel

TODO:
    - Correct border effects of running mean

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np


# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def shift_data(data, val_old, val_new, dtype=np.uint16):
    """
    Shift data from old to new values.

    > **CAUTION:**
    > The values of `val_old`, `val_new` and `data` needs to be integers.

    The basic function is based on Ashwini_Chaudhary solution:
    https://stackoverflow.com/a/29408060

    Parameters
    ----------
    data : ndarray, list, list of ndarrays
        1D data or a list of data.

    val_old : ndarray or list
        Values in data which should be replaced. All values needs to be within
        the range of `[data.min(), data.max()]`

    val_new : ndarray or list
        Values which will be used instead of old ones.

    dtype : data-type, optional
        The desired data-type. Needs to be of type unsigned integer.

    Returns
    -------
    data : ndarray
        Shifted data in same shape as input.

    """
    # check data-type
    if not np.issubdtype(dtype, np.integer):
        raise TypeError('An unsigned integer type is needed.')

    # offset data and val_old to allow negative values
    offset = np.min(data)

    # convert to np.array
    val_old = (np.asarray(val_old) - offset).astype(dtype)
    val_new = (np.asarray(val_new) - offset).astype(dtype)

    # flatten data
    data, shape_kwargs = _flatten_data(data)

    # convert data and shift
    data = (data - offset).astype(dtype)

    # shift data
    conv = np.arange(data.max() + 1, dtype=dtype)
    conv[val_old] = val_new
    data_shifted = conv[data]

    # shift data back
    data_shifted = data_shifted.astype(np.integer) + offset

    # reshape
    data_shifted = _unflatten_data(data_shifted, shape_kwargs)
    return data_shifted


def runningmean(data, window):
    r"""
    Compute centered running average with given window size.

    This function returns the centered based running average of the given
    data. The output of this function is of the same length as the input,
    by assuming that the given data is zero before and after the given
    series. Hence, there are border affects which are not corrected.

    > **CAUTION:**
    > If the given window is even (not symmetric) it will be shifted towards
    > the beginning of the current value. So for `window=4`, it will consider
    > the current position \(i\), the two to the left \(i-2\) and \(i-1\) and
    > one to the right \(i+1\).

    Function is taken from lapis:
    https://stackoverflow.com/questions/13728392/moving-average-or-running-mean

    Parameters
    ----------
    data : ndarray
        One dimensional numpy array.

    window : int
        Integer which specifies window-width.

    Returns
    -------
    data_rmean : ndarray
        Data which is time-averaged over the specified window.

    """
    # Calculate running mean
    data_runningmean = np.convolve(data, np.ones(window) / window, mode='same')

    return data_runningmean


def _format_state_trajectory(trajs):
    """Convert state trajectory to list of ndarrays."""
    # 1d ndarray
    if isinstance(trajs, np.ndarray):
        if len(trajs.shape) == 1:
            trajs = [trajs]
    # list
    elif isinstance(trajs, list):
        # list of integers
        if all((np.issubdtype(type(traj), np.integer) for traj in trajs)):
            trajs = [np.array(trajs)]
        # list of lists
        elif all((isinstance(traj, list) for traj in trajs)):
            trajs = [np.asarray(traj) for traj in trajs]
        # not list of ndarrays
        elif not all((isinstance(traj, np.ndarray) for traj in trajs)):
            raise TypeError('Wrong data type of trajs.')

    # check for integers
    if not all((np.issubdtype(traj.dtype, np.integer) for traj in trajs)):
        raise TypeError('States needs to be integers.')

    return trajs


def _flatten_data(data):
    """
    Flatten data to 1D ndarray.

    This method flattens ndarrays, list of ndarrays to a 1D ndarray. This can
    be undone with _unflatten_data().

    Parameters
    ----------
    data : ndarray, list, list of ndarrays
        1D data or a list of data.

    Returns
    -------
    data : ndarray
        Flattened data.

    kwargs : dict
        Dictionary with information to restore shape.

    """
    kwargs = {}

    # flatten data
    if isinstance(data, list):
        # list of ndarrays
        if all((isinstance(row, np.ndarray) for row in data)):
            # get shape and flatten
            kwargs['limits'] = np.cumsum([len(row) for row in data])
            data = np.concatenate(data)
        # list of numbers
        else:
            data = np.asarray(data)
    elif isinstance(data, np.ndarray):
        # get shape and flatten
        kwargs['data_shape'] = data.shape
        data = data.flatten()

    return data, kwargs


def _unflatten_data(data, kwargs):
    """
    Unflatten data to original structure.

    This method undoes _flatten_data().

    Parameters
    ----------
    data : ndarray
        Flattened data.

    kwargs : dict
        Dictionary with information to restore shape. Provided by
        _flatten_data().

    Returns
    -------
    data : ndarray, list, list of ndarrays
        Data with restored shape.

    """
    # reshape
    if 'data_shape' in kwargs:
        data = data.reshape(kwargs['data_shape'])
    elif 'limits' in kwargs:
        data = np.split(data, kwargs['limits'])[:-1]

    return data
