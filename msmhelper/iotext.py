"""
Input and output text files.

BSD 3-Clause License
Copyright (c) 2019-2020, Daniel Nagel
All rights reserved.

Author: Daniel Nagel

TODO:
    - create todo

"""
# ~~~ IMPORT ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import datetime
import getpass  # get username with getpass.getuser()
import os
import platform  # get pc name with platform.node()
import sys

import numpy as np
import pandas as pd

import __main__ as main
from msmhelper import tools

# ~~~ RUNTIME USER INFORMATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
try:
    SCRIPT_DIR, SCRIPT_NAME = os.path.split(main.__file__)
except AttributeError:
    SCRIPT_DIR, SCRIPT_NAME = None, 'console'

RUI = {'user': getpass.getuser(),
       'platform': platform.node(),
       'date': datetime.datetime.now(),
       'script_dir': SCRIPT_DIR,
       'script_name': SCRIPT_NAME}


# ~~~ ERROR CLASS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class FileError(Exception):
    """An exception for wrongly formated input files."""


# ~~~ FUNCTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def opentxt(file_name, comment='#', nrows=None, **kwargs):
    r"""
    Open a text file.

    This method can load an nxm array of floats from an ascii file. It uses
    either pandas read_csv for a single comment or as fallback the slower numpy
    laodtxt for multiple comments.

    .. warning:: In contrast to pandas the order of usecols will be used. So if
        using ´data = opentxt(..., uscols=[1, 0])´ you acces the first column
        by `data[:, 0]` and the second one by `data[:, 1]`.

    Parameters
    ----------
    file_name : string
        Name of file to be opened.

    comment : str or array of str, optional
        Characters with which a comment starts.

    nrows : int, optional
        The maximum number of lines to be read

    usecols : int-array, optional
        Columns to be read from the file (zero indexed).

    skiprows : int, optional
        The number of leading rows which will be skipped.

    dtype : data-type, optional
        Data-type of the resulting array. Default: float.

    Returns
    -------
    data : ndarray
        Data read from the text file.

    """
    if len(comment) == 1:
        # pandas does not support array of single char
        if not isinstance(comment, str):
            comment = comment[0]

        # force pandas to load in stated order without sorting
        cols = kwargs.pop('usecols', None)
        if cols is not None:
            idx = np.argsort(cols)
            cols = np.asarray(cols).astype(np.integer)[idx]

        data = pd.read_csv(file_name,
                           sep=r'\s+',
                           header=None,
                           comment=comment,
                           nrows=nrows,
                           usecols=cols,
                           **kwargs).values
        if data.shape[-1] == 1:
            return data.flatten()
        else:
            # swap columns back to ensure correct order
            if cols is not None:
                data = tools.swapcols(data, idx, np.arange(len(idx)))
            return data
    else:
        return np.loadtxt(file_name,
                          comments=comment,
                          max_rows=nrows,
                          **kwargs)


def savetxt(file_name, data, header=None, fmt='%.5f'):
    """
    Save nxm array of floats to a text file.

    It uses numpys savetxt method and extends the header with information
    of execution.

    Parameters
    ----------
    file_name : string
        File name to store data.

    data : ndarray
        Data to be stored.

    header : str, optional
        Comment written into the header of the output file.

    fmt : str or sequence of strs, optional
        See numpy.savetxt fmt.

    """
    # prepare header comments
    header_comment = ('This file was generated with {script_name}:\n'
                      .format(**RUI))
    for arg in sys.argv:  # loop over the given arguments
        header_comment += '{arg} '.format(arg=arg)
    header_comment += '\n\n{date}, {user}@{platform}'.format(**RUI)
    if header:  # print column title if given
        header_comment += '\n' + header

    # save file
    np.savetxt(file_name, data, fmt=fmt, header=header_comment)


def opentxt_limits(file_name, limits_file=None, **kwargs):
    """
    Load file and split according to limit file.

    Both, the limit file and the trajectory file needs to be a single column
    file. If limits_file is not provided it will return [traj]. The trajectory
    will of dtype np.int16, so the states needs to be smaller than 32767.

    Parameters
    ----------
    file_name : string
        Name of file to be opened.

    limits_file : str, optional
        File name of limit file. Should be single column ascii file.

    kwargs
        The Parameters defined in opentxt.

    Returns
    -------
    traj : ndarray
        Return array of subtrajectories.

    """
    # open trajectory
    if 'dtype' in kwargs and not np.issubdtype(kwargs['dtype'], np.integer):
        raise TypeError('dtype should be integer')
    else:
        kwargs['dtype'] = np.int16

    traj = opentxt(file_name, **kwargs)
    if len(traj.shape) != 1:
        raise FileError('Shoud be single column file.')

    # open limits
    limits = open_limits(limits_file=limits_file, data_length=len(traj))

    # split trajectory
    return np.split(traj, limits)[:-1]


def open_limits(data_length, limits_file=None):
    """
    Load and check limit file.

    The limits give the length of each single trajectory. So e.g.
    [5, 5, 5] for 3 equally-sized subtrajectories of length 5.

    Parameters
    ----------
    data_length : int
        Length of data read.

    limits_file : str, optional
        File name of limit file. Should be single column ascii file.

    """
    if limits_file is None:
        return np.array([data_length])  # for single trajectory
    else:
        # open limits file
        limits = opentxt(limits_file)
        if len(limits.shape) != 1:
            raise FileError('Shoud be single column file.')

        # convert to cumulative sum
        limits = np.cumsum(limits)
        if data_length != limits[-1]:
            raise ValueError('Limits are inconsistent to data.')

        return limits
