URLS=[
"msmhelper/index.html",
"msmhelper/benchmark.html",
"msmhelper/statetraj.html",
"msmhelper/tools.html",
"msmhelper/msm.html",
"msmhelper/compare.html",
"msmhelper/md.html",
"msmhelper/tests.html",
"msmhelper/iotext.html",
"msmhelper/linalg.html"
];
INDEX=[
{
"ref":"msmhelper",
"url":0,
"doc":"                                        Docs \u2022  Features \u2022  Installation \u2022  Usage    msmhelper >  Warning > This package is still in beta stage. Please open an issue if you encounter > any bug/error. This is a package with helper functions to work with discrete state trajectories and Markov state models. In contrast to  pyemma and  msmbuilder it features a very limited set of functionality. This repo is prepared to be published. In the next weeks the source code will be cleaned up, tutorials will be added and this readme will be extended. This package will be published soon: > D. Nagel, and G. Stock, >  msmhelper: A Python Package for Markov State Modeling of Protein Dynamics , > in preparation We kindly ask you to cite this article in case you use this software package for published works.  Features - Simple usage with sleek function-based API - Supports latest Python 3.10 - Extensive [documentation](https: moldyn.github.io/msmhelper) with many command line scripts -  .  Installation The package is called  msmhelper and is available via [PyPI](https: pypi.org/project/msmhelper) or [conda](https: anaconda.org/conda-forge/msmhelper). To install it, simply call:   python3 -m pip install  upgrade msmhelper   or   conda install -c conda-forge msmhelper   or for the latest dev version    via ssh key python3 -m pip install git+ssh: git@github.com/moldyn/msmhelper.git  or via password-based login python3 -m pip install git+https: github.com/moldyn/msmhelper.git    Usage This package is mainly based on  numpy and  numba for all computational complex tasks.  Usage   import msmhelper as mh  .    Roadmap: - Add unit tests for all functions - Add examples usage scripts - Create typing module  Development  Additional Requirements: - wemake-python-styleguide - flake8-spellcheck  Pytest Running pytest with numba needs an additional flag   export NUMBA_DISABLE_JIT=1  pytest    Credits: - [numpy](https: docs.scipy.org/doc/numpy) - [realpython](https: realpython.com/)"
},
{
"ref":"msmhelper.benchmark",
"url":1,
"doc":"Benchmark Markov State Model. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved."
},
{
"ref":"msmhelper.benchmark.ck_test",
"url":1,
"doc":"This function is an alias of  chapman_kolmogorov_test .See its docstring for further help.",
"func":1
},
{
"ref":"msmhelper.benchmark.chapman_kolmogorov_test",
"url":1,
"doc":"Calculate the Chapman Kolmogorov equation. This method estimates the Chapman Kolmogorov equation  T(\\tau n) = T^n(\\tau)\\;. Projected onto the diagonal this is known as the Chapman Kolmogorov test. For more details see, e.g., the review Prinz et al.[^1]. [^1]: Prinz et al.  Markov models of molecular kinetics: Generation and validation ,  J. Chem. Phys. , 134, 174105 (2011), doi:[10.1063/1.3565032](https: doi.org/10.1063/1.3565032) Parameters      trajs : StateTraj or list or ndarray or list of ndarray State trajectory/trajectories. The states should start from zero and need to be integers. lagtimes : list or ndarray int Lagtimes for estimating the markov model given in [frames]. tmax : int Longest time to evaluate the CK equation given in [frames]. Returns    - cktest : dict Dictionary holding for each lagtime the ckequation and with 'md' the reference.",
"func":1
},
{
"ref":"msmhelper.benchmark.bh_test",
"url":1,
"doc":"This function is an alias of  buchete_hummer_test .See its docstring for further help.",
"func":1
},
{
"ref":"msmhelper.benchmark.buchete_hummer_test",
"url":1,
"doc":"Calculate the Buchete Hummer test. This method estimates the Buchete Hummer autocorrelation test. Projecting the state trajectory onto the right eigenvectors of the row normalized transition matrix  C_{lm} (t) = \\langle \\phi_l[s(\\tau +t)] \\phi_m[S(\\tau)]\\rangle where \\(\\phi_i\\) is the \\(i\\)-th right eigenvector. Buchete and Hummer[^2] showed that for a Markovian system it obeys an exponentil decay, corresponds to  C_{lm} (t) = \\delta_{lm} \\exp(-t / t_k) with the implied timescale \\(t_k = - \\tau_\\text{lag} / \\ln \\lambda_k\\). [^2]: Buchete and Hummer  Coarse master equations for peptide folding dynamics ,  J. Phys. Chem. , 112, 6057-6069 (2008), doi:[10.1021/jp0761665](https: doi.org/10.1021/jp0761665) Parameters      trajs : StateTraj or list or ndarray or list of ndarray State trajectory/trajectories. The states should start from zero and need to be integers. lagtime : int Lagtimes for estimating the markov model given in [frames]. tmax : int Longest time to evaluate the CK equation given in [frames]. Returns    - bhtest : dict Dictionary holding for each lagtime the ckequation and with 'md' the reference.",
"func":1
},
{
"ref":"msmhelper.statetraj",
"url":2,
"doc":"Class for handling discrete state trajectories. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved."
},
{
"ref":"msmhelper.statetraj.StateTraj",
"url":2,
"doc":"Class for handling discrete state trajectories. Initialize StateTraj and convert to index trajectories. If called with StateTraj instance, it will be retuned instead. Parameters      trajs : list or ndarray or list of ndarray State trajectory/trajectories. The states need to be integers."
},
{
"ref":"msmhelper.statetraj.StateTraj.states",
"url":2,
"doc":"Return active set of states. Returns    - states : ndarray Numpy array holding active set of states."
},
{
"ref":"msmhelper.statetraj.StateTraj.nstates",
"url":2,
"doc":"Return number of states. Returns    - nstates : int Number of states."
},
{
"ref":"msmhelper.statetraj.StateTraj.ntrajs",
"url":2,
"doc":"Return number of trajectories. Returns    - ntrajs : int Number of trajectories."
},
{
"ref":"msmhelper.statetraj.StateTraj.nframes",
"url":2,
"doc":"Return cummulated length of all trajectories. Returns    - nframes : int Number of frames of all trajectories."
},
{
"ref":"msmhelper.statetraj.StateTraj.state_trajs",
"url":2,
"doc":"Return state trajectory. Returns    - trajs : list of ndarrays List of ndarrays holding the input data."
},
{
"ref":"msmhelper.statetraj.StateTraj.state_trajs_flatten",
"url":2,
"doc":"Return flattened state trajectory. Returns    - trajs : ndarray 1D ndarrays representation of state trajectories."
},
{
"ref":"msmhelper.statetraj.StateTraj.index_trajs",
"url":2,
"doc":"Return index trajectory. Same as  self.trajs Returns    - trajs : list of ndarrays List of ndarrays holding the input data."
},
{
"ref":"msmhelper.statetraj.StateTraj.trajs",
"url":2,
"doc":"Return index trajectory. Returns    - trajs : list of ndarrays List of ndarrays holding the input data."
},
{
"ref":"msmhelper.statetraj.StateTraj.trajs_flatten",
"url":2,
"doc":"Return flattened index trajectory. Returns    - trajs : ndarray 1D ndarrays representation of index trajectories."
},
{
"ref":"msmhelper.statetraj.StateTraj.estimate_markov_model",
"url":2,
"doc":"Estimates Markov State Model. This method estimates the MSM based on the transition count matrix. Parameters      lagtime : int Lag time for estimating the markov model given in [frames]. Returns    - T : ndarray Transition rate matrix. permutation : ndarray Array with corresponding states.",
"func":1
},
{
"ref":"msmhelper.statetraj.StateTraj.state_to_idx",
"url":2,
"doc":"Get idx corresponding to state. Parameters      state : int State to get idx of. Returns    - idx : int Idx corresponding to state.",
"func":1
},
{
"ref":"msmhelper.statetraj.LumpedStateTraj",
"url":2,
"doc":"Class for handling lumped discrete state trajectories. Initialize LumpedStateTraj and convert to index trajectories. If called with LumpedStateTraj instance, it will be retuned instead. Parameters      macrotrajs : list or ndarray or list of ndarray Lumped state trajectory/trajectories. The states need to be integers and all states needs to correspond to union of microstates. microtrajs : list or ndarray or list of ndarray State trajectory/trajectories. EaThe states should start from zero and need to be integers."
},
{
"ref":"msmhelper.statetraj.LumpedStateTraj.states",
"url":2,
"doc":"Return active set of macrostates. Returns    - states : ndarray Numpy array holding active set of states."
},
{
"ref":"msmhelper.statetraj.LumpedStateTraj.nstates",
"url":2,
"doc":"Return number of macrostates. Returns    - nstates : int Number of states."
},
{
"ref":"msmhelper.statetraj.LumpedStateTraj.microstate_trajs",
"url":2,
"doc":"Return microstate trajectory. Returns    - trajs : list of ndarrays List of ndarrays holding the input data."
},
{
"ref":"msmhelper.statetraj.LumpedStateTraj.microstate_trajs_flatten",
"url":2,
"doc":"Return flattened state trajectory. Returns    - trajs : ndarray 1D ndarrays representation of state trajectories."
},
{
"ref":"msmhelper.statetraj.LumpedStateTraj.state_trajs",
"url":2,
"doc":"Return macrostate trajectory. Returns    - trajs : list of ndarrays List of ndarrays holding the input macrostate data."
},
{
"ref":"msmhelper.statetraj.LumpedStateTraj.state_trajs_flatten",
"url":2,
"doc":"Return flattened macrostate trajectory. Returns    - trajs : ndarray 1D ndarrays representation of macrostate trajectories."
},
{
"ref":"msmhelper.statetraj.LumpedStateTraj.microstates",
"url":2,
"doc":"Return active set of states. Returns    - states : ndarray Numpy array holding active set of states."
},
{
"ref":"msmhelper.statetraj.LumpedStateTraj.nmicrostates",
"url":2,
"doc":"Return number of states. Returns    - nstates : int Number of states."
},
{
"ref":"msmhelper.statetraj.LumpedStateTraj.estimate_markov_model",
"url":2,
"doc":"Estimates Markov State Model. This method estimates the microstate MSM based on the transition count matrix, followed by Szabo-Hummer projection formalism to macrostates. Parameters      lagtime : int Lag time for estimating the markov model given in [frames]. Returns    - T : ndarray Transition rate matrix. permutation : ndarray Array with corresponding states.",
"func":1
},
{
"ref":"msmhelper.statetraj.LumpedStateTraj.ntrajs",
"url":2,
"doc":"Return number of trajectories. Returns    - ntrajs : int Number of trajectories."
},
{
"ref":"msmhelper.statetraj.LumpedStateTraj.nframes",
"url":2,
"doc":"Return cummulated length of all trajectories. Returns    - nframes : int Number of frames of all trajectories."
},
{
"ref":"msmhelper.statetraj.LumpedStateTraj.index_trajs",
"url":2,
"doc":"Return index trajectory. Same as  self.trajs Returns    - trajs : list of ndarrays List of ndarrays holding the input data."
},
{
"ref":"msmhelper.statetraj.LumpedStateTraj.trajs",
"url":2,
"doc":"Return index trajectory. Returns    - trajs : list of ndarrays List of ndarrays holding the input data."
},
{
"ref":"msmhelper.statetraj.LumpedStateTraj.trajs_flatten",
"url":2,
"doc":"Return flattened index trajectory. Returns    - trajs : ndarray 1D ndarrays representation of index trajectories."
},
{
"ref":"msmhelper.statetraj.LumpedStateTraj.state_to_idx",
"url":2,
"doc":"Get idx corresponding to state. Parameters      state : int State to get idx of. Returns    - idx : int Idx corresponding to state.",
"func":1
},
{
"ref":"msmhelper.tools",
"url":3,
"doc":"Set of helpful functions. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved. TODO: - Correct border effects of running mean"
},
{
"ref":"msmhelper.tools.shift_data",
"url":3,
"doc":"Shift integer array (data) from old to new values. >  CAUTION: > The values of  val_old ,  val_new and  data needs to be integers. The basic function is based on Ashwini_Chaudhary solution: https: stackoverflow.com/a/29408060 Parameters      array : StateTraj or ndarray or list or list of ndarrays 1D data or a list of data. val_old : ndarray or list Values in data which should be replaced. All values needs to be within the range of  [data.min(), data.max()] val_new : ndarray or list Values which will be used instead of old ones. dtype : data-type, optional The desired data-type. Needs to be of type unsigned integer. Returns    - array : ndarray Shifted data in same shape as input.",
"func":1
},
{
"ref":"msmhelper.tools.rename_by_population",
"url":3,
"doc":"Rename states sorted by their population starting from 1. Parameters      trajs : list or ndarray or list of ndarrays State trajectory or list of state trajectories. return_permutation : bool Return additionaly the permutation to achieve performed renaming. Default is False. Returns    - trajs : ndarray Renamed data. permutation : ndarray Permutation going from old to new state nameing. So the  i th state of the new naming corresponds to the old state  permutation[i-1] .",
"func":1
},
{
"ref":"msmhelper.tools.rename_by_index",
"url":3,
"doc":"Rename states sorted by their numerical values starting from 0. Parameters      trajs : list or ndarray or list of ndarrays State trajectory or list of state trajectories. return_permutation : bool Return additionaly the permutation to achieve performed renaming. Default is False. Returns    - trajs : ndarray Renamed data. permutation : ndarray Permutation going from old to new state nameing. So the  i th state of the new naming corresponds to the old state  permutation[i-1] .",
"func":1
},
{
"ref":"msmhelper.tools.unique",
"url":3,
"doc":"Apply numpy.unique to traj. Parameters      trajs : list or ndarray or list of ndarrays State trajectory or list of state trajectories. kwargs Arguments of [numpy.unique()](NP_DOC.numpy.unique.html) Returns    - unique Array containing all states, see numpy for more details.",
"func":1
},
{
"ref":"msmhelper.tools.runningmean",
"url":3,
"doc":"Compute centered running average with given window size. This function returns the centered based running average of the given data. The output of this function is of the same length as the input, by assuming that the given data is zero before and after the given series. Hence, there are border affects which are not corrected. >  CAUTION: > If the given window is even (not symmetric) it will be shifted towards > the beginning of the current value. So for  window=4 , it will consider > the current position \\(i\\), the two to the left \\(i-2\\) and \\(i-1\\) and > one to the right \\(i+1\\). Function is taken from lapis: https: stackoverflow.com/questions/13728392/moving-average-or-running-mean Parameters      array : ndarray One dimensional numpy array. window : int Integer which specifies window-width. Returns    - array_rmean : ndarray Data which is time-averaged over the specified window.",
"func":1
},
{
"ref":"msmhelper.tools.swapcols",
"url":3,
"doc":"Interchange cols of an ndarray. This method swaps the specified columns.  todo Optimize memory usage Parameters      array : ndarray 2D numpy array. indicesold : integer or ndarray 1D array of indices. indicesnew : integer or ndarray 1D array of new indices Returns    - array_swapped : ndarray 2D numpy array with swappend columns.",
"func":1
},
{
"ref":"msmhelper.tools.get_runtime_user_information",
"url":3,
"doc":"Get user runtime information. >  CAUTION: > For python 3.5 or lower the date is not formatted and contains > microscends. Returns    - RUI : dict Holding username in 'user', pc name in 'pc', date of execution 'date', path of execution 'script_dir' and name of execution main file 'script_name'. In case of interactive usage, script_name is 'console'.",
"func":1
},
{
"ref":"msmhelper.tools.format_state_traj",
"url":3,
"doc":"Convert state trajectory to list of ndarrays. Parameters      trajs : list or ndarray or list of ndarray State trajectory/trajectories. The states should start from zero and need to be integers. Returns    - trajs : list of ndarray Return list of ndarrays of integers.",
"func":1
},
{
"ref":"msmhelper.tools.matrix_power",
"url":3,
"doc":"Calculate matrix power with np.linalg.matrix_power. Same as numpy function, except only for float matrices. See [np.linalg.matrix_power](NP_DOC/numpy.linalg.matrix_power.html). Parameters      matrix : ndarray 2d matrix of type float. power : int, float Power of matrix. Returns    - matpow : ndarray Matrix power.",
"func":1
},
{
"ref":"msmhelper.tools.find_first",
"url":3,
"doc":"Return first occurance of item in array.",
"func":1
},
{
"ref":"msmhelper.msm",
"url":4,
"doc":"Create Markov State Model. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved. Authors: Daniel Nagel Georg Diez"
},
{
"ref":"msmhelper.msm.estimate_markov_model",
"url":4,
"doc":"Estimates Markov State Model. This method estimates the MSM based on the transition count matrix. Parameters      trajs : statetraj or list or ndarray or list of ndarray State trajectory/trajectories. The states should start from zero and need to be integers. lagtime : int Lag time for estimating the markov model given in [frames]. Returns    - T : ndarray Transition rate matrix. permutation : ndarray Array with corresponding states.",
"func":1
},
{
"ref":"msmhelper.msm.row_normalize_matrix",
"url":4,
"doc":"Row normalize the given 2d matrix. Parameters      mat : ndarray Matrix to be row normalized. Returns    - mat : ndarray Normalized matrix.",
"func":1
},
{
"ref":"msmhelper.msm.implied_timescales",
"url":4,
"doc":"Calculate the implied timescales. Calculate the implied timescales for the given values.  todo catch if for higher lagtimes the dimensionality changes Parameters      trajs : StateTraj or list or ndarray or list of ndarray State trajectory/trajectories. The states should start from zero and need to be integers. lagtimes : list or ndarray int Lagtimes for estimating the markov model given in [frames]. This is not implemented yet! reversible : bool If reversibility should be enforced for the markov state model. Returns    - T : ndarray Transition rate matrix.",
"func":1
},
{
"ref":"msmhelper.msm.peq",
"url":4,
"doc":"This function is an alias of  equilibrium_population .See its docstring for further help.",
"func":1
},
{
"ref":"msmhelper.msm.equilibrium_population",
"url":4,
"doc":"Calculate equilibirum population. If there are non ergodic states, their population is set to zero. Parameters      tmat : ndarray Quadratic transition matrix, needs to be ergodic. allow_non_ergodic : bool If True only the largest ergodic subset will be used. Otherwise it will throw an error if not ergodic. Returns    - peq : ndarray Equilibrium population of input matrix.",
"func":1
},
{
"ref":"msmhelper.compare",
"url":5,
"doc":"Set of helpful functions for comparing markov state models. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved."
},
{
"ref":"msmhelper.compare.compare_discretization",
"url":5,
"doc":"Compare similarity of two state discretizations. This method compares the similarity of two state discretizations of the same dataset. There are two different methods, 'directed' gives a measure on how high is the probable to assign a frame correclty knowing the  traj1 . Hence splitting a state into many is not penalized, while merging multiple into a single state is. Selecting 'symmetric' it is check in both directions, so it checks for each state if it is possible to assigned it forward or backward. Hence, splitting and merging states is not penalized. Parameters      traj1 : StateTraj like First state discretization. traj2 : StateTraj like Second state discretization. method : ['symmetric', 'directed'] Selecting similarity norm. 'symmetric' compares if each frame is forward or backward assignable, while 'directed' checks only if it is forard assignable. Returns    - similarity : float Similarity going from [0, 1], where 1 means identical and 0 no similarity at all.",
"func":1
},
{
"ref":"msmhelper.md",
"url":6,
"doc":"Set of functions for analyzing the MD trajectory. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved."
},
{
"ref":"msmhelper.md.estimate_wt",
"url":6,
"doc":"This function is an alias of  estimate_waiting_times .See its docstring for further help.",
"func":1
},
{
"ref":"msmhelper.md.estimate_waiting_times",
"url":6,
"doc":"Estimates waiting times between stated states. The stated states (from/to) will be treated as a basin. The function calculates all transitions from first entering the start-basin until first reaching the final-basin. Parameters      trajs : statetraj or list or ndarray or list of ndarray State trajectory/trajectories. The states should start from zero and need to be integers. start : int or list of States to start counting. final : int or list of States to start counting. Returns    - wt : ndarray List of waiting times, given in frames.",
"func":1
},
{
"ref":"msmhelper.md.estimate_paths",
"url":6,
"doc":"Estimates paths and waiting times between stated states. The stated states (from/to) will be treated as a basin. The function calculates all transitions from first entering the start-basin until first reaching the final-basin. The results will be listed by the corresponding pathways, where loops are removed occuring first. Parameters      trajs : statetraj or list or ndarray or list of ndarray State trajectory/trajectories. The states should start from zero and need to be integers. start : int or list of States to start counting. final : int or list of States to start counting. Returns    - paths : ndarray List of waiting times, given in frames.",
"func":1
},
{
"ref":"msmhelper.md.estimate_msm_wt",
"url":6,
"doc":"This function is an alias of  estimate_msm_waiting_times .See its docstring for further help.",
"func":1
},
{
"ref":"msmhelper.md.estimate_msm_waiting_times",
"url":6,
"doc":"Estimates waiting times between stated states. The stated states (from/to) will be treated as a basin. The function calculates all transitions from first entering the start-basin until first reaching the final-basin. Parameters      trajs : statetraj or list or ndarray or list of ndarray State trajectory/trajectories. The states should start from zero and need to be integers. lagtime : int Lag time for estimating the markov model given in [frames]. start : int or list of States to start counting. final : int or list of States to start counting. steps : int Number of MCMC propagation steps of MCMC run. return_list : bool If true a list of all events is returned, else a dictionary is returned. Returns    - wt : ndarray List of waiting times, given in frames.",
"func":1
},
{
"ref":"msmhelper.md.propagate_MCMC",
"url":6,
"doc":"Propagate MCMC trajectory. Parameters      trajs : statetraj or list or ndarray or list of ndarray State trajectory/trajectories. The states should start from zero and need to be integers. lagtime : int Lag time for estimating the markov model given in [frames]. steps : int Number of MCMC propagation steps. start : int or list of, optional State to start propagating. Default (-1) is random state. Returns    - mcmc : ndarray MCMC trajecory.",
"func":1
},
{
"ref":"msmhelper.tests",
"url":7,
"doc":"Set of helpful test functions. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved."
},
{
"ref":"msmhelper.tests.is_quadratic",
"url":7,
"doc":"Check if matrix is quadratic. Parameters      matrix : ndarray, list of lists Matrix which is checked if is 2d array. Returns    - is_quadratic : bool",
"func":1
},
{
"ref":"msmhelper.tests.is_state_traj",
"url":7,
"doc":"Check if state trajectory is correct formatted. Parameters      trajs : list of ndarray State trajectory/trajectories need to be lists of ndarrays of integers. Returns    - is_state_traj : bool",
"func":1
},
{
"ref":"msmhelper.tests.is_index_traj",
"url":7,
"doc":"Check if states can be used as indices. Parameters      trajs : list of ndarray State trajectory/trajectories need to be lists of ndarrays of integers. Returns    - is_index : bool",
"func":1
},
{
"ref":"msmhelper.tests.is_tmat",
"url":7,
"doc":"This function is an alias of  is_transition_matrix .See its docstring for further help.",
"func":1
},
{
"ref":"msmhelper.tests.is_transition_matrix",
"url":7,
"doc":"Check if transition matrix. Rows and cols of zeros (non-visited states) are accepted. Parameters      matrix : ndarray Transition matrix. atol : float, optional Absolute tolerance. Returns    - is_tmat : bool",
"func":1
},
{
"ref":"msmhelper.tests.is_ergodic",
"url":7,
"doc":"Check if matrix is ergodic. Taken from: Wielandt, H. \"Unzerlegbare, Nicht Negativen Matrizen.\" Mathematische Zeitschrift. Vol. 52, 1950, pp. 642\u2013648. Parameters      matrix : ndarray Transition matrix. atol : float, optional Absolute tolerance. Returns    - is_ergodic : bool",
"func":1
},
{
"ref":"msmhelper.tests.is_fuzzy_ergodic",
"url":7,
"doc":"Check if matrix is ergodic, up to missing states or trap states. If there are two or more disjoint Parameters      matrix : ndarray Transition matrix. atol : float, optional Absolute tolerance. Returns    - is_fuzzy_ergodic : bool",
"func":1
},
{
"ref":"msmhelper.tests.ergodic_mask",
"url":7,
"doc":"Create mask for filtering ergodic submatrix. Parameters      matrix : ndarray Transition matrix. atol : float, optional Absolute tolerance. Returns    - mask : bool ndarray",
"func":1
},
{
"ref":"msmhelper.iotext",
"url":8,
"doc":"Input and output text files. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved."
},
{
"ref":"msmhelper.iotext.FileError",
"url":8,
"doc":"An exception for wrongly formated input files."
},
{
"ref":"msmhelper.iotext.opentxt",
"url":8,
"doc":"Open a text file. This method can load an nxm array of floats from an ascii file. It uses either pandas read_csv for a single comment or as fallback the slower numpy laodtxt for multiple comments.  warning In contrast to pandas the order of usecols will be used. So if using \u00b4data = opentxt( ., uscols=[1, 0])\u00b4 you acces the first column by  data[:, 0] and the second one by  data[:, 1] . Parameters      file_name : string Name of file to be opened. comment : str or array of str, optional Characters with which a comment starts. nrows : int, optional The maximum number of lines to be read usecols : int-array, optional Columns to be read from the file (zero indexed). skiprows : int, optional The number of leading rows which will be skipped. dtype : data-type, optional Data-type of the resulting array. Default: float. Returns    - array : ndarray Data read from the text file.",
"func":1
},
{
"ref":"msmhelper.iotext.savetxt",
"url":8,
"doc":"Save nxm array of floats to a text file. It uses numpys savetxt method and extends the header with information of execution. Parameters      file_name : string File name to store data. array : ndarray Data to be stored. header : str, optional Comment written into the header of the output file. fmt : str or sequence of strs, optional See numpy.savetxt fmt.",
"func":1
},
{
"ref":"msmhelper.iotext.opentxt_limits",
"url":8,
"doc":"Load file and split according to limit file. If limits_file is not provided it will return [traj]. Parameters      file_name : string Name of file to be opened. limits_file : str, optional File name of limit file. Should be single column ascii file. kwargs The Parameters defined in opentxt. Returns    - traj : ndarray Return array of subtrajectories.",
"func":1
},
{
"ref":"msmhelper.iotext.openmicrostates",
"url":8,
"doc":"Load 1d file and split according to limit file. Both, the limit file and the trajectory file needs to be a single column file. If limits_file is not provided it will return [traj]. The trajectory will of dtype np.int16, so the states needs to be smaller than 32767. Parameters      file_name : string Name of file to be opened. limits_file : str, optional File name of limit file. Should be single column ascii file. kwargs The Parameters defined in opentxt. Returns    - traj : ndarray Return array of subtrajectories.",
"func":1
},
{
"ref":"msmhelper.iotext.open_limits",
"url":8,
"doc":"Load and check limit file. The limits give the length of each single trajectory. So e.g. [5, 5, 5] for 3 equally-sized subtrajectories of length 5. Parameters      data_length : int Length of data read. limits_file : str, optional File name of limit file. Should be single column ascii file. Returns    - limits : ndarray Return cumsum of limits.",
"func":1
},
{
"ref":"msmhelper.linalg",
"url":9,
"doc":"Basic linear algebra method. BSD 3-Clause License Copyright (c) 2019-2020, Daniel Nagel All rights reserved."
},
{
"ref":"msmhelper.linalg.eigl",
"url":9,
"doc":"This function is an alias of  left_eigenvectors .See its docstring for further help.",
"func":1
},
{
"ref":"msmhelper.linalg.left_eigenvectors",
"url":9,
"doc":"Estimate left eigenvectors. Estimates the left eigenvectors and corresponding eigenvalues of a quadratic matrix. Parameters      matrix : ndarray Quadratic 2d matrix eigenvectors and eigenvalues or determined of. Returns    - eigenvalues : ndarray N eigenvalues sorted by their value (descending). eigenvectors : ndarray N eigenvectors sorted by descending eigenvalues.",
"func":1
},
{
"ref":"msmhelper.linalg.eig",
"url":9,
"doc":"This function is an alias of  right_eigenvectors .See its docstring for further help.",
"func":1
},
{
"ref":"msmhelper.linalg.right_eigenvectors",
"url":9,
"doc":"Estimate right eigenvectors. Estimates the right eigenvectors and corresponding eigenvalues of a quadratic matrix. Parameters      matrix : ndarray Quadratic 2d matrix eigenvectors and eigenvalues or determined of. Returns    - eigenvalues : ndarray N eigenvalues sorted by their value (descending). eigenvectors : ndarray N eigenvectors sorted by descending eigenvalues.",
"func":1
},
{
"ref":"msmhelper.linalg.eiglvals",
"url":9,
"doc":"This function is an alias of  left_eigenvalues .See its docstring for further help.",
"func":1
},
{
"ref":"msmhelper.linalg.left_eigenvalues",
"url":9,
"doc":"Estimate left eigenvalues. Estimates the left eigenvalues of a quadratic matrix. Parameters      matrix : ndarray Quadratic 2d matrix eigenvalues or determined of. Returns    - eigenvalues : ndarray N eigenvalues sorted by their value (descending).",
"func":1
},
{
"ref":"msmhelper.linalg.eigvals",
"url":9,
"doc":"This function is an alias of  right_eigenvalues .See its docstring for further help.",
"func":1
},
{
"ref":"msmhelper.linalg.right_eigenvalues",
"url":9,
"doc":"Estimate right eigenvalues. Estimates the right eigenvalues of a quadratic matrix. Parameters      matrix : ndarray Quadratic 2d matrix eigenvalues or determined of. Returns    - eigenvalues : ndarray N eigenvalues sorted by their value (descending).",
"func":1
}
]