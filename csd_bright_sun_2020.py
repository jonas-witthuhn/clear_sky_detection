#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
The Bright-Sun clear-sky detection methodology, 2020.
- Cloudless sky CSDc
- Clear sun CSDs
Port from matlab script from Jamie Bright git repository:
    https://github.com/JamieMBright/csd-library.git
    BrightSun2020CSDc.m
    BrightSun2020CSDs.m

@author: Jonas Witthuhn

References
----------
Alia-Martinez et al. 2016:
    M. Alia-Martinez, J. Antonanzas, R. Urraca, F. J. Martinez-De-Pison, and
    F. Antonanzas-Torres, "Benchmark of algorithms for solar clear- sky
    detection", J. Renew. Sustain. Energy, vol. 8, no. 3, 2016.

Ellis 2018:
    PVLIB detect_ghics at commit
    f4e4ad3bbe335fa49bf300ce53784d69d719ca98
    https://github.com/pvlib/pvlib-python/pull/596/commits

Reno, M. J. and Hansen, C. W. 2016.:
    Identification of periods of clear sky irradiance in time series of
    GHI measurements. Renewable Energy. 90, 520-531.

Shen et al. 2018:
    Shen, Yu; Wei, Haikun; Zhu, Tingting; Zhao, Xin; Zhang, Kanjian. 2018. A
    Data-driven Clear Sky Model for Direct Normal Irradiance. IOP Conf.
    Series: Journal of Physics: Conf. Series. 1072.

Inman et al. 2015:
    Inman, Rich H; Edson, James G and Coimbra, Carlos F M. 2015. Impact of
    local broadband turbidity estimation on forecasting of clear sky direct
    normal irradiance Solar Energy. 117, 125-138

Gueymard et al. 2019:
    Gueymard, C.A; Bright, J.M.; Lingfors, D.; Habte, A. and Sengupta, M.
    2019. A posteriori clear-sky identification methods in solar irradiance
    time series: Review and preliminary validation using sky imagers.
    Renewable and Sustainable Energy Reviews. In review.

Larraneta et al. 2017:
    Larraneta, M; Reno, M J; Lillo-bravo, I; Silva-p, M A. 2017. Identifying
    periods of clear sky direct normal irradiance. Renewable Energy. 113,
    756-763.

Quesada-Ruiz et al. 2015:
    Quesada-Ruiz, Samuel Linares-Rodriguez, Alvaro Ruiz-arias, José A
    Pozo-Vázquez, David. 2015. ScienceDirect An advanced ANN-based method to
    estimate hourly solar radiation from multi-spectral MSG imagery. Solar
    Energy. 115, 494-504.

Zhang et al. 2018a:
    Zhang, Wenqi; Florita, Anthony R; Hodge, Bri-mathias; Mather, Barry.
    2018. Modeling and Simulation of High Frequency Solar Irradiance.
    Preprint to Journal of Photovoltaics.

Zhang et al. 2018b:
    Zhang, Wenqi Kleiber, William Florita, Anthony R Hodge, Bri-mathias
    Mather, Barry. 2018. A stochastic downscaling approach for generating
    high-frequency solar irradiance scenarios. Solar Energy. 176, 370-379.

"""
import numpy as np
from scipy.linalg import hankel
from scipy.optimize import minimize



### Parameterisation
# The methodology applies the Reno 2016 methodology to each sub-component
# of irradiance. There have been many proposed parametrisations in
# literature---all contained here and commented out if not used.
# This methodology mixes and matches the different parametrisations and
# also proposes new limits to certain tests.
## Reno 2016 Parameterisations for Optimisation
RENO_GHI_MEAN_DIFF = 75.
RENO_GHI_MAX_DIFF = 75.
RENO_GHI_LOWER_LINE_LEN = -5
RENO_GHI_UPPER_LINE_LEN = 10
RENO_GHI_VAR_DIFF = 0.1 # old reno 0.005
RENO_GHI_SLOPE_DEV = 8
# optimisation lower irradaicne limit threshold (Wm-2)
OPT_THRES = 30.
## Bright 2019 parameterisations for modified Reno
# GHI
GHI_C1_BIG_ZENITH = 0.5
GHI_C1_MEAN_DIFF = 0.125
GHI_C1_SMALL_ZENITH = 0.25
GHI_C2_BIG_ZENITH = 0.5
GHI_C2_MAX_DIFF = 0.125
GHI_C2_SMALL_ZENITH = 0.25
GHI_C3_LOWER_LINE_LEN = -0.5
GHI_C3_SMALL_ZENITH = -7
GHI_C4_VAR_DIFF = 0.4
GHI_C5_SLOPE_DEV = 15
GHI_C5_SMALL_ZENITH = GHI_C5_SLOPE_DEV * 3
# DIF
DIF_C1_BIG_ZENITH = 0.5
DIF_C1_MEAN_DIFF = 0.5
DIF_C1_SMALL_ZENITH = 0.25
DIF_C2_BIG_ZENITH = 0.5
DIF_C2_MAX_DIFF = 0.5
DIF_C2_SMALL_ZENITH = 0.25
DIF_C3_LOWER_LINE_LEN = -1.7
DIF_C3_SMALL_ZENITH = -6
DIF_C4_VAR_DIFF = 0.2
DIF_C5_SLOPE_DEV = 8
DIF_C5_SMALL_ZENITH = DIF_C5_SLOPE_DEV * 3
# set parametrisation for the turnaround in the zenith tuned parameters
# At this zenith parameter, the criteria 1,2,4,5 for Bright-Sun will be
# exactly as set.
ZENITH_TURN_AROUND = 30
# the window length is as standard across all the Reno variants at 10 mins
WINDOW_LENGTH = 10
# the max iterations comes from Ellis 2018 whereby optimisations are
# acheived up until 20 attempts or convergence is reached.
MAX_ITERATIONS = 20
# assert a minimum and maximum allowable modification to the clear-sky
# irradiance curve, where 1 = no change.
UPPER_ALPHA_LIMIT = 1.5
LOWER_ALPHA_LIMIT = 0.7


def _calculate_statistics(irradiance):
    """
    Calculate statistics for use in the Reno criteria.

    Parameters
    ----------
    irradiance : TYPE
        DESCRIPTION.

    Returns
    -------
    i_mean : TYPE
        DESCRIPTION.
    i_max : TYPE
        DESCRIPTION.
    i_diff : TYPE
        DESCRIPTION.
    i_slope : TYPE
        DESCRIPTION.
    i_slope_nstd : TYPE
        DESCRIPTION.
    i_line_length : TYPE
        DESCRIPTION.

    """
    # produce Hankel matrices for GHI and GHIcs as defined by window_length. a
    # Hankel matrix is essentially a staggered set of time series where each
    # column is the same data offset by 1 time step, this means that each row
    # has a "window" length time series, and each column has the entire time
    # series offset by the column number.
    i_window = hankel(irradiance, [np.nan]*WINDOW_LENGTH)

    # Calculate the statistics for use in the criteria. This approach is lifted
    # from Ellis 2018, Inman 2015 and Reno 2012/2016.
    # calculate measurement statistics for ghi.
    i_mean = np.nanmean(i_window, axis=1) # mean is taken on a row wise basis
    i_mean[i_mean==0]=np.nan
    i_max = np.nanmax(i_window, axis=1)
    i_max[i_max==0]=np.nan
    idx = ~np.isnan(i_mean)
    
    # diff reduces the matrix size by 1 due to delta between. The resultant is
    # increased by a row of NaNs representing that there is no t+1 for the
    # final time step.
    i_diff = np.vstack((np.diff(i_window.T).T, [np.nan]*WINDOW_LENGTH))
    
    # row wise standard deviation. pass ddof=1 for normalization by N-1
    idx[-2:]=False # last line will be all nan anyway, second last has only one value
    i_slope_nstd = np.zeros(len(i_mean))*np.nan
    i_slope_nstd[idx] = np.nanstd(i_diff[idx,:], axis=1, ddof=1) / i_mean[idx]
    i_slope_nstd[-2] = np.nanstd(i_diff[-2,:], ddof=0) / i_mean[-2]
    i_line_length = np.nansum(np.sqrt(i_diff**2 + 1), axis=1)
    return i_mean, i_max, i_diff, i_slope_nstd, i_line_length



def bright_sun_csds(date, zen, ghi, ghics, dif, difcs, longitude,
                    WINDOW_FILTER_1=1,
                    TOLERANCE_FILTER_1=1,
                    WINDOW_FILTER_2=1,
                    TOLERANCE_FILTER_2=1):
    return bright_sun_csdc(date, zen, ghi, ghics, dif, difcs, longitude,
                           WINDOW_FILTER_1,
                           TOLERANCE_FILTER_1,
                           WINDOW_FILTER_2,
                           TOLERANCE_FILTER_2)


def bright_sun_csdc(date, zen, ghi, ghics, dif, difcs, longitude,
                    WINDOW_FILTER_1=90,
                    TOLERANCE_FILTER_1=10,
                    WINDOW_FILTER_2=30,
                    TOLERANCE_FILTER_2=0):
    """
    The Bright-Sun clear-sky detection methodology, 2020. Cloudless sky CSDc.
    Assumption: all data is in 1-min resolution.

    Parameters
    ----------
    date : array, datetime64
        datetime objects for all inputs [UTC]
    zen : array, float
        Zenith angle in degrees. Corresponding to all inputs
    ghi : array, float
        Global horizontal irradiance. Not necessary
        to be continuous so long as all vectors perfectly corresponds.
    ghics : array, float
        Clear-sky global horizontal irradiance.
    dif : array, float
        Diffuse horizontal irradiance. Not necessary
        to be continuous so long as all vectors perfectly corresponds.
    difcs : array, float
        Clear-sky diffuse horizontal irradiance
    longitude : float, [degree east]
        Longitude in degree east of the measurement site.

    Returns
    -------
    csd : array, bool
        clear-sky detection for cloudless sky. "False" means that clouds were
        suspected whereas "True" means that the hour is clear.
    csd_tricomponent : array, bool
        clear-sky detection from the reno tri component optimisation, this
        returns many many more suspected clear-sky periods without the
        durational filters etc.
    ghics_optimised : array, float
        the resulting optimised clear-sky global irradiance. Note that closure
        is not maintained or guaranteed with this approach, GHIcs should be
        recalculated to maintian closure.
    dnics_optimised : array, float
        the resulting optimised clear-sky direct normal irradiance. Note that
        closure is not maintained or guaranteed with this approach, GHIcs
        should be recalculated to maintian closure.
    difcs_optimized : array, float
        the resulting optimised clear-sky diffuse irradiance. Note that closure
        is not maintained or guaranteed with this approach, GHIcs should be
        recalculated to maintian closure.

    """
    # Duration filter strictness
    # First durational window. Higher time but lower strictness.
    # The window is the duration size of the window in mins.
    #WINDOW_FILTER_1 = 90
    # Tolerance is the permissible number of CSD periods within the window.
    #TOLERANCE_FILTER_1 = 10
    # Second durational window. Lower time but higher strictness.
    #WINDOW_FILTER_2 = 30
    #TOLERANCE_FILTER_2 = 0

    ### make sure that inputs are numpy arrays
    date = np.array(date, dtype='datetime64[m]')
    zen = np.array(zen)
    ghi = np.array(ghi)
    ghics = np.array(ghics)
    dif = np.array(dif)
    difcs = np.array(difcs)
    longitude = float(longitude)

    ### check input lengths
    if len(np.unique([len(date),
                      len(zen),
                      len(ghi),
                      len(ghics),
                      len(dif),
                      len(difcs)])) != 1:
        raise ValueError("Input vars must be equal in length.")

    ### clean data if there is no radiation
    ghi[ghi<=0]=np.nan
    dif[dif<=0]=np.nan
    ghics[ghics<=0]=np.nan
    difcs[difcs<=0]=np.nan
    ### clean data from nan values
    # index where input is nan
    idx_nan = np.isnan(ghi)+np.isnan(ghics)+np.isnan(dif)+np.isnan(difcs)
    idx_nan += np.isnan(zen)+np.isnat(date)
    # index where input is not nan
    idx_not_nan = ~idx_nan.copy()
    # find the length of the required output for sanity
    n_output = len(ghi)
    # now clean input data
    date = date[idx_not_nan]
    zen = zen[idx_not_nan]
    ghi = ghi[idx_not_nan]
    ghics = ghics[idx_not_nan]
    dif = dif[idx_not_nan]
    difcs = difcs[idx_not_nan]
    # what remains is perfectly continuous time series of all the variables.
    # The tests do not matter should time skip because all tests are
    # comparative to the clear-sky irradiance and they too also skip.

    ### calculate LST (local solar time)
    lst = date + np.array(60.*(longitude/15.)).astype('timedelta64[m]')

    ### calculate often used variables
    mu0 = np.cos(np.deg2rad(zen))

    ### calculate DNI
    # make sure dif is not greater than ghi
    dif[dif > ghi] = ghi[dif > ghi]
    difcs[difcs > ghics] = ghics[difcs > ghics]
    # calculate dni from closure
    dni = (ghi-dif)/mu0
    dnics = (ghics-difcs)/mu0

    ### Initial CSD guess using Reno for Optimisation Only
    ##########################################################################
    ### GHI
    statistics = _calculate_statistics(ghi)
    meas_mean = statistics[0]
    meas_max = statistics[1]
    meas_slope = statistics[2]
    meas_slope_nstd = statistics[3]
    meas_line_length = statistics[4]
    ### GHIcs
    statistics = _calculate_statistics(ghics)
    clear_mean = statistics[0]
    clear_max = statistics[1]
    clear_slope = statistics[2]
    clear_line_length = statistics[4]
    ### GHI + GHIcs
    line_diff = meas_line_length - clear_line_length

    # initialise the criteria matrices where False = cloud. The default assumption
    # is that cloud is present, satisfaction of all criteria results in an
    # assertion of True for that time step.
    c1 = np.zeros(len(ghi)).astype(bool)
    c2 = np.zeros(len(ghi)).astype(bool)
    c3 = np.zeros(len(ghi)).astype(bool)
    c4 = np.zeros(len(ghi)).astype(bool)
    c5 = np.zeros(len(ghi)).astype(bool)
    c6 = np.zeros(len(ghi)).astype(bool)

    # perform criteria. We use the Ellis parameterisation for GHI.
    # these criteria are very well documented in all Reno, Ellis and Inman
    # papers. Particularly in the figure 5 of Inman 2015!

    # if clear>meas, this is ok
    c1[np.abs(meas_mean - clear_mean) < RENO_GHI_MEAN_DIFF] = True
    c2[np.abs(meas_max - clear_max) < RENO_GHI_MAX_DIFF] = True
    c3[(line_diff > RENO_GHI_LOWER_LINE_LEN)*(line_diff < RENO_GHI_UPPER_LINE_LEN)] = True
    c4[meas_slope_nstd < RENO_GHI_VAR_DIFF] = True
    # last row of meas_slope and clear_slope is always completely nan
    idx = np.nanmax(np.abs(meas_slope[:-1,:] - clear_slope[:-1,:]), axis=1) < RENO_GHI_SLOPE_DEV
    idx = list(idx)+[False]
    c5[idx] = True
    # this 6th criteria only exists in the Ellis 2018 coded version in PVlib.
    # It is a sensibility check for NaN results, though probably redundant due
    # to the prior screening.
    c6[(clear_mean != 0)*~np.isnan(clear_mean)] = True
    # should any criteria be False (cloud), then the time-step is deemed cloudy
    csd_ghiz = c1 * c2 * c3 * c4 * c5 * c6
    # initialise the first CSD guess (False - cloud, True - clear)
    csd_inital = csd_ghiz

    ### Optimisation of clear sky irradiance
    ##########################################################################
    # This is a similar principal as employed by Alia-Martinez 2016 and Ellis
    # 2018.

    ## Find all the unique days
    # find the unique days so that optimisation can be done on a daily basis.
    dtn = lst.astype('datetime64[D]')
    unique_days = np.unique(dtn)

    # initial clear-sky optimisation is that the clear-sky irradiance that is
    # input is already perfect (e.g. alpha=1), with each iteration and
    # optimisation, we will adjust 1 based on those periods identified as clear
    # by the modified Zhang.
    alpha_ghi = np.ones(len(unique_days))
    alpha_dif = np.ones(len(unique_days))
    alpha_dni = np.ones(len(unique_days))

    # First we look through every single day identified in the input data
    for d, uday in enumerate(unique_days):
        # find indices within the data that correspond to this day
        idxs = dtn == uday

        # isolate the variables to only this day
        ghid = ghi[idxs]
        ghicsd = ghics[idxs]
        difd = dif[idxs]
        difcsd = difcs[idxs]
        dnid = dni[idxs]
        dnicsd = dnics[idxs]
        csdd = csd_inital[idxs]

        # apply the optimisation lower limit threshold
        csdd[(difd < OPT_THRES) + (ghid < OPT_THRES)] = False

        # isolate clear sky within this day
        test_ghi = ghid[csdd]
        test_ghics = ghicsd[csdd]
        test_dif = difd[csdd]
        test_difcs = difcsd[csdd]
        test_dni = dnid[csdd]
        test_dnics = dnicsd[csdd]

        # define the rmse functions for optimisation strategy
        rmse_ghi = lambda x: np.sqrt(np.nanmean((test_ghi - x*test_ghics)**2))
        rmse_dif = lambda x: np.sqrt(np.nanmean((test_dif - x*test_difcs)**2))
        rmse_dni = lambda x: np.sqrt(np.nanmean((test_dni - x*test_dnics)**2))

        # if there were at least 60 clear periods to optimise with, then we can
        # proceed. Fewer sites may lead to really strange optimisation, which
        # is undesireable. 60 seems a solid amount for trustworthy
        # optimsiations.
        if np.count_nonzero(csdd) > 60:
            ## Beginning of optimisation
            # initialize the "current_alpha", which is always 1
            current_alpha = alpha_ghi[d]
            # set the previous_alpha as NaN so that it cannot pass the if
            # statement featured in the following "while" loop
            previous_alpha = np.nan
            # initialize the iteration count
            _iter = 0
            # Enter the while loop, that will continue until the optimisation
            # has converged withun 0.00001 or if 20 iterations has occured.
            while (_iter < MAX_ITERATIONS) and \
                  (np.round(current_alpha, 4) != np.round(previous_alpha, 4)):
                # run a Multidimensional unconstrained nonlinear minimization
                # (Nelder-Mead) search of the function=rmse minimising through alpha.
                previous_alpha = current_alpha
                res = minimize(rmse_ghi, current_alpha, method='Nelder-Mead')
                current_alpha = float(res.x)
                # update the iteration count
                _iter += 1
            alpha_ghi[d] = current_alpha

            ## repeat for DIF
            current_alpha = alpha_dif[d]
            previous_alpha = np.nan
            _iter = 0
            while (_iter < MAX_ITERATIONS) and \
                  (np.round(current_alpha, 4) != np.round(previous_alpha, 4)):
                previous_alpha = current_alpha
                res = minimize(rmse_dif, current_alpha, method='Nelder-Mead')
                current_alpha = float(res.x)
                _iter += 1
            alpha_dif[d] = current_alpha

            ## repeat for DNI
            current_alpha = alpha_dni[d]
            previous_alpha = np.nan
            _iter = 0
            while (_iter < MAX_ITERATIONS) and \
                  (np.round(current_alpha, 4) != np.round(previous_alpha, 4)):
                previous_alpha = current_alpha
                res = minimize(rmse_dni, current_alpha, method='Nelder-Mead')
                current_alpha = float(res.x)
                _iter += 1
            alpha_dni[d] = current_alpha
        # Occasionally, the Reno fails at identifying clear periods (such as in
        # extreme latitudes in the antarctic), as such, we assert some upper
        # limits to the possible alpha correction.
        #GHI
        temp = alpha_ghi[d]
        if temp > UPPER_ALPHA_LIMIT: temp = UPPER_ALPHA_LIMIT
        if temp < LOWER_ALPHA_LIMIT: temp = LOWER_ALPHA_LIMIT
        alpha_ghi[d] = temp
        #DIF
        temp = alpha_dif[d]
        if temp > UPPER_ALPHA_LIMIT: temp = UPPER_ALPHA_LIMIT
        if temp < LOWER_ALPHA_LIMIT: temp = LOWER_ALPHA_LIMIT
        alpha_dif[d] = temp
        #DNI
        temp = alpha_dni[d]
        if temp > UPPER_ALPHA_LIMIT: temp = UPPER_ALPHA_LIMIT
        if temp < LOWER_ALPHA_LIMIT: temp = LOWER_ALPHA_LIMIT
        alpha_dni[d] = temp
        # Apply the clear-sky correction factors to the clear-sky estimates for
        # this day. Should the estimate already be ideal, alpha=1 and no change
        # will occur.
        ghics[idxs] = ghics[idxs]*alpha_ghi[d]
        difcs[idxs] = difcs[idxs]*alpha_dif[d]
        dnics[idxs] = dnics[idxs]*alpha_dni[d]

    ### perform tri-component analysis.
    ##########################################################################
    # tri component analysis is a fancy way of saying we perform criteria on
    # all irradiance components (global, direct and diffuse). Each subcomponent
    # undergoes its own CSD methodology and only if all component CSDs
    # corroborate that it is infact cloud free does the tri-component CSD time
    # series register as clear.
    ## GHI
    # reuse the previousely calculated statistics
    # initialize again criteria matrices
    c1 = np.zeros(len(ghi)).astype(bool)
    c2 = np.zeros(len(ghi)).astype(bool)
    c3 = np.zeros(len(ghi)).astype(bool)
    c4 = np.zeros(len(ghi)).astype(bool)
    c5 = np.zeros(len(ghi)).astype(bool)
    c6 = np.zeros(len(ghi)).astype(bool)
    # We also introduce the zenith flexibility introduced by Larraneta 2017.
    # We observe in many occasions events where high zenith angles behave
    # smoothly as if clear sky, however, are often lower than the clear sky
    # curve (particularly if not a very good clearsky model. For that reason,
    # we must apply some flexibility with zenith angle.
    z = np.arange(20, 90.01, 0.01)
    zidx = np.searchsorted(z, ZENITH_TURN_AROUND)
    inds = np.searchsorted(z, zen)
    # for zen > 90 inds will now have a value of -1 instead of len(z)
    inds[inds >= len(z)] = -1
    # produce a linearly spaced correction factor as proposed by Larraneta
    # 2017. Note that they used three fixed bins whereas  we smooth this
    # through interpolation
    c1_lim = list(np.linspace(GHI_C1_BIG_ZENITH, GHI_C1_MEAN_DIFF, zidx))
    c1_lim.extend(np.linspace(GHI_C1_MEAN_DIFF, GHI_C1_SMALL_ZENITH, len(z)-zidx))
    c1_lim = np.array(c1_lim[::-1])
    # find the indices where the zenith angles correspond.
    c1_lim = c1_lim[inds]
    # same principle for low and high zenith
    c2_lim = list(np.linspace(GHI_C2_BIG_ZENITH, GHI_C2_MAX_DIFF, zidx))
    c2_lim.extend(np.linspace(GHI_C2_MAX_DIFF, GHI_C2_SMALL_ZENITH, len(z)-zidx))
    c2_lim = np.array(c2_lim[::-1])
    c2_lim = c2_lim[inds]
    # apply a relaxation for very low zenith to c3 and c5
    c3_lim_lower = list(np.linspace(GHI_C3_LOWER_LINE_LEN, GHI_C3_LOWER_LINE_LEN, zidx))
    c3_lim_lower.extend(np.linspace(GHI_C3_LOWER_LINE_LEN, GHI_C3_SMALL_ZENITH, len(z)-zidx))
    c3_lim_lower = np.array(c3_lim_lower[::-1])
    c3_lim_lower = c3_lim_lower[inds]
    c3_lim_upper = np.abs(c3_lim_lower)

    c5_lim = list(np.linspace(GHI_C5_SLOPE_DEV, GHI_C5_SLOPE_DEV, zidx))
    c5_lim.extend(np.linspace(GHI_C5_SLOPE_DEV, GHI_C5_SMALL_ZENITH, len(z)-zidx))
    c5_lim = np.array(c5_lim[::-1])
    c5_lim = c5_lim[inds]

    # perform criteria. We use the Ellis parameterisation for GHI.
    # these criteria are very well documented in all Reno, Ellis and Inman
    # papers. Particularly in the figure 5 of Inman 2015!
    c1[np.abs(meas_mean - clear_mean)/clear_mean < c1_lim] = True
    c2[np.abs(meas_max - clear_max)/clear_max < c2_lim] = True
    c3[(line_diff > c3_lim_lower)*(line_diff < c3_lim_upper)] = True
    c4[meas_slope_nstd < GHI_C4_VAR_DIFF] = True
    idx = np.nanmax(np.abs(meas_slope[:-1,:] - clear_slope[:-1,:]), axis=1) < c5_lim[:-1]
    idx = list(idx)+[False]
    c5[idx] = True
    c6[(clear_mean != 0)*~np.isnan(clear_mean)] = True
    csd_ghi = c1 * c2 * c3 * c4 * c5 * c6

    ## GHI
    statistics = _calculate_statistics(dif)
    meas_mean = statistics[0]
    meas_max = statistics[1]
    meas_slope = statistics[2]
    meas_slope_nstd = statistics[3]
    meas_line_length = statistics[4]
    ### GHIcs
    statistics = _calculate_statistics(difcs)
    clear_mean = statistics[0]
    clear_max = statistics[1]
    clear_slope = statistics[2]
    clear_line_length = statistics[4]
    ### GHI + GHIcs
    line_diff = meas_line_length - clear_line_length
    # initialize again criteria matrices
    c1 = np.zeros(len(ghi)).astype(bool)
    c2 = np.zeros(len(ghi)).astype(bool)
    c3 = np.zeros(len(ghi)).astype(bool)
    c4 = np.zeros(len(ghi)).astype(bool)
    c5 = np.zeros(len(ghi)).astype(bool)
    c6 = np.zeros(len(ghi)).astype(bool)

    c1_lim = list(np.linspace(DIF_C1_BIG_ZENITH, DIF_C1_MEAN_DIFF, zidx))
    c1_lim.extend(np.linspace(DIF_C1_MEAN_DIFF, DIF_C1_SMALL_ZENITH, len(z)-zidx))
    c1_lim = np.array(c1_lim[::-1])
    inds = np.searchsorted(z, zen)
    inds[inds >= len(z)] = -1
    c1_lim = c1_lim[inds]

    c2_lim = list(np.linspace(DIF_C2_BIG_ZENITH, DIF_C2_MAX_DIFF, zidx))
    c2_lim.extend(np.linspace(DIF_C2_MAX_DIFF, DIF_C2_SMALL_ZENITH, len(z)-zidx))
    c2_lim = np.array(c2_lim[::-1])
    c2_lim = c2_lim[inds]

    c3_lim_lower = list(np.linspace(DIF_C3_LOWER_LINE_LEN, DIF_C3_LOWER_LINE_LEN, zidx))
    c3_lim_lower.extend(np.linspace(DIF_C3_LOWER_LINE_LEN, DIF_C3_SMALL_ZENITH, len(z)-zidx))
    c3_lim_lower = np.array(c3_lim_lower[::-1])
    c3_lim_lower = c3_lim_lower[inds]
    c3_lim_upper = np.abs(c3_lim_lower)

    c5_lim = list(np.linspace(DIF_C5_SLOPE_DEV, DIF_C5_SLOPE_DEV, zidx))
    c5_lim.extend(np.linspace(DIF_C5_SLOPE_DEV, DIF_C5_SMALL_ZENITH, len(z)-zidx))
    c5_lim = np.array(c5_lim[::-1])
    c5_lim = c5_lim[inds]

    c1[np.abs(meas_mean - clear_mean)/clear_mean < c1_lim] = True
    c2[np.abs(meas_max - clear_max)/clear_max < c2_lim] = True
    c3[(line_diff > c3_lim_lower)*(line_diff < c3_lim_upper)] = True
    c4[meas_slope_nstd < GHI_C4_VAR_DIFF] = True
#     c5[np.nanmax(np.abs(meas_slope - clear_slope), axis=1) < c5_lim] = True
    idx = np.nanmax(np.abs(meas_slope[:-1,:] - clear_slope[:-1,:]), axis=1) < c5_lim[:-1]
    idx = list(idx)+[False]
    c5[idx] = True
    c6[(clear_mean != 0)*~np.isnan(clear_mean)] = True

    csd_dif = c1 * c2 * c3 * c4 * c5 * c6

    ### DNI
    # Apply the Quesada-Ruiz 2015 methodology, though due to optimisation of
    # the clear-sky curves, we can afford to be more conservative and apply
    # 0.95 instead.
    z = np.arange(30, 90.01, 0.01)
    kc_lims = np.linspace(0.5, 0.9, len(z))[::-1]
    inds = np.searchsorted(z, zen)
    inds[inds >= len(z)] = -1
    kc_lim = kc_lims[inds]

    # find the clear sky index for DNI/beam (kcb)
    kcb = dni / dnics
    # preallocate DNIcsd
    csd_dni = np.zeros(len(ghi)).astype(bool)
    # kcb below the limit means we assert that period as clear (True)
    csd_dni[kcb > kc_lim] = True

    ### Combined CSD
    csd_overall = csd_ghi * csd_dif * csd_dni


    ### Duration criteria
    ###########################################################################
    # we build an hour durational filter looking ahead and behind for 45
    # minutes. should there not have been a continuous CSD for an hour, then we
    # reject all those instances.

    # safety check on windows
    if WINDOW_FILTER_1 <= 1:
        if TOLERANCE_FILTER_1 == WINDOW_FILTER_1:
            TOLERANCE_FILTER_1 = 2
        WINDOW_FILTER_1 = 2
    if WINDOW_FILTER_2 <= 1:
        if TOLERANCE_FILTER_2 == WINDOW_FILTER_2:
            TOLERANCE_FILTER_2 = 2
        WINDOW_FILTER_2 = 2

    # First Duration Filter
    csdh_1st_duration = np.nansum(hankel(~csd_overall, [np.nan]*WINDOW_FILTER_1),
                                  axis=1)
    csdh_1st_duration = np.array([np.nan]*int(WINDOW_FILTER_1/2) \
                                 + list(csdh_1st_duration[:-int(WINDOW_FILTER_1/2)]))
    csd_1st_duration = ~(csdh_1st_duration > TOLERANCE_FILTER_1)
    # this test makes morning and evening during sunlight hours impossible,
    # therefore we relax the CSD at low zenith. However, in polar climates,
    # some whole weeks can be >80.
    
    A = np.arange(len(zen)).astype(int)
    B = np.argwhere(np.round(zen) == 85.).astype(int)
    # reduce B so that it has only two indexes per day (sunrise, sunset)
    b = [B[0]]
    for bt in B[1:]:
        if bt > 50+b[-1]:
            b.append(int(bt))
    b = np.array(b, dtype=int)[:,np.newaxis]
    D = np.abs(A-b)
    dist_to_sunrise_set = np.min(D, axis=0)
    csd_1st_duration[dist_to_sunrise_set < WINDOW_FILTER_1] = True

    # Second Duration Filter
    csdh_2st_duration = np.nansum(hankel(~csd_overall, [np.nan]*WINDOW_FILTER_2),
                                  axis=1)
    csdh_2st_duration = np.array([np.nan]*int(WINDOW_FILTER_2/2) \
                                 + list(csdh_2st_duration[:-int(WINDOW_FILTER_2/2)]))
    csd_2st_duration = ~(csdh_2st_duration > TOLERANCE_FILTER_2)

    # As sun-down is considered cloudy (which it should not be), the second
    # filter unfavourably elimiates these periods. For that reason, the scond
    # filter during sun down proximity (defined by the
    # distance_to_nearest_sunset principle) is overridden by a less strict
    # duration filter.
    # Proximity to Sundown Duration Filter
    window_3rd_duration = 10
    tolerance_3rd_duration = 2
    csdh_3st_duration = np.nansum(hankel(~csd_overall, [np.nan]*window_3rd_duration),
                                  axis=1)
    csdh_3st_duration = np.array([np.nan]*int(window_3rd_duration/2) \
                                 + list(csdh_3st_duration[:-int(window_3rd_duration/2)]))
    csd_3st_duration = ~(csdh_3st_duration < tolerance_3rd_duration)

    # override the 2nd duration filter at peak proximity to sunrise defined as
    # within the first filter window mins of sunrise and where the 10min filter
    # found it permissable.
    csd_2st_duration[csd_3st_duration*(dist_to_sunrise_set < WINDOW_FILTER_1)] = True

    # CSD corroboration from the tri-component and two duration filters
    csd_nan_indexed = csd_overall*csd_1st_duration*csd_2st_duration

    # outputs that are indexed
    csd = np.zeros(n_output).astype(bool)
    csd[idx_not_nan] = csd_nan_indexed

    # output the irradiance components optimized
    ghics_optimised = np.ones(n_output)*np.nan
    dnics_optimised = np.ones(n_output)*np.nan
    difcs_optimised = np.ones(n_output)*np.nan

    ghics_optimised[idx_not_nan] = ghics
    dnics_optimised[idx_not_nan] = dnics
    difcs_optimised[idx_not_nan] = difcs

    # output the CSD from tri component
    csd_tricomponent = np.zeros(n_output).astype(bool)
    csd_tricomponent[idx_not_nan] = csd_overall
    
    # mask missing values
    csd_tricomponent=np.ma.masked_where(idx_nan, csd_tricomponent)
    csd=np.ma.masked_where(idx_nan, csd)
    return csd, csd_tricomponent, ghics_optimised, dnics_optimised, difcs_optimised
