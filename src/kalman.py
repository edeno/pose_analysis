import warnings

import numpy as np
from scipy.linalg import block_diag

try:
    from IPython import get_ipython

    if get_ipython() is not None:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
except ImportError:
    def tqdm(*args, **kwargs):
        if args:
            return args[0]
        return kwargs.get('iterable', None)


def kalman_filter(data, state_transition, state_to_observed,
                  state_covariance, measurement_covariance,
                  prior_state, prior_covariance, inverse=np.linalg.inv,
                  disable_progressbar=False):
    '''Handles missing observations

    Code modified from https://github.com/rlabbe/filterpy

    Parameters
    ----------
    data : ndarray, shape (n_time, n_observables)
        Observations from sensors
    state_transition : ndarray, shape (n_states, n_states)
        State transition matrix, F
    state_to_observed : ndarray, shape (n_observables, n_states)
        Measurement function/Observation Model, H
    state_covariance : ndarray, shape (n_states, n_states)
        Process covariance, Q
    measurement_covariance : ndarray, shape (n_observables, n_observables)
        Observation covariance, R
    prior_state : ndarray, shape (n_states,)
        Initial state mean
    prior_covariance : ndarray, shape (n_states, n_states)
        Initial state covariance (belief in state)
    inverse : function, optional

    Returns
    -------
    posterior_mean : ndarray (n_time, n_states)
    posterior_covariance : ndarray (n_time, n_states, n_states)

    '''
    n_time, n_states = data.shape[0], state_transition.shape[0]
    posterior_mean = np.zeros((n_time, n_states))
    posterior_covariance = np.zeros((n_time, n_states, n_states))

    posterior_mean[0] = prior_state.copy()
    posterior_covariance[0] = prior_covariance.copy()

    identity = np.eye(n_states)

    for time_ind in tqdm(np.arange(1, n_time), desc='frames',
                         disable=disable_progressbar):
        # Predict
        prior_mean = state_transition @ posterior_mean[time_ind - 1]
        prior_covariance = (
            state_transition @ posterior_covariance[time_ind - 1] @
            state_transition.T + state_covariance)

        # Update
        system_uncertainty = (
            state_to_observed @ prior_covariance @ state_to_observed.T
            + measurement_covariance)

        # kalman gain (n_states, n_observables)
        # prediction uncertainty vs. measurement uncertainty
        kalman_gain = prior_covariance @ state_to_observed.T @ inverse(
            system_uncertainty)
        # innovation
        prediction_error = (data[time_ind] - state_to_observed @ prior_mean)

        # Handle missing data by not updating the estimate and covariance
        is_missing = np.isnan(data[time_ind])
        prediction_error[is_missing] = 0.0
        kalman_gain[:, is_missing] = 0.0

        # Update mean
        posterior_mean[time_ind] = prior_mean + kalman_gain @ prediction_error

        # Update covariance
        I_KH = identity - kalman_gain @ state_to_observed
        posterior_covariance[time_ind] = (
            I_KH @ prior_covariance @ I_KH.T +
            kalman_gain @ measurement_covariance @ kalman_gain.T)

    return posterior_mean, posterior_covariance


def rts_smoother(posterior_mean, posterior_covariance, state_transition,
                 state_covariance, inverse=np.linalg.inv,
                 disable_progressbar=False):
    '''Runs the Rauch-Tung-Striebal Kalman smoother on a set of
    means and covariances computed by a Kalman filter.

    Code modified from https://github.com/rlabbe/filterpy.

    Parameters
    ----------
    posterior_mean : ndarray, shape (n_time, n_states)
    posterior_covariance : ndarray, shape (n_time, n_states, n_states)
    state_transition : ndarray, shape (n_states, n_states)
    state_covariance : ndarray, shape (n_states, n_states)
    inverse : function, optional

    Returns
    -------
    smoothed_mean : ndarray, shape (n_time, n_states)
    smoothed_covariances : ndarray, shape (n_time, n_states, n_states)

    '''
    n_time, n_states = posterior_mean.shape
    smoothed_mean = posterior_mean.copy()
    smoothed_covariances = posterior_covariance.copy()

    for time_ind in tqdm(np.arange(n_time - 2, -1, -1), desc='frames',
                         disable=disable_progressbar):
        prior_covariance = (state_transition @ posterior_covariance[time_ind] @
                            state_transition.T + state_covariance)
        smoother_gain = (posterior_covariance[time_ind] @ state_transition.T @
                         inverse(prior_covariance))
        smoothed_mean[time_ind] += smoother_gain @ (
            smoothed_mean[time_ind + 1] - state_transition @
            smoothed_mean[time_ind])
        smoothed_covariances[time_ind] += (smoother_gain @ (
            smoothed_covariances[time_ind + 1] - prior_covariance) @
            smoother_gain.T)

    return smoothed_mean, smoothed_covariances


def make_head_position_model(centroids, frame_rate, measurement_variance=1E-1,
                             process_variance=5):
    data = np.concatenate((centroids['red'], centroids['green']), axis=1)
    dt = 1 / frame_rate

    q = np.array([[0.25 * dt**4, 0.5 * dt**3, 0.5 * dt**2],
                  [0.5 * dt**3,        dt**2,          dt],
                  [0.5 * dt**2,           dt,         1.0]])
    state_covariance = block_diag(q, q) * process_variance

    f = np.array([[1.0,  dt, 0.5 * dt**2],
                  [0.0, 1.0,           0],
                  [0.0, 0.0,         1.0]])
    state_transition = block_diag(f, f)

    h = np.array([1, 0, 0])
    state_to_observed = np.concatenate((block_diag(h, h), block_diag(h, h)))

    # Observation covariance
    measurement_covariance = np.eye(4) * measurement_variance
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        initial_x = np.nanmean(data[:, [0, 2]], axis=1)
        initial_x = initial_x[np.nonzero(~np.isnan(initial_x))[0][0]]

        initial_y = np.nanmean(data[:, [1, 3]], axis=1)
        initial_y = initial_y[np.nonzero(~np.isnan(initial_y))[0][0]]

    prior_state = np.array([initial_x, 0, 0, initial_y, 0, 0])
    prior_covariance = np.diag([1, 250, 6000, 1, 250, 6000])

    return {'data': data,
            'state_transition': state_transition,
            'state_to_observed': state_to_observed,
            'state_covariance': state_covariance,
            'measurement_covariance': measurement_covariance,
            'prior_state': prior_state,
            'prior_covariance': prior_covariance}


def make_head_orientation_model(centroids, frame_rate,
                                measurement_variance=1E-1, process_variance=5):
    data = np.concatenate((centroids['red'], centroids['green']), axis=1)
    dt = 1 / frame_rate

    q = np.array([[0.25 * dt**4, 0.5 * dt**3, 0.5 * dt**2],
                  [0.5 * dt**3,        dt**2,          dt],
                  [0.5 * dt**2,           dt,         1.0]])
    state_covariance = block_diag(q, q, q, q) * process_variance

    f = np.array([[1.0,  dt, 0.5 * dt**2],
                  [0.0, 1.0,           0],
                  [0.0, 0.0,         1.0]])
    state_transition = block_diag(f, f, f, f)

    h = np.array([1, 0, 0])
    state_to_observed = block_diag(h, h, h, h)

    # Observation covariance
    measurement_covariance = np.eye(4) * measurement_variance

    x1 = data[~np.isnan(data[:, 0]), 0][0]
    y1 = data[~np.isnan(data[:, 1]), 1][0]
    x2 = data[~np.isnan(data[:, 2]), 2][0]
    y2 = data[~np.isnan(data[:, 3]), 3][0]
    prior_state = np.array([x1, 0, 0, y1, 0, 0,
                            x2, 0, 0, y2, 0, 0])
    prior_covariance = np.diag([1, 250, 6000, 1, 250, 6000,
                                1, 250, 6000, 1, 250, 6000])

    return {'data': data,
            'state_transition': state_transition,
            'state_to_observed': state_to_observed,
            'state_covariance': state_covariance,
            'measurement_covariance': measurement_covariance,
            'prior_state': prior_state,
            'prior_covariance': prior_covariance}


def filter_smooth_data(model, disable_progressbar):
    posterior_mean, posterior_covariance = kalman_filter(
        **model, disable_progressbar=disable_progressbar)
    posterior_mean, posterior_covariance = rts_smoother(
        posterior_mean, posterior_covariance, model['state_transition'],
        model['state_covariance'], disable_progressbar=disable_progressbar)

    return posterior_mean, posterior_covariance


# def extract_position_data(centroids, frame_rate, frame_size, n_frames,
#                           cm_to_pixels, disable_progressbar=False):
#     centroids = {color: convert_to_cm(data, frame_size, cm_to_pixels)
#                  for color, data in centroids.items()}
#
#     head_position_model = make_head_position_model(centroids, frame_rate)
#     head_position_mean, head_position_covariance = filter_smooth_data(
#         head_position_model, disable_progressbar=disable_progressbar)
