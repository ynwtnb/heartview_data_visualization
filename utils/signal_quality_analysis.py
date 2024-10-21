import pandas as pd
import numpy as np

def __calculate_correct_beats(df_ppg, df_ecg, lag, corrected = False, seg = None):
    '''
    Calculate the number of correct beats (beats in agreement) between the corrected PPG signal and the reference ECG signal.

    Parameters
    ------------
    lag: np.timedelta64
        Lag in milliseconds.
    corrected: bool
        If True, calculate the number of correct beats between the corrected PPG signal and the reference ECG signal.
    seg: int
        Segment number. If None, calculate for the whole signal.
    '''
    if seg == None:
        ref_df = df_ecg
        df = df_ppg
    
    else:
        ref_df = df_ecg[df_ecg['Segment'] == seg]
        df = df_ppg[df_ppg['Segment'] == seg]
        
    ref_beats_timestamp = ref_df.loc[ref_df['Peak'] == 1, 'Timestamp'].values
    if corrected:
        beats_timestamp = df.loc[df['Peak after correction'] == 1, 'Timestamp'].values
    else:
        beats_timestamp = df.loc[df['Peak'] == 1, 'Timestamp'].values
    
    shifted_beats = beats_timestamp + lag

    time_diffs = np.abs(ref_beats_timestamp[:, None] - shifted_beats)
    min_diffs = np.min(time_diffs, axis = 1)
    correct_ref_indices = np.where(min_diffs < np.timedelta64(150, 'ms'))[0]
    correct_indices = np.argmin(time_diffs, axis=1)[correct_ref_indices]

    n_correct_beats = len(correct_ref_indices)

    return n_correct_beats, correct_indices, correct_ref_indices

def __identify_best_lag(df_ppg, df_ecg, lags, corrected = False):
    '''
    Calculate the best lag for the corrected PPG signal to align with the reference ECG signal.

    Parameters
    ------------
    lags: array-like or int
        List of lags or a single lag to be tested.
    corrected: bool
        If True, calculate the best lag for the corrected PPG signal.
    
    Returns
    ------------
    best_lag: np.timedelta64
        The best lag.
    max_correct_beats: int
        The number of correct beats with the best lag.
    df_ppg: pandas.DataFrame
        Dataframe of the corrected PPG signal (with lagged timestamps).
    df_ecg: pandas.DataFrame
        Dataframe of the reference ECG signal.
    '''
    best_lag = None
    max_correct_beats = -1
    best_correct_indices = []

    if type(lags) == list:
        for lag in lags:
            n_correct_beats, correct_indices, correct_ref_indices = __calculate_correct_beats(df_ppg = df_ppg, df_ecg = df_ecg, lag = lag, corrected = corrected)
            if n_correct_beats > max_correct_beats:
                max_correct_beats = n_correct_beats
                best_lag = lag
                best_correct_indices = correct_indices
                best_correct_ref_indices = correct_ref_indices
    else:
        max_correct_beats, best_correct_indices, best_correct_ref_indices = __calculate_correct_beats(df_ppg = df_ppg, df_ecg = df_ecg, lag = lags, corrected = corrected)
        best_lag = lags

    df_ppg_lagged = add_lag(df_ppg, best_lag, corrected)

    df_ecg_correct = __add_correct_beats(df_ecg, best_correct_ref_indices, corrected)
    df_ppg_correct = __add_correct_beats(df_ppg_lagged, best_correct_indices, corrected)

    return best_lag, max_correct_beats, df_ppg_correct, df_ecg_correct

def add_lag(df, lag, corrected = False):
    '''
    Add lag to the timestamps in order to align ECG and PPG signals.
    The specified lag will be added to the 'Timestamp' column.

    Parameters:
    ------------
    df: pandas.DataFrame
        Dataframe of the signal.
        This dataframe should contain 'Timestamp' column.
    lag: np.timedelta64
        Lag in milliseconds.
    corrected: bool
        If True, the lag is added to the timestamp of the corrected beats.
        If False, the lag is added to the timestamp of the original beats.
    '''
    df_lagged = df.copy()
    if corrected:
        df_lagged['Lagged timestamp after correction'] = df_lagged['Timestamp'] + lag
    else:
        df_lagged['Lagged timestamp'] = df_lagged['Timestamp'] + lag
    
    return df_lagged

def __add_correct_beats(df, correct_peaks, corrected = False):
    '''
    Add 'Correct peak' column to the raw data to indicate the correct beats.

    Parameters:
    ------------
    df: pandas.DataFrame
        Dataframe of the signal.
    correct_peaks: List
        A list of indices of the correct peaks.
    corrected: bool
        If True, the correct peaks will be added to the corrected beats.
        If False, the correct peaks will be added to the original beats.
    '''
    df = df.copy()

    if corrected:
        df['Correct peak after correction'] = 0
        try:
            correct_indices = df[df['Peak after correction'] == 1].reset_index(drop = False).iloc[correct_peaks]['index']
        except:
            # Check if the signal is reference signal
            assert 'Lagged timestamp' not in df.columns, 'You need to run beat_correction() method before adding correct beats.'
            correct_indices = df[df['Peak'] == 1].reset_index(drop = False).iloc[correct_peaks]['index']
        df.loc[df.index.isin(correct_indices), 'Correct peak after correction'] = 1
        
    else:
        df['Correct peak'] = 0
        correct_indices = df[df['Peak'] == 1].reset_index(drop = False).iloc[correct_peaks]['index']
        df.loc[df.index.isin(correct_indices), 'Correct peak'] = 1
    
    return df

def compare_beat_locations_Charlton(df_ppg, df_ecg, lag_type = 'best', lag = None):
    '''
    Compare the beat locations between the PPG signal and the reference ECG signal using Charlton et al.'s method.

    Parameters
    -----------
    df_ppg: pandas.DataFrame
        Dataframe of the PPG signal.
        This dataframe should contain 'Peak' and 'Peak after correction' columns.
    df_ecg: pandas.DataFrame
        Dataframe of the reference ECG signal.
        This dataframe should contain 'Peak' column.
    lag_type: str
        Type of lag to be used for the comparison. Choose from 'best', 'detected', 'corrected', or 'average'.
        This parameter determines whether to fix the lag between the corrected/uncorrected PPG signal, and if so, which lag to use.
        If 'best,' separate best lag for corrected and uncorrected signals are used.
        If 'detected,' the best lag for uncorrected beats will be used for both uncorrected and corrected signals.
        If 'corrected,' the best lag for corrected beats will be used for both uncorrected and corrected signals.
        If 'average,' the average of the best lags for detected and corrected beats will be used for both uncorrected and corrected signals.'
        If 'fixed,' the lag parameter will be used.
    lag: np.timedelta64
        Lag in seconds. If lag_type is 'fixed', this parameter will be used.
    
    Returns
    ---------
    result_df: pandas.DataFrame
        Dataframe containing the comparison results.
    df_ppg_combined: pandas.DataFrame
        Dataframe of the PPG signal with the lagged timestamps and correct beats.
        This dataframe contains 'Lagged timestamps', 'Lagged timestamps corrected', 'Correct peak', and 'Correct peak after correction' columns.
    df_ecg_combined: pandas.DataFrame
        Dataframe of the reference ECG signal with the correct beats.
        This dataframe contains 'Lagged timestamps', 'Lagged timestamps corrected', 'Correct peak', and 'Correct peak after correction' columns.

    Reference
    ---------
    Charlton PH, Kotzen K, Mejía-Mejía E, Aston PJ, Budidha K, Mant J, et al. Detecting beats in the photoplethysmogram: benchmarking open-source algorithms. Physiological Measurement 2022;43. https://doi.org/10.1088/1361-6579/ac826d.
    '''
    lags = [np.timedelta64(int(lag * 1000), 'ms') for lag in np.arange(-2, 2.02, 0.02)]  # from -10s to 10s in 20ms increments
    if lag_type == 'best':
        best_lag_detected, max_correct_beats_detected, df_ppg_detected, df_ecg_detected = __identify_best_lag(df_ppg, df_ecg, lags, corrected = False)
        best_lag_corrected, max_correct_beats_corrected, df_ppg_combined, df_ecg_combined = __identify_best_lag(df_ppg_detected, df_ecg_detected, lags, corrected = True)
    elif lag_type == 'detected':
        best_lag_detected, max_correct_beats_detected, df_ppg_detected, df_ecg_detected = __identify_best_lag(df_ppg, df_ecg, lags, corrected = False)
        best_lag_corrected, max_correct_beats_corrected, df_ppg_combined, df_ecg_combined = __identify_best_lag(df_ppg_detected, df_ecg_detected, best_lag_detected, corrected = True)
    elif lag_type == 'corrected':
        best_lag_corrected, max_correct_beats_corrected, df_ppg_corrected, df_ecg_corrected = __identify_best_lag(df_ppg, df_ecg, lags, corrected = True)
        best_lag_detected, max_correct_beats_detected, df_ppg_combined, df_ecg_combined = __identify_best_lag(df_ppg_corrected, df_ecg_corrected, best_lag_corrected, corrected = False)
    elif lag_type == 'average':
        best_lag_detected, max_correct_beats_detected, df_ppg_detected, df_ecg_detected = __identify_best_lag(df_ppg, df_ecg, lags, corrected = False)
        best_lag_corrected, max_correct_beats_corrected, df_ppg_corrected, df_ecg_corrected = __identify_best_lag(df_ppg_detected, df_ecg_detected, lags, corrected = True)
        best_lag_average = (best_lag_detected + best_lag_corrected) / 2
        best_lag_detected, max_correct_beats_detected, df_ppg_combined, df_ecg_combined = __identify_best_lag(df_ppg_corrected, df_ecg_corrected, best_lag_average, corrected = False)
        best_lag_corrected, max_correct_beats_corrected, df_ppg_combined, df_ecg_combined = __identify_best_lag(df_ppg_combined, df_ecg_combined, best_lag_average, corrected = True)
    elif lag_type == 'fixed':
        lag = np.timedelta64(int(lag * 1000), 'ms')
        best_lag_detected, max_correct_beats_detected, df_ppg_detected, df_ecg_detected = __identify_best_lag(df_ppg, df_ecg, lag, corrected = False)
        best_lag_corrected, max_correct_beats_corrected, df_ppg_combined, df_ecg_combined = __identify_best_lag(df_ppg_detected, df_ecg_detected, lag, corrected = True)
    else:
        raise ValueError('Invalid lag type. Please choose from "best", "detected", "corrected", or "average".')

    n_ref = len(df_ecg[df_ecg['Peak'] == 1])
    n_corrected = len(df_ppg[df_ppg['Peak after correction'] == 1])
    n_detected = len(df_ppg[df_ppg['Peak'] == 1])

    # Sensitivity
    se_detected = round(max_correct_beats_detected / n_ref * 100, 2)
    se_corrected = round(max_correct_beats_corrected / n_ref * 100, 2)
    # Positive predictive value
    ppv_detected = round(max_correct_beats_detected / n_detected * 100, 2)
    ppv_corrected = round(max_correct_beats_corrected / n_corrected * 100, 2)
    # F1 score
    f1_detected = round((2 * ppv_detected * se_detected) / (ppv_detected + se_detected), 2)
    f1_corrected = round((2 * ppv_corrected * se_corrected) / (ppv_corrected + se_corrected), 2)

    result_df = pd.DataFrame({
        'Type': ['Detected', 'Corrected'],
        'Best lag (s)': [best_lag_detected, best_lag_corrected],
        'Max correct beats': [max_correct_beats_detected, max_correct_beats_corrected],
        'Sensitivity': [se_detected, se_corrected],
        'PPV': [ppv_detected, ppv_corrected],
        'F1': [f1_detected, f1_corrected]
    })
    result_df['Best lag (s)'] = result_df['Best lag (s)'].apply(lambda x: round(pd.to_timedelta(x).total_seconds(), 2))

    return result_df, df_ppg_combined, df_ecg_combined