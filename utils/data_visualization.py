import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, fixed
from datetime import datetime

def plot_raw_interactive(data: pd.DataFrame, fs: int, colname: str, time_colname = 'Timestamp', peak_colname = None, seg_sec = 30) -> None:
    """
    Plot raw data interactively.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing raw data.
    colname : str
        Column name to plot.
    time_colname : str
        Column name of timestamp. By default, 'Timestamp'.
    fs : int
        Sampling frequency.
    seg_sec : int
        Segment size in seconds.
    """
    range_width = fs * seg_sec
    data_copy = data.copy()
    data_copy[time_colname] = pd.to_datetime(data_copy[time_colname])

    interact_args = {
        'start_seg': IntSlider(min=1, max=int(np.ceil(len(data) / range_width)), step=1, value=0, continuous_update=False),
        'range_width': fixed(range_width),
        'data': fixed(data_copy),
        'colname': fixed(colname),
        'time_colname': fixed(time_colname),
        'peak_colname': fixed(peak_colname)
    }
        
    interact(__update_plot_raw, **interact_args)

def __update_plot_raw(data: pd.DataFrame, start_seg: int, range_width: int, colname: str, time_colname: str, peak_colname = None) -> None:
    """
    Update plot of raw data interactively.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing raw data.
    start_seg : int
        Start segment number.
    range_width : int
        Range of one segment in indices.
    colname : str
        Column name to plot.
    time_colname : str
        Column name of timestamp.
    peak_colname : str
        Column name of peak if any. Default is None.
    """
    start_idx = (start_seg - 1) * range_width
    end_idx = start_seg * range_width

    fig, ax = plt.subplots(figsize=(18, 2))
    data_seg = data.iloc[start_idx:end_idx]
    ax.plot(data_seg[time_colname], data_seg[colname], color = 'grey', label = colname)
    if peak_colname:
        peaks = data_seg[data_seg[peak_colname] == 1]
        ax.scatter(peaks[time_colname], peaks[colname], color = 'tomato', s = 30, zorder = 3, label = 'Detected peak')
    ax.set_title(f'{colname} - Segment {start_seg}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel(colname)
    ax.legend(loc = 'upper right')
    plt.show()

def plot_artifacts_interactive(data: pd.DataFrame, fs: int, artifacts_idx: list, colname: str, time_colname = 'Timestamp', peak_colname = 'Peak', seg_sec = 30) -> None:
    """
    Plot artifacts interactively.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing raw data.
    fs : int
        Sampling frequency.
    artifacts_idx : array-like
        An array-like object of indices of artifacts.
    colname : str
        Column name to plot.
    time_colname : str
        Column name of timestamp.  By default, 'Timestamp'.
    peak_colname : str
        Column name of peak. Default is 'Peak.'
    seg_sec : int
        Segment size in seconds.
    """
    range_width = fs * seg_sec
    data_copy = data.copy()
    data_copy[time_colname] = pd.to_datetime(data_copy[time_colname])

    interact_args = {
        'start_seg': IntSlider(min=1, max=int(np.ceil(len(data) / range_width)), step=1, value=0, continuous_update=False),
        'range_width': fixed(range_width),
        'artifacts_idx': fixed(artifacts_idx),
        'data': fixed(data_copy),
        'colname': fixed(colname),
        'time_colname': fixed(time_colname),
        'peak_colname': fixed(peak_colname)
    }
        
    interact(__update_plot_artifacts, **interact_args)

def __update_plot_artifacts(data: pd.DataFrame, artifacts_idx: list, start_seg: int, range_width: int, colname: str, time_colname: str, peak_colname = 'Peak') -> None:
    """
    Update plot of artifacts interactively.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing raw data.
    start_seg : int
        Start segment number.
    range_width : int
        Range of one segment in indices.
    colname : str
        Column name to plot.
    time_colname : str
        Column name of timestamp.
    peak_colname : str
        Column name of peak.
    """
    start_idx = (start_seg - 1) * range_width
    end_idx = start_seg * range_width

    fig, ax = plt.subplots(figsize=(18, 2))
    data_seg = data.iloc[start_idx:end_idx]
    ax.plot(data_seg[time_colname], data_seg[colname], color = 'grey', label = colname)
    peaks = data_seg[data_seg[peak_colname] == 1]
    ax.scatter(peaks[time_colname], peaks[colname], color = 'tomato', s = 30, zorder = 3, label = 'Detected peak')
    artifacts = data_seg[data_seg.index.isin(artifacts_idx)]
    if len(artifacts) > 0:
        ax.scatter(artifacts[time_colname], artifacts[colname], color = 'gold', edgecolors = 'tomato', linewidths=2, s = 30, zorder = 3, label = 'Artifact')
    ax.set_title(f'{colname} - Segment {start_seg}')
    ax.set_xlabel('Timestamp')
    ax.set_ylabel(colname)
    ax.legend(loc = 'upper right')
    plt.show()

def plot_correction_interactive(data: pd.DataFrame, fs: int, colname: str, time_colname = 'Timestamp', peak_colname = 'Peak', peak_colname_correction = 'Peak after correction', seg_sec = 30) -> None:
    """
    Plot corrected data interactively.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing raw data.
    fs : int
        Sampling frequency.
    colname : str
        Column name to plot.
    time_colname : str
        Column name of timestamp. By default, 'Timestamp'.
    peak_colname : str
        Column name of peak.
    seg_sec : int
        Segment size in seconds.
    """
    range_width = fs * seg_sec
    data_copy = data.copy()
    data_copy[time_colname] = pd.to_datetime(data_copy[time_colname])

    interact_args = {
        'start_seg': IntSlider(min=1, max=int(np.ceil(len(data) / range_width)), step=1, value=0, continuous_update=False),
        'range_width': fixed(range_width),
        'data': fixed(data_copy),
        'colname': fixed(colname),
        'time_colname': fixed(time_colname),
        'peak_colname': fixed(peak_colname),
        'peak_colname_correction': fixed(peak_colname_correction)
    }
        
    interact(__update_plot_correction, **interact_args)

def __update_plot_correction(data: pd.DataFrame, start_seg: int, range_width: int, colname: str, time_colname: str, peak_colname: str, peak_colname_correction: str) -> None:
    """
    Update plot of corrected data interactively.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing raw data.
    corrected_data : pd.DataFrame
        Dataframe containing corrected data.
    start_seg : int
        Start segment number.
    range_width : int
        Range of one segment in indices.
    colname : str
        Column name to plot.
    time_colname : str
        Column name of timestamp.
        peak_colname : str
        Column name of peak.
    peak_colname_correction : str
        Column name of peak after correction.
    """
    start_idx = (start_seg - 1) * range_width
    end_idx = start_seg * range_width

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 6))
    data_seg = data.iloc[start_idx:end_idx]
    
    for ax in axes:
        ax.plot(data_seg[time_colname], data_seg[colname], color = 'grey', label = colname)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel(colname)
    peaks = data_seg[data_seg[peak_colname] == 1]
    scatter_before_correction = axes[0].scatter(peaks[time_colname], peaks[colname], color = 'tomato', s = 30, zorder = 3, label = 'Peak before correction')
    scatter_before_correction = axes[2].scatter(peaks[time_colname], peaks[colname], color = 'tomato', s = 30, zorder = 3, label = 'Peak before correction')
    peaks_after_correction = data_seg[data_seg[peak_colname_correction] == 1]
    scatter_after_correction = axes[1].scatter(peaks_after_correction[time_colname], peaks_after_correction[colname], color = 'limegreen', s = 30, zorder = 3, label = 'Peak after correction')
    scatter_after_correction = axes[2].scatter(peaks_after_correction[time_colname], peaks_after_correction[colname], color = 'limegreen', s = 10, zorder = 4, label = 'Peak after correction')
    fig.suptitle(f'{colname} - Segment {start_seg}')
    fig.legend(handles=[scatter_before_correction, scatter_after_correction], labels=['Peak before correction', 'Peak after correction'], loc = 'upper right')
    plt.show()
    plt.tight_layout()

def plot_mims_artifacts_interactive(data: pd.DataFrame, mims: pd.DataFrame, artifacts_idx: list, fs: int, colname: str, seg_sec: int, mims_threshold: float, peak_colname = 'Peak',time_colname = 'Timestamp'):
    """
    Plot MIMS scores along with artifacts

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing raw data.
    mims : pd.DataFrame
        Dataframe containing MIMS scores.
        This DataFrame should contain a 'HEADER_TIME_STAMP' column and 'MIMS_UNIT' column.
    artifacts_ix : list
        List of indices of artifacts.
    fs : int
        Sampling frequency of raw data.
    colname : str
        Column name to plot (ECG or PPG column) in the data file.
    seg_sec : int
        Segment size in seconds.
    mims_threshold : float
        Threshold for MIMS score. MIMS scores higher than this value will be highlighted.
    peak_colname : str
        Column name of peak in the data file. By default, 'Peak'.
    time_colname : str
        Column name of timestamp in the data file. By default, 'Timestamp'.
    """
    range_width = fs * seg_sec
    data_copy = data.copy()
    data_copy[time_colname] = pd.to_datetime(data_copy[time_colname])

    interact_args = {
        'start_seg': IntSlider(min=1, max=int(np.ceil(len(data) / range_width)), step=1, value=0, continuous_update=False),
        'range_width': fixed(range_width),
        'artifacts_idx': fixed(artifacts_idx),
        'data': fixed(data_copy),
        'mims': fixed(mims),
        'colname': fixed(colname),
        'mims_threshold': fixed(mims_threshold),
        'time_colname': fixed(time_colname),
        'peak_colname': fixed(peak_colname)
    }
        
    interact(__update_plot_mims_artifacts, **interact_args)

def detect_invalid_segments_mims(signal_start_time: datetime, signal_end_time: datetime, mims: pd.DataFrame, threshold=0.5):
    """
    This function returns the intervals with large motion based on MIMS scores.

    Parameters
    ----------
    signal_start_time : datetime
        Start time of the signal. The sampling frequency of the original signals and MIMS scores can be different, so this timestamp
        will be used to determine the start time of the motion interval.
    signal_end_time : datetime
        End time of the signal. The sampling frequency of the original signals and MIMS scores can be different, so this timestamp
        will be used to determine the end time of the motion interval.
    mims : pd.DataFrame
        Dataframe containing MIMS scores.
        This DataFrame should contain a 'HEADER_TIME_STAMP' column and 'MIMS_UNIT' column.
    threshold : float
        Threshold for MIMS score. MIMS scores higher than this value will be considered as motion. Default is 0.5.
    """
    # Detect intervals with large motion
    motion_intervals = []
    in_motion = False
    start_time = None
    
    for i in range(len(mims)):
        if mims['MIMS_UNIT'].iloc[i] > threshold:
            if not in_motion:
                in_motion = True
                # If the motion starts from the beginning of the data, set the start time as the smaller timestamp between the signal start time and the mims start time
                if i == 0:
                    start_time = min(signal_start_time, mims['HEADER_TIME_STAMP'].iloc[i])
                else:
                    start_time = mims['HEADER_TIME_STAMP'].iloc[i]
        else:
            if in_motion:
                in_motion = False
                end_time = mims['HEADER_TIME_STAMP'].iloc[i]
                motion_intervals.append((start_time, end_time))
    
    # If the motion continues until the end of the data
    if in_motion:
        end_time = max(signal_end_time, mims['HEADER_TIME_STAMP'].iloc[len(mims) - 1])
        motion_intervals.append((start_time, end_time))
    
    return motion_intervals if motion_intervals else None

def __update_plot_mims_artifacts(data: pd.DataFrame, mims: pd.DataFrame, artifacts_idx: list, start_seg: int, range_width: int, colname: str, time_colname: str, mims_threshold: float, peak_colname = 'Peak') -> None:
    """
    Update plot of artifacts interactively.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing raw data.
    mims : pd.DataFrame
        Dataframe containing MIMS scores.
        This DataFrame should contain a 'HEADER_TIME_STAMP' column and 'MIMS_UNIT' column.
    start_seg : int
        Start segment number.
    range_width : int
        Range of one segment in indices.
    colname : str
        Column name to plot.
    time_colname : str
        Column name of timestamp.
    peak_colname : str
        Column name of peak. By default, 'Peak'.
    mims_threshold : float
        Threshold for MIMS score. MIMS scores higher than this value will be highlighted. Default is None.
    """
    start_idx = (start_seg - 1) * range_width
    end_idx = min(start_seg * range_width, len(data)-1)

    start_time = data[time_colname].iloc[start_idx]
    end_time = data[time_colname].iloc[end_idx]

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 4))
    axes = axes.flatten()
    data_seg = data.iloc[start_idx:end_idx]
    if end_idx < len(data):
        mims_seg = mims.loc[(mims['HEADER_TIME_STAMP'] >= start_time) & (mims['HEADER_TIME_STAMP'] < end_time)]
    else:
        mims_seg = mims.loc[(mims['HEADER_TIME_STAMP'] >= start_time) & (mims['HEADER_TIME_STAMP'] <= end_time)]
    
    axes[0].plot(data_seg[time_colname], data_seg[colname], color = 'grey', label = colname)
    peaks = data_seg[data_seg[peak_colname] == 1]
    axes[0].scatter(peaks[time_colname], peaks[colname], color = 'tomato', s = 30, zorder = 3, label = 'Detected peak')
    artifacts = data_seg[data_seg.index.isin(artifacts_idx)]
    if len(artifacts) > 0:
        axes[0].scatter(artifacts[time_colname], artifacts[colname], color = 'gold', edgecolors = 'tomato', linewidths=2, s = 30, zorder = 3, label = 'Artifact')
    axes[0].set_title(f'{colname} - Segment {start_seg}')
    axes[0].set_xlabel('Timestamp')
    axes[0].set_ylabel(colname)
    axes[0].legend(loc = 'upper right')
    axes[1].plot(mims_seg['HEADER_TIME_STAMP'], mims_seg['MIMS_UNIT'], color = 'deepskyblue', label = 'MIMS score')
    axes[1].set_xlabel('Timestamp')
    axes[1].set_ylabel('MIMS score')
    axes[1].set_title(f'MIMS score (aceleration) - Segment {start_seg}')

    motion_intervals = detect_invalid_segments_mims(signal_start_time=start_time, signal_end_time=end_time, mims=mims_seg, threshold=mims_threshold)
    if motion_intervals:
        for motion_interval in motion_intervals:
            axes[0].axvspan(motion_interval[0], motion_interval[1], color='gold', alpha=0.3, zorder=0, label='Motion interval')
            axes[1].axvspan(motion_interval[0], motion_interval[1], color='gold', alpha=0.3, zorder=0, label='Motion interval')

    plt.tight_layout()
    plt.show()

def plot_mims_correction_interactive(data: pd.DataFrame, mims: pd.DataFrame, fs: int, colname: str, mims_threshold: float, time_colname = 'Timestamp', peak_colname = 'Peak', peak_colname_correction = 'Peak after correction', seg_sec = 30) -> None:
    """
    Plot corrected data interactively.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing raw data.
    mims : pd.DataFrame
        Dataframe containing MIMS scores.
        This DataFrame should contain a 'HEADER_TIME_STAMP' column and 'MIMS_UNIT' column.
    fs : int
        Sampling frequency.
    colname : str
        Column name to plot in data file.
    mims_threshold : float
        Threshold for MIMS score. MIMS scores higher than this value will be highlighted
    time_colname : str
        Column name of timestamp in data file. By default, 'Timestamp'.
    peak_colname : str
        Column name of peak in data file. By default, 'Peak'.
    seg_sec : int
        Segment size in seconds.
    """
    range_width = fs * seg_sec
    data_copy = data.copy()
    data_copy[time_colname] = pd.to_datetime(data_copy[time_colname])

    interact_args = {
        'start_seg': IntSlider(min=1, max=int(np.ceil(len(data) / range_width)), step=1, value=0, continuous_update=False),
        'range_width': fixed(range_width),
        'data': fixed(data_copy),
        'mims': fixed(mims),
        'colname': fixed(colname),
        'mims_threshold': fixed(mims_threshold),
        'time_colname': fixed(time_colname),
        'peak_colname': fixed(peak_colname),
        'peak_colname_correction': fixed(peak_colname_correction)
    }
        
    interact(__update_plot_mims_correction, **interact_args)

def __update_plot_mims_correction(data: pd.DataFrame, mims: pd.DataFrame, start_seg: int, range_width: int, colname: str, mims_threshold: float, time_colname: str, peak_colname: str, peak_colname_correction: str) -> None:
    """
    Update plot of corrected data interactively.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing raw data.
    mims : pd.DataFrame
        Dataframe containing MIMS scores. MIMS scores higher than this value will be highlighted.
    corrected_data : pd.DataFrame
        Dataframe containing corrected data.
    start_seg : int
        Start segment number.
    range_width : int
        Range of one segment in indices.
    colname : str
        Column name to plot.
    mims_threshold : float
        Threshold for MIMS score. MIMS scores higher than this value will be highlighted
    time_colname : str
        Column name of timestamp.
        peak_colname : str
        Column name of peak.
    peak_colname_correction : str
        Column name of peak after correction.
    """
    start_idx = (start_seg - 1) * range_width
    end_idx = start_seg * range_width

    start_time = data[time_colname].iloc[start_idx]
    end_time = data[time_colname].iloc[end_idx]

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(15, 8))
    data_seg = data.iloc[start_idx:end_idx]
    if end_idx < len(data):
        mims_seg = mims.loc[(mims['HEADER_TIME_STAMP'] >= start_time) & (mims['HEADER_TIME_STAMP'] < end_time)]
    else:
        mims_seg = mims.loc[(mims['HEADER_TIME_STAMP'] >= start_time) & (mims['HEADER_TIME_STAMP'] <= end_time)]

    for ax in axes[0:3]:
        ax.plot(data_seg[time_colname], data_seg[colname], color = 'grey', label = colname)
        ax.set_xlabel('Timestamp')
        ax.set_ylabel(colname)
    peaks = data_seg[data_seg[peak_colname] == 1]
    scatter_before_correction = axes[0].scatter(peaks[time_colname], peaks[colname], color = 'tomato', s = 30, zorder = 3, label = 'Peak before correction')
    scatter_before_correction = axes[2].scatter(peaks[time_colname], peaks[colname], color = 'tomato', s = 30, zorder = 3, label = 'Peak before correction')
    peaks_after_correction = data_seg[data_seg[peak_colname_correction] == 1]
    scatter_after_correction = axes[1].scatter(peaks_after_correction[time_colname], peaks_after_correction[colname], color = 'limegreen', s = 30, zorder = 3, label = 'Peak after correction')
    scatter_after_correction = axes[2].scatter(peaks_after_correction[time_colname], peaks_after_correction[colname], color = 'limegreen', s = 10, zorder = 4, label = 'Peak after correction')
    axes[3].plot(mims_seg['HEADER_TIME_STAMP'], mims_seg['MIMS_UNIT'], color = 'deepskyblue', label = 'MIMS score')
    axes[3].set_xlabel('Timestamp')
    axes[3].set_ylabel('MIMS score')

    motion_intervals = detect_invalid_segments_mims(signal_start_time=start_time, signal_end_time=end_time, mims=mims_seg, threshold=mims_threshold)
    if motion_intervals:
        for motion_interval in motion_intervals:
            for ax in axes:
                ax.axvspan(motion_interval[0], motion_interval[1], color='gold', alpha=0.3, zorder=0, label='Motion interval')

    fig.suptitle(f'{colname} - Segment {start_seg}')
    fig.legend(handles=[scatter_before_correction, scatter_after_correction], labels=['Peak before correction', 'Peak after correction'], loc = 'upper right')
    plt.show()
    plt.tight_layout()

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

def add_lag_to_motion_intervals(motion_intervals, lag):
        '''
        A function to add lag to the detected motion intervals.
        Lag for detected signals and corrected signals may differ, so this function is used to add the lag to the motion intervals.

        Parameters
        ------------
        motion_intervals: List of tuples
            A list of tuples of the intervals with large motion (start timestamp, end timestamp)
        lag: np.timedelta64
            Lag to add to the motion intervals.
        
        Returns
        ------------
        motion_intervals_lagged: List of tuples
            A list of tuples of the intervals with large motion (start timestamp, end timestamp) with lag added.
        '''
        motion_intervals_lagged = []
        for start, end in motion_intervals:
            motion_intervals_lagged.append((start + lag, end + lag))
        return motion_intervals_lagged

def __update_plot_correct(start_seg, df_ppg, df_ecg, fs_ecg, seg_size, plot_acc = False, plot_mims = False, mims = None, mims_threshold = 0.5):
    '''
    A function to update a plot of correct beats.
    This function is called in plot_correct function.

    Parameters
    ------------
    start_seg: int
        Segment number to start plotting.
    df_ppg: pandas.DataFrame
        PPG data.
    df_ecg: pandas.DataFrame
        ECG data.
    fs_ecg: int
        Sampling frequency of the ECG data.
    seg_size: int
        Segment size in seconds.
    plot_acc: bool
        If True, plot the raw acceleration data.
    plot_mims: bool
        If True, plot the MIMS data.
    mims: pandas.DataFrame
        If plot_mims is True, MIMS data should be provided.
        This dataframe should contain 'Timestamp' and 'MIMS_UNIT' columns.
    mims_threshold: float
        Threshold for detecting invalid segments in MIMS data.
    '''
    start_index = int(round((start_seg - 1) * seg_size * fs_ecg, 0))
    range_width = int(seg_size * fs_ecg)
    df_ref = df_ecg
    df = df_ppg

    start_time = df_ref['Timestamp'][start_index]
    if start_index + range_width >= len(df_ref):
        end_time = df_ref['Timestamp'][len(df_ref) - 1]
    else:
        end_time = df_ref['Timestamp'][start_index + range_width]
    df_ref_trimmed = df_ref[(df_ref['Timestamp'] >= start_time) & (df_ref['Timestamp'] < end_time)]
    df_trimmed_corrected = df[(df['Lagged timestamp after correction'] >= start_time) & (df['Lagged timestamp after correction'] < end_time)]
    df_trimmed_detected = df[(df['Lagged timestamp'] >= start_time) & (df['Lagged timestamp'] < end_time)]
        
    if plot_acc and plot_mims:
        fig, axes = plt.subplots(5, 1, figsize = (18, 8))
    elif plot_acc or plot_mims:
        fig, axes = plt.subplots(4, 1, figsize = (18, 6))
    else:
        fig, axes = plt.subplots(3, 1, figsize = (18, 4))

    axes[0].plot(df_ref_trimmed['Timestamp'], df_ref_trimmed['Filtered'], color = 'lightgrey')
    ax_twin_0 = axes[0].twinx()
    ax_twin_0.plot(df_trimmed_detected['Lagged timestamp'], df_trimmed_detected['Filtered'], color = 'grey')
    reference_scatter = axes[0].scatter(df_ref_trimmed.loc[df_ref_trimmed['Peak'] == 1]['Timestamp'], 
                df_ref_trimmed.loc[(df_ref_trimmed['Peak'] == 1)]['Filtered'], color = 'lightgrey', s = 40, zorder = 3, label = 'Reference')
    correct_scatter = axes[0].scatter(df_ref_trimmed.loc[df_ref['Correct peak'] == 1]['Timestamp'], 
                df_ref_trimmed.loc[df_ref['Correct peak'] == 1]['Filtered'], color = 'khaki', s = 15, zorder = 4, label = 'Correct')
    detected_scatter = ax_twin_0.scatter(df_trimmed_detected.loc[df_trimmed_detected['Peak'] == 1]['Lagged timestamp'], 
                df_trimmed_detected.loc[df_trimmed_detected['Peak'] == 1]['Filtered'], color = 'lightcoral', s = 40, zorder = 3, label = 'Detected')
    correct_scatter = ax_twin_0.scatter(df_trimmed_detected.loc[df_trimmed_detected['Correct peak'] == 1]['Lagged timestamp'], 
                df_trimmed_detected.loc[df_trimmed_detected['Correct peak'] == 1]['Filtered'], color = 'gold', s = 15, zorder = 4, label = 'Correct')
    
    axes[1].plot(df_ref_trimmed['Timestamp'], df_ref_trimmed['Filtered'], color = 'lightgrey')
    ax_twin_1 = axes[1].twinx()
    ax_twin_1.plot(df_trimmed_corrected['Lagged timestamp after correction'], df_trimmed_corrected['Filtered'], color = 'grey')
    reference_scatter = axes[1].scatter(df_ref_trimmed.loc[df_ref_trimmed['Peak'] == 1]['Timestamp'], 
                df_ref_trimmed.loc[(df_ref_trimmed['Peak'] == 1)]['Filtered'], color = 'lightgrey', s = 40, zorder = 3, label = 'Reference')
    correct_scatter = axes[1].scatter(df_ref_trimmed.loc[df_ref['Correct peak after correction'] == 1]['Timestamp'], 
                df_ref_trimmed.loc[df_ref['Correct peak after correction'] == 1]['Filtered'], color = 'khaki', s = 15, zorder = 4, label = 'Correct')
    corrected_scatter = ax_twin_1.scatter(df_trimmed_corrected.loc[df_trimmed_corrected['Peak after correction'] == 1]['Lagged timestamp after correction'], 
                df_trimmed_corrected.loc[df_trimmed_corrected['Peak after correction'] == 1]['Filtered'], color = 'limegreen', s = 40, zorder = 3, label = 'Corrected')
    correct_scatter = ax_twin_1.scatter(df_trimmed_corrected.loc[df_trimmed_corrected['Correct peak after correction'] == 1]['Lagged timestamp after correction'], 
                df_trimmed_corrected.loc[df_trimmed_corrected['Correct peak after correction'] == 1]['Filtered'], color = 'gold', s = 15, zorder = 4, label = 'Correct')
    
    axes[2].plot(df_ref_trimmed['Timestamp'], df_ref_trimmed['Filtered'], color = 'lightgrey')
    ax_twin_2 = axes[2].twinx()
    ax_twin_2.plot(df_trimmed_corrected['Lagged timestamp'], df_trimmed_corrected['Filtered'], color = 'grey')
    reference_scatter = axes[2].scatter(df_ref_trimmed.loc[df_ref_trimmed['Peak'] == 1]['Timestamp'], 
                df_ref_trimmed.loc[(df_ref_trimmed['Peak'] == 1)]['Filtered'], color = 'lightgrey', s = 40, zorder = 3, label = 'Reference')
    detected_scatter = ax_twin_2.scatter(df_trimmed_detected.loc[df_trimmed_detected['Peak'] == 1]['Lagged timestamp'], 
                df_trimmed_detected.loc[df_trimmed_detected['Peak'] == 1]['Filtered'], color = 'lightcoral', s = 40, zorder = 3, label = 'Detected')
    corrected_scatter = ax_twin_2.scatter(df_trimmed_corrected.loc[df_trimmed_corrected['Peak after correction'] == 1]['Lagged timestamp after correction'], 
                df_trimmed_corrected.loc[df_trimmed_corrected['Peak after correction'] == 1]['Filtered'], color = 'limegreen', s = 15, zorder = 4, label = 'Corrected')

    if plot_acc:
        df_trimmed_acc = df_trimmed_corrected.copy()[['Timestamp', 'ACC_x', 'ACC_y', 'ACC_z']].dropna()
        axes[3].plot(df_trimmed_acc['Timestamp'], df_trimmed_acc['ACC_x'])
        axes[3].plot(df_trimmed_acc['Timestamp'], df_trimmed_acc['ACC_y'])
        axes[3].plot(df_trimmed_acc['Timestamp'], df_trimmed_acc['ACC_z'])
        axes[3].legend(['ACC_x', 'ACC_y', 'ACC_z'])
        #if motion_intervals:
            #for start, end in motion_intervals:
                #axes[2].axvspan(start, end, color='red', alpha=0.3)
    
    if plot_mims:
        if mims.empty:
            raise ValueError('MIMS data is not provided.')
        lag_detected = df['Lagged timestamp'].iloc[0] - df['Timestamp'].iloc[0]
        lag_corrected = df['Lagged timestamp after correction'].iloc[0] - df['Timestamp'].iloc[0]
        invalid_segments = detect_invalid_segments_mims(signal_start_time=df_ppg['Timestamp'].iloc[0], signal_end_time=df_ppg['Timestamp'].iloc[-1], mims = mims, threshold = mims_threshold)
        if plot_acc:
            ax_idx = 4
        else:
            ax_idx = 3
        try:
            mims_lagged = add_lag(df = mims, lag = lag_corrected, corrected=True)
        except KeyError:
            mims_renamed = mims.rename(columns = {'HEADER_TIME_STAMP': 'Timestamp'})
            mims_lagged = add_lag(df = mims_renamed, lag = lag_corrected, corrected=True)
        axes[ax_idx].plot(mims_lagged['Lagged timestamp after correction'], mims['MIMS_UNIT'], color = 'deepskyblue')

        if invalid_segments:
            # Add the lag to the motion intervals
            if lag_detected != lag_corrected:
                invalid_segments_lagged_detected = add_lag_to_motion_intervals(motion_intervals=invalid_segments, lag=lag_detected)
                invalid_segments_lagged_corrected = add_lag_to_motion_intervals(motion_intervals=invalid_segments, lag=lag_corrected)
            else:
                invalid_segments_lagged_detected = add_lag_to_motion_intervals(motion_intervals=invalid_segments, lag=lag_detected)
                invalid_segments_lagged_corrected = invalid_segments_lagged_detected
            
            for start, end in invalid_segments_lagged_corrected:
                axes[ax_idx].axvspan(start, end, color='gold', alpha=0.3)
                axes[1].axvspan(start, end, color='gold', alpha=0.3)
                if plot_acc:
                    axes[3].axvspan(start, end, color='gold', alpha=0.3)
            for start, end in invalid_segments_lagged_detected:
                axes[0].axvspan(start, end, color='gold', alpha=0.3)
        axes[ax_idx].legend(['MIMS'])

    for ax in axes:
        ax.set_xlim([start_time, end_time])
    
    fig.legend(handles=[reference_scatter, corrected_scatter, detected_scatter, correct_scatter], labels=['Ground truth beats', 'Beats after correction', 'Beats before correction', 'Agreement w/ ECG'], loc='right')
    fig.suptitle(f'Segment {start_seg}')
    
def plot_correct(df_ppg, df_ecg, fs_ppg, fs_ecg, plot_acc = False, plot_mims = False, mims = None, mims_threshold = 0.5, seg_size=30):
    '''
    A function to interactively plot segment-by-segment signals and correct beats.
    This figure has three plots: reference signal/beats, corrected beats, and the comparison of the reference and corrected signal/beats.

    Parameters
    ------------
    df_ppg: pandas.DataFrame
        PPG data (each column should be in the format of 'Timestamp', 'Filtered', 'Peak', 'Correct peak', 'Lagged timestamp', 'Peak after correction', 'Correct peak after correction', 'Lagged timestamp after correction')
    df_ecg: pandas.DataFrame
        ECG data (each column should be in the format of 'Timestamp', 'Filtered', 'Peak', 'Correct peak', 'Correct peak after correction')
    fs_ecg: int
        Sampling frequency of the ECG data.
    plot_acc: bool
        If True, plot the raw acceleration data.
    plot_mims: bool
        If True, plot the MIMS data.
    mims: pandas.DataFrame
        If plot_mims is True, MIMS data should be provided.
        This dataframe should contain 'Timestamp' and 'MIMS_UNIT' columns.
    mims_threshold: float
        MIMS scores higher than this value will be highlighted.
    '''
    if mims is None:
        mims = pd.DataFrame()

    interact_args = {
        'start_seg': IntSlider(min=1, max=int(np.ceil(len(df_ecg) / (seg_size * fs_ecg))), step=1, value=0, continuous_update=False),
        'df_ppg': fixed(df_ppg),
        'df_ecg': fixed(df_ecg),
        'fs_ecg': fixed(fs_ecg),
        'plot_acc': fixed(plot_acc),
        'plot_mims': fixed(plot_mims),
        'mims': fixed(mims),
        'mims_threshold': fixed(mims_threshold),
        'seg_size': fixed(seg_size)
    }

    interact(__update_plot_correct, **interact_args)