import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, fixed

def plot_raw_interactive(data: pd.DataFrame, fs: int, colname: str, time_colname = 'Timestamp', peak_colname = None, seg_sec = 30) -> None:
    """
    Plot raw data interactively.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing raw data.
        This DataFrame should contain a 'Timestamp' column.
    colname : str
        Column name to plot.
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
        'fs': fixed(fs),
        'data': fixed(data_copy),
        'colname': fixed(colname),
        'time_colname': fixed(time_colname),
        'peak_colname': fixed(peak_colname)
    }
        
    interact(__update_plot_raw, **interact_args)

def __update_plot_raw(data: pd.DataFrame, fs: int, start_seg: int, range_width: int, colname: str, time_colname: str, peak_colname = None) -> None:
    """
    Update plot of raw data interactively.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing raw data.
        This DataFrame should contain a 'Timestamp' column.
    fs : int
        Sampling frequency.
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
        This DataFrame should contain a 'Timestamp' column.
    fs : int
        Sampling frequency.
    artifacts_idx : array-like
        An array-like object of indices of artifacts.
    colname : str
        Column name to plot.
    time_colname : str
        Column name of timestamp.
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
        'fs': fixed(fs),
        'artifacts_idx': fixed(artifacts_idx),
        'data': fixed(data_copy),
        'colname': fixed(colname),
        'time_colname': fixed(time_colname),
        'peak_colname': fixed(peak_colname)
    }
        
    interact(__update_plot_artifacts, **interact_args)

def __update_plot_artifacts(data: pd.DataFrame, fs: int, artifacts_idx: list, start_seg: int, range_width: int, colname: str, time_colname: str, peak_colname = 'Peak') -> None:
    """
    Update plot of artifacts interactively.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing raw data.
        This DataFrame should contain a 'Timestamp' column.
    fs : int
        Sampling frequency.
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

def plot_correction(data: pd.DataFrame, fs: int, colname: str, time_colname = 'Timestamp', peak_colname = 'Peak', peak_colname_correction = 'Peak after correction', seg_sec = 30) -> None:
    """
    Plot corrected data interactively.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing raw data.
        This DataFrame should contain a 'Timestamp' column.
    fs : int
        Sampling frequency.
    colname : str
        Column name to plot.
    time_colname : str
        Column name of timestamp.
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
        'fs': fixed(fs),
        'data': fixed(data_copy),
        'colname': fixed(colname),
        'time_colname': fixed(time_colname),
        'peak_colname': fixed(peak_colname),
        'peak_colname_correction': fixed(peak_colname_correction)
    }
        
    interact(__update_plot_correction, **interact_args)

def __update_plot_correction(data: pd.DataFrame, fs: int, start_seg: int, range_width: int, colname: str, time_colname: str, peak_colname: str, peak_colname_correction: str) -> None:
    """
    Update plot of corrected data interactively.
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing raw data.
        This DataFrame should contain a 'Timestamp' column.
    fs : int
        Sampling frequency.
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