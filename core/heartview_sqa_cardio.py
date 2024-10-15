import pandas as pd
import numpy as np
from tqdm import tqdm
from math import ceil, floor
import pdb

DEBUGGING = False

if DEBUGGING == True:
    import sys
    sys.setrecursionlimit(10000)

# ============================== CARDIOVASCULAR ==============================
class Cardio:
    """
    A class for signal quality assessment on cardiovascular data, including
    electrocardiograph (ECG) or photoplethysmograph (PPG) data.

    Parameters/Attributes
    ---------------------
    fs : int
        The sampling rate of the cardiovascular data.
    """

    def __init__(self, fs):
        """
        Initialize the Cardiovascular object.

        Parameters
        ----------
        fs : int
            The sampling rate of the ECG or PPG recording.
        """
        self.fs = int(fs)

    def compute_metrics(self, data, beats_ix, ts_col = None, seg_size = 60,
                        rolling_window = None, rolling_step = 15,
                        show_progress = True):
        """
        Compute all SQA metrics for cardiovascular data by segment or
        moving window. Metrics per segment or moving window include numbers
        of detected, expected, missing, and artifactual beats and
        percentages of missing and artifactual beats.

        Parameters
        ----------
        data : pandas.DataFrame
            A data frame containing pre-processed ECG or PPG data.
        beats_ix : array_like
            An array containing the indices of detected beats.
        seg_size : int
            The segment size in seconds; by default, 60.
        ts_col : str, optional
            The name of the column containing timestamps; by default, None.
            If a string value is given, the output will contain a timestamps
            column.
        rolling_window : int, optional
            The size, in seconds, of the sliding window across which to
            compute the SQA metrics; by default, None.
        rolling_step : int, optional
            The step size, in seconds, of the sliding windows; by default, 15.
        show_progress : bool, optional
            Whether to display a progress bar while the function runs; by
            default, True.

        Returns
        -------
        metrics : pandas.DataFrame
            A data frame with all computed SQA metrics per segment.

        Notes
        -----
        If a value is given in the `rolling_window` parameter, the rolling
        window approach will override the segmented approach, ignoring any
        `seg_size` value.
        """
        df = data.copy()
        df.index = df.index.astype(int)
        df.loc[beats_ix, 'Beat'] = 1
        if rolling_window is not None:
            metrics = pd.DataFrame()
            if ts_col is not None:
                seconds = self.get_seconds(
                    data, beats_ix, ts_col, show_progress)
                s = 1
                for n in range(0, len(seconds), rolling_step):

                    # Get missing beats
                    window_missing = seconds.iloc[n:(n + rolling_window)]
                    n_expected = window_missing['Mean HR'].median()
                    n_detected = window_missing['N Beats'].sum()
                    n_missing = (n_expected - n_detected) \
                        if n_expected > n_detected else 0
                    perc_missing = round((n_missing / n_detected) * 100, 2)
                    ts = window_missing['Timestamp'].iloc[0]

                    # Get artifactual beats
                    artifacts = self.get_artifacts(
                        df, beats_ix, seg_size = 1, ts_col = ts_col)
                    window_artifact = artifacts.iloc[n:(n + rolling_window)]
                    n_artifact = window_artifact['N Artifact'].sum()
                    perc_artifact = round((n_artifact / n_detected) * 100, 2)

                    # Output summary
                    metrics = pd.concat([metrics, pd.DataFrame.from_records([{
                        'Moving Window': s,
                        'Timestamp': ts,
                        'N Expected': n_expected,
                        'N Detected': n_detected,
                        'N Missing': n_missing,
                        '% Missing': perc_missing,
                        'N Artifact': n_artifact,
                        '% Artifact': perc_artifact
                    }])], ignore_index = True).reset_index(drop = True)
                    s += 1
            else:
                seconds = self.get_seconds(data, beats_ix, show_progress)
                s = 1
                for n in range(0, len(seconds), rolling_step):

                    # Get missing beats
                    window_missing = seconds.iloc[n:(n + rolling_window)]
                    n_expected = window_missing['Mean HR'].median()
                    n_detected = window_missing['N Beats'].sum()
                    n_missing = (n_expected - n_detected) \
                        if n_expected > n_detected else 0
                    perc_missing = round((n_missing / n_detected) * 100, 2)

                    # Get artifactual beats
                    artifacts = self.get_artifacts(df, beats_ix, seg_size = 1)
                    window_artifact = artifacts.iloc[n:(n + rolling_window)]
                    n_artifact = window_artifact['N Artifact'].sum()
                    perc_artifact = round((n_artifact / n_detected) * 100, 2)

                    # Output summary
                    metrics = pd.concat([metrics, pd.DataFrame.from_records([{
                        'Moving Window': s,
                        'N Expected': n_expected,
                        'N Detected': n_detected,
                        'N Missing': n_missing,
                        '% Missing': perc_missing,
                        'N Artifact': n_artifact,
                        '% Artifact': perc_artifact
                    }])], ignore_index = True).reset_index(drop = True)
                    s += 1
        else:
            if ts_col is not None:
                missing = self.get_missing(
                    df, beats_ix, seg_size, min_hr = 40, ts_col = ts_col,
                    show_progress = show_progress)
                artifacts = self.get_artifacts(
                    df, beats_ix, seg_size, ts_col)
                metrics = pd.merge(missing, artifacts, on = ['Segment', 'Timestamp'])
            else:
                missing = self.get_missing(
                    df, beats_ix, seg_size, show_progress = show_progress)
                artifacts = self.get_artifacts(df, beats_ix, seg_size)
                metrics = pd.merge(missing, artifacts, on = ['Segment'])
        return metrics

    def get_artifacts(self, data, beats_ix, seg_size = 60, ts_col = None):
        """
        Identify all artifactual beats with the Hegarty-Craver et al. (2018)
        and criterion beat difference (Berntson et al., 1990) methods.

        Parameters
        ----------
        data : pandas.DataFrame
            A data frame containing the pre-processed ECG or PPG data.
        beats_ix : array_like
            An array containing the indices of detected beats.
        seg_size : int
            The size of the segment in seconds; by default, 60.
        ts_col : str, optional
            The name of the column containing timestamps; by default, None.
            If a string value is given, the output will contain a timestamps
            column.

        Returns
        -------
        artifacts : pandas.DataFrame
            A data frame with the number and proportion of artifactual beats
            per segment.
        """
        df = data.copy()
        valid, hegarty = self.identify_artifacts_hegarty(beats_ix)
        cbd = self.identify_artifacts_cbd(beats_ix)
        artif = np.union1d(hegarty, cbd)
        df.loc[beats_ix, 'Beat'] = 1
        df.loc[artif, 'Artifact'] = 1

        n_seg = ceil(len(df) / (self.fs * seg_size))
        segments = pd.Series(np.arange(1, n_seg + 1))
        n_detected = df.groupby(
            df.index // (self.fs * seg_size))['Beat'].sum()
        n_artifact = df.groupby(
            df.index // (self.fs * seg_size))['Artifact'].sum()
        perc_artifact = round((n_artifact / n_detected) * 100, 2)

        if ts_col is not None:
            timestamps = df.groupby(
                df.index // (self.fs * seg_size)).first()[ts_col]
            artifacts = pd.concat([
                segments,
                timestamps,
                n_artifact,
                perc_artifact,
            ], axis = 1)
            artifacts.columns = [
                'Segment',
                'Timestamp',
                'N Artifact',
                '% Artifact',
            ]
        else:
            artifacts = pd.concat([
                segments,
                n_artifact,
                perc_artifact,
            ], axis = 1)
            artifacts.columns = [
                'Segment',
                'N Artifact',
                '% Artifact',
            ]
        return artifacts

    def identify_artifacts_hegarty(self, beats_ix, initial_hr = 'auto', prev_n = 6):
        """
        Identify locations of artifactual beats in cardiovascular data based
        on the approach by Hegarty-Craver et al. (2018).

        Parameters
        ----------
        beats_ix : array_like
            An array containing the indices of detected beats.
        initial_hr : int, float, optional
            The heart rate value for the first interbeat interval (IBI) to be
            validated against; by default, 80 bpm (750 ms).
        prev_n : int, optional
            The number of preceding IBIs to validate against; by default, 6.

        Returns
        -------
        tuple
            A tuple containing indices of validated beats:

            valid_beats : array_like
                An array containing the indices of valid beats.
            invalid_beats : array_like
                An array containing the indices of invalid beats.

        References
        ----------
        Hegarty-Craver, M. et al. (2018). Automated respiratory sinus
        arrhythmia measurement: Demonstration using executive function
        assessment. Behavioral Research Methods, 50, 1816–1823.
        """

        ibis = (np.diff(beats_ix) / self.fs) * 1000
        beats = beats_ix[1:]  # drop the first beat
        artifact_beats = []
        valid_beats = [beats_ix[0]]  # assume first beat is valid

        # Set the initial IBI to compare against
        if initial_hr == 'auto':
            successive_diff = np.abs(np.diff(ibis))
            min_diff_ix = np.convolve(successive_diff, np.ones(6) / 6, mode = 'valid').argmin()
            first_ibi = ibis[min_diff_ix:min_diff_ix + 6].mean()
            #print('Estimated average HR (bpm): ', floor(60 / (first_ibi / self.fs)))
        else:
            first_ibi = self.fs * 60 / initial_hr

        for n in range(len(ibis)):
            current_ibi = ibis[n]
            current_beat = beats[n]

            # Check against an estimate of the first N IBIs
            if n < prev_n:
                if n == 0:
                    ibi_estimate = first_ibi
                else:
                    next_five = np.insert(ibis[:n], 0, first_ibi)
                    ibi_estimate = np.median(next_five)

            # Check against an estimate of the preceding N IBIs
            else:
                ibi_estimate = np.median(ibis[n - (prev_n):n])

            # Set the acceptable/valid range of IBIs
            low = (26 / 32) * ibi_estimate
            high = (44 / 32) * ibi_estimate

            if low <= current_ibi <= high:
                valid_beats.append(current_beat)
            else:
                artifact_beats.append(current_beat)

        return valid_beats, artifact_beats

    def identify_artifacts_cbd(self, beats_ix, neighbors = 5, tol = 1):
        """
        Identify locations of abnormal interbeat intervals (IBIs) using the
        criterion beat difference test by Berntson et al. (1990).

        Parameters
        ----------
        beats_ix : array_like
            An array containing the indices of detected beats.
        neighbors : int
            The number of surrounding IBIs with which to derive the criterion
            beat difference score; by default, 5.
        tol : float
            A configurable hyperparameter used to fine-tune the stringency of
            the criterion beat difference test; by default, 1. Increasing this
            value makes the test more lenient.

        Returns
        -------
        bad_ibis_ix : numpy.array
            A NumPy array containing the indices of beats that fail the
            criterion beat difference test.

        Notes
        -----
        The original source code is from work by Hoemann et al. (2020).

        References
        ----------
        Berntson, G., Quigley, K., Jang, J., Boysen, S. (1990). An approach to
        artifact identification: Application to heart period data.
        Psychophysiology, 27(5), 586–598.

        Hoemann, K. et al. (2020). Context-aware experience sampling reveals
        the scale of variation in affective experience. Scientific Reports,
        10(1), 1–16.
        """

        # Derive IBIs from beat indices
        ibis = ((np.ediff1d(beats_ix)) / self.fs) * 1000

        # Compute consecutive absolute differences across IBIs
        ibi_diffs = np.abs(np.ediff1d(ibis))

        # Initialize an array to store "bad" IBIs
        ibi_bad = np.zeros(shape = len(ibis))
        artifacts_ix = []

        if len(ibi_diffs) < neighbors:
            neighbors = len(ibis)

        for ii in range(len(ibi_diffs)):

            # If there are not enough neighbors in the beginning
            if ii < int(neighbors / 2) + 1:
                select = np.concatenate(
                    (ibi_diffs[:ii], ibi_diffs[(ii + 1):(neighbors + 1)]))
                select_ibi = np.concatenate(
                    (ibis[:ii], ibis[(ii + 1):(neighbors + 1)]))

            # If there are not enough neighbors at the end
            elif (len(ibi_diffs) - ii) < (int(neighbors / 2) + 1) and (
                    len(ibi_diffs) - ii) > 1:
                select = np.concatenate(
                    (ibi_diffs[-(neighbors - 1):ii], ibi_diffs[ii + 1:]))
                select_ibi = np.concatenate(
                    (ibis[-(neighbors - 1):ii], ibis[ii + 1:]))

            # If there is only one neighbor left to check against
            elif len(ibi_diffs) - ii == 1:
                select = ibi_diffs[-(neighbors - 1):-1]
                select_ibi = ibis[-(neighbors - 1):-1]

            else:
                select = np.concatenate(
                    (ibi_diffs[ii - int(neighbors / 2):ii],
                    ibi_diffs[(ii + 1):(ii + 1 + int(neighbors / 2))]))
                select_ibi = np.concatenate(
                    (ibis[ii - int(neighbors / 2):ii],
                    ibis[(ii + 1):(ii + 1 + int(neighbors / 2))]))

            # Calculate the quartile deviation
            QD = self._quartile_deviation(select)

            # Calculate the maximum expected difference (MED)
            MED = 3.32 * QD

            # Calculate the minimal artifact difference (MAD)
            MAD = (np.median(select_ibi) - 2.9 * QD) / 3

            # Calculate the criterion beat difference score
            criterion_beat_diff = (MED + MAD) / 2

            # Find indices of IBIs that fail the CBD check
            if (ibi_diffs[ii]) > tol * criterion_beat_diff:

                bad_neighbors = int(neighbors * 0.25)
                if ii + (bad_neighbors - 1) < len(beats_ix):
                    artifacts_ix.append(beats_ix[ii:(ii + bad_neighbors)])
                else:
                    artifacts_ix.append(beats_ix[ii:(ii + (bad_neighbors - 1))])
                ibi_bad[ii + 1] = 1

        artifacts_ix = np.array(artifacts_ix).flatten()
        return artifacts_ix

    def correct_interval(self, beats_ix, seg_size = 60, initial_hr = 'auto', prev_n = 6, min_bpm = 40, max_bpm = 200, 
                            hr_estimate_window = 6, print_estimated_hr = True, short_threshold = (24 / 32),  long_threshold = (44 / 32), extra_threshold = (52 / 32), adaptive_threshold = False, ibi_data = None, window_num = 6, mims = None, mims_threshold = 0.2, ignore_motion_seg = False, motion_per_threshold = 0.2):
        '''
        Correct artifactual beats in cardiovascular data based
        on the approach by Hegarty-Craver et al. (2018).

        Parameters
        ----------
        beats_ix : array_like
            An array containing the indices of detected beats.
        initial_hr : int, float, optional
            The heart rate value for the first interbeat interval (IBI) to be
            validated against; by default, automatically set ('auto').
        prev_n : int, optional
            The number of preceding IBIs to validate against; by default, 6.
        min_bpm : int, optional
            The minimum possible heart rate in beats per minute (bpm); by default, 40.
        max_bpm : int, optional
            The maximum possible heart rate in beats per minute (bpm); by default, 200.
        hr_estimate_window : int, optional
            The window size for estimating the heart rate; by default, 6.
        print_estimated_hr : bool, optional
            Whether to print the estimated heart rate; by default, True.
        short_threshold : float, optional
            The threshold for short IBIs; by default, 24/32.
        long_threshold : float, optional
            The threshold for long IBIs; by default, 44/32.
        extra_threshold : float, optional
            The threshold for extra long IBIs; by default, 52/32.
        adaptive_threshold : bool, optional
            Whether to use adaptive thresholds for IBI correction; by default, False.
        ibi_data : pandas.DataFrame, optional
            A DataFrame containing timestamps and IBIs in milliseconds; by default, None.
            This DataFrame should contain 'Timestamp' and 'IBI' columns.
        window_num : int, optional
            The window size (number of previous clean IBIs) used for calculating the adaptive thresholds; by default, 6.
        mims : pandas.DataFrame, optional
            If adaptive_threshold is True, use this mims DataFrame to detect motion artifacts.
            This DataFrame should contain 'Timestamp' and 'MIMS_UNIT' columns.
        mims_threshold : float, optional
            If adaptive_threshold is True, the threshold for detecting motion artifacts; by default, 0.2.
        ignore_motion_seg : bool, optional
            Whether to ignore segments with large motion artifacts; by default, False.
        motion_per_threshold : float, optional
            If ignore_motion_seg is True, the threshold for considering a segment as unusable; by default, 0.2.
            0.2 means that if 20% of the segment is detected as motion artifact, the segment is considered uncorrectable.

        Returns
        -------
        original: pandas.DataFrame
            A data frame containing the original IBIs (millisecond-based and index-based) and beat indices.
        corrected: pandas.DataFrame
            A data frame containing the corrected IBIs (millisecond-based and index-based) and beat indices.

        References
        ----------
        Hegarty-Craver, M. et al. (2018). Automated respiratory sinus
        arrhythmia measurement: Demonstration using executive function
        assessment. Behavioral Research Methods, 50, 1816–1823.
        '''
        global MIN_BPM, MAX_BPM
        MIN_BPM = min_bpm
        MAX_BPM = max_bpm
        MAX_COUNT = 10              # maximum number of consecutive corrections
        
        ibis = np.diff(beats_ix)
        beats = beats_ix[1:]        # drop the first beat

        global cnt, corrected_ibis, corrected_beats, corrected_flags
        global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, current_flag, correction_flags
        cnt = 0                     # increment when correcting the ibi and decrement when accepting the ibi
        prev_ibi = 0
        prev_beat = 0
        prev_flag = None
        current_ibi = 0
        current_beat = 0
        current_flag = None
        corrected_ibis = []
        corrected_beats = []
        corrected_flags = []
        correction_flags = [0 for i in range(len(beats))]

        ## DELETE THIS LATER
        global n_insert
        n_insert = 0

        class MaxNFifo:
            '''
            A class for FIFO with N elements at maximum.

            Parameters/Attributes
            ---------------------
            prev_n : int
                The maximum number of elements in the FIFO.
            '''
            def __init__(self, prev_n, item = None):
                '''
                Initialize the FIFO object.
                
                Parameters
                ---------------------
                prev_n : int
                    The maximum number of elements in the FIFO.
                item : int, optional
                    The initial item to add to the FIFO; by default, None.
                    The item is added twice if it is not None.
                '''
                self.prev_n = prev_n
                if item != None:
                    self.queue = [item, item]
                else:
                    self.queue = []

            def push(self, item):
                '''
                Push an item to the FIFO. If the number of elements exceeds the maximum, remove the first element.

                Parameters
                ---------------------
                item : int
                    The item to add to the FIFO.
                '''
                self.queue.append(item)
                if len(self.queue) > self.prev_n + 1:
                    self.queue.pop(0)

            def get_queue(self):
                '''
                Return the FIFO queue.

                Return
                ---------------------
                queue : list
                '''
                return self.queue
            
            def change_last(self, item):
                '''
                Change the last item in the FIFO queue.

                Parameters
                ---------------------
                item : int
                    The new item to replace the last item in the queue.
                '''
                self.queue[-1] = item
            
            def reset(self, item = None):
                '''
                Reset the FIFO queue. 
                If an item is given, reset the queue with the item. If not, reset the queue with an empty list.

                Parameters
                ---------------------
                item: int, optional
                    The item to add to the FIFO; by default, None.
                    The item is added twice if it is not None.
                '''
                if item == None:
                    self.queue = []
                else:
                    self.queue = [item, item]
        
        # Set the initial IBI to compare against
        global prev_ibis_fifo, first_ibi, correction_failed
        if initial_hr == 'auto':
            successive_diff = np.abs(np.diff(ibis))
            min_diff_ix = np.convolve(successive_diff, np.ones(hr_estimate_window) / hr_estimate_window, mode = 'valid').argmin()
            first_ibi = ibis[min_diff_ix:min_diff_ix + hr_estimate_window].mean()
            if print_estimated_hr:
                print('Estimated average HR (bpm): ', floor(60 / (first_ibi / self.fs)))
        else:
            first_ibi = self.fs * 60 / initial_hr
        
        prev_ibis_fifo = MaxNFifo(prev_n, first_ibi)        # FIFO for the previous n+1 IBIs
        correction_failed = MaxNFifo(prev_n - 1)            # Store whether the correction failed for the last n IBIs

        def detect_invalid_segments_mims(mims_data = mims, threshold=mims_threshold):
            '''
            A function to detect intervals with large motion artifacts based on the MIMS data.

            Parameters
            ---------------------
            mims_data : pandas.DataFrame
                A data frame containing the MIMS data with 'Timestamp' and 'MIMS_UNIT' columns.
            threshold : float
                The threshold for detecting motion artifacts; by default, 0.2.
            
            Returns
            ---------------------
            motion_intervals : list
                A list of tuples containing the start and end times of the motion intervals.
            '''
            # Detect intervals with large motion
            motion_intervals = []
            in_motion = False
            start_time = None
            
            for i in range(len(mims_data)):
                if mims_data['MIMS_UNIT'].iloc[i] > threshold:
                    if not in_motion:
                        in_motion = True
                        start_time = mims_data['Timestamp'][i]
                else:
                    if in_motion:
                        in_motion = False
                        end_time = mims_data['Timestamp'][i]
                        motion_intervals.append((start_time, end_time))
            
            # If the motion continues until the end of the data
            if in_motion:
                end_time = mims_data['Timestamp'][len(mims_data) - 1]
                motion_intervals.append((start_time, end_time))
            
            return motion_intervals if motion_intervals else None

        def segment_signal(data, seg_size = seg_size):
            data_segmented = data.copy()
            signal_start_time = data['Timestamp'].iloc[0]
            signal_end_time = data['Timestamp'].iloc[-1]
            n_seg = int(np.ceil((signal_end_time - signal_start_time).total_seconds() / seg_size))
            segments = []
            for i in range(n_seg):
                seg_start = signal_start_time + pd.Timedelta(seconds = i * seg_size)
                seg_end = signal_start_time + pd.Timedelta(seconds = (i + 1) * seg_size)
                if seg_end > signal_end_time:
                    seg_end = signal_end_time
                segments.append((seg_start, seg_end))
                if i == n_seg - 1:
                    data_segmented.loc[(data_segmented['Timestamp'] >= seg_start) & (data_segmented['Timestamp'] <= seg_end), 'Segment'] = i + 1
                else:
                    data_segmented.loc[(data_segmented['Timestamp'] >= seg_start) & (data_segmented['Timestamp'] < seg_end), 'Segment'] = i + 1
            segments_df = pd.DataFrame({
                'Segment': np.arange(1, n_seg + 1),
                'Start': [seg[0] for seg in segments],
                'End': [seg[1] for seg in segments]
            })
            return data_segmented, segments_df

        def identify_uncorrectable_segments(data, motion_intervals, motion_per_threshold):
            '''
            A function to identify uncorrectable segments.

            Parameters
            ---------------------
            data : pandas.DataFrame
                A data frame containing the IBIs with 'Timestamp' and 'IBI' columns.
            motion_intervals : list
                A list of tuples containing the start and end times of the motion intervals.
            motion_per_threshold : float
                The threshold for considering a segment as uncorrectable.
            
            Returns
            ---------------------
            data_segmented : pandas.DataFrame
                A data frame containing the segmented IBIs with 'Segment' column.
            segments_df : pandas.DataFrame
                A data frame containing the segment numbers, start and end times, and motion percentages.
            '''
            data_segmented, segments_df = segment_signal(data = data, seg_size = seg_size)
            seg_no = segments_df['Segment'].values

            motion_interval_idx = 0
            # For each segment
            for seg in seg_no:
                # Get the start and end times of the segment
                seg_start = segments_df.loc[segments_df['Segment'] == seg, 'Start'].values[0]
                seg_end = segments_df.loc[segments_df['Segment'] == seg, 'End'].values[0]
                # Initialize the sum of motion time in the segment
                sum_motion_time = 0
                # For each motion interval
                if motion_intervals is not None:
                    while motion_interval_idx < len(motion_intervals):
                        # Get the start and end times of the motion interval
                        motion_start = motion_intervals[motion_interval_idx][0]
                        motion_end = motion_intervals[motion_interval_idx][1]
                        # If the motion interval overlaps with the segment, add the period to the sum
                        # If the motion interval starts before the segment and ends within or after the segment
                        if (motion_start <= seg_start) and (seg_start <= motion_end):
                            sum_motion_time += (min(motion_end, seg_end) - seg_start).total_seconds()
                            if seg_end < motion_end:
                                break
                        # If the motion interval starts within the segment and ends within or after the segment
                        elif (seg_start <= motion_start) and (motion_start <= seg_end):
                            sum_motion_time += (min(motion_end, seg_end) - motion_start).total_seconds()
                            if seg_end < motion_end:
                                break
                        # If the motion interval starts after the segment, go to the next segment
                        elif (seg_end < motion_start):
                            break
                        
                        # If the motion interval ends in this segment, go to the next motion interval
                        if motion_end < seg_end:
                            motion_interval_idx += 1
            
                motion_percentage = round(sum_motion_time / seg_size, 4)
                segments_df.loc[segments_df['Segment'] == seg, 'Motion Percentage (%)'] = motion_percentage * 100
                if motion_percentage >= motion_per_threshold:
                    segments_df.loc[segments_df['Segment'] == seg, 'Correctable'] = False
                else:
                    segments_df.loc[segments_df['Segment'] == seg, 'Correctable'] = True
                
            return data_segmented, segments_df

        def is_invalid(row, motion_intervals):
            '''
            A function to check if the IBI is invalid based on the motion intervals and the valid IBI range.
            This function is called by the identify_invalid_IBIs function.

            Parameters
            ---------------------
            row : pandas.Series
                A row of the DataFrame containing the IBIs.
            motion_intervals : list
                A list of tuples containing the start and end times of the motion intervals.
            
            Returns
            ---------------------
            bool
                True if the IBI is invalid, False otherwise.
            '''
            # Check if bpm is in the 40 <= BPM <= 200 range
            if row['IBI'] < 300 or row['IBI'] > 1500:
                return True
            # Check if there is no motion intervals detected
            if motion_intervals is None:
                return False
            else:
                # Check if the IBI is within the motion intervals
                for start, end in motion_intervals:
                    if row['Timestamp'] >= start and row['Timestamp'] <= end:
                        return True
                return False

        def identify_invalid_IBIs(df_ibi, motion_intervals):
            '''
            A function to classify each IBI as invalid or valid based on the motion intervals and the valid IBI range.

            Parameters
            ---------------------
            df_ibi : pandas.DataFrame
                A data frame containing the IBIs.
            motion_intervals : list
                A list of tuples containing the start and end times of the motion intervals.
            
            Returns
            ---------------------
            df : pandas.DataFrame
                A data frame containing the Timestamp, IBIs, and a boolian column 'invalid' indicating whether the IBI is invalid.
            '''
            df = df_ibi.copy()
            if not df.empty:
                df['invalid'] = df.apply(is_invalid, motion_intervals = motion_intervals, axis = 1)
            else:
                df['invalid'] = np.nan
            return df

        def calculate_thresholds(ibis, motion_intervals, window_num):
            '''
            A function to calculate the adaptive thresholds for short, long, and extra long IBIs.

            Parameters
            ---------------------
            ibis : pandas.DataFrame
                A DataFrame containing the IBIs.
                This DataFrame should contain 'Timestamp' and 'IBI' columns.
            motion_intervals : list
                A list of tuples containing the start and end times of the motion intervals.
            window_num : int
                The window size (number of previous clean IBIs) used for calculating the adaptive thresholds.
            '''
            # Mark the IBIs as invalid if they are outside the valid range or within the motion intervals
            ibi_marked_invalid = identify_invalid_IBIs(ibis, motion_intervals)
            # Remove the invalid IBIs
            clean_ibi = ibi_marked_invalid[ibi_marked_invalid['invalid'] == False].copy()
            
            # Calculate the estimated IBI for each IBI
            clean_ibi['Estimated IBI'] = clean_ibi['IBI'].rolling(window=7, min_periods=1, closed='left').median()
            
            # Calculate the ratio of the estimated IBI to the actual IBI
            clean_ibi['Estimated IBI / IBI'] = clean_ibi['Estimated IBI'] / clean_ibi['IBI']
            
            # Calculate the mean and standard deviation of the estimated IBI / IBI
            clean_ibi['Mean'] = clean_ibi['Estimated IBI / IBI'].rolling(window=window_num, min_periods=1, closed='both') \
                .apply(lambda x: x.mean() if not x.empty else np.nan)
            # Approximate the standard deviation using the interquartile range
            clean_ibi['Std'] = clean_ibi['Estimated IBI / IBI'].rolling(window=window_num, min_periods=1, closed='both') \
                .apply(lambda x: (x.quantile(0.75) - x.quantile(0.25)) / 2 * 1.48 if not x.empty else np.nan)
            
            # Calculate the short, long, and extra long thresholds
            # If the mean and standard deviation are not available, use the default threshold values
            clean_ibi['Short threshold'] = (clean_ibi['Mean'] - 2 * clean_ibi['Std']).fillna(0.75)
            clean_ibi['Long threshold'] = (clean_ibi['Mean'] + 2 * clean_ibi['Std']).fillna(1.375)
            clean_ibi['Extra threshold'] = (clean_ibi['Mean'] + 4 * clean_ibi['Std']).fillna(1.625)
            
            return clean_ibi
        
        def adjust_threshold(row):
            if row['Short threshold'] < 0.5:
                row['Short threshold'] = 0.5
            if row['Long threshold'] > 1.5:
                row['Long threshold'] = 1.5
            if row['Extra threshold'] > 2.5:
                row['Extra threshold'] = 2.5
            return row

        def convert_ibi_idx_to_time(start_time, ibis):
            '''
            A function to convert the IBI indices to time-based DataFrame.
            This function is used when adaptive_threshold = True.

            Parameters
            ---------------------
            start_time : datetime
                The start time of the data.
            ibis : array_like
                An array containing index-based IBI.
            
            Returns
            ---------------------
            ibi_df : pandas.DataFrame
                A data frame containing the Timestamp and IBI columns.
            '''
            timestamps = []
            timestamp = start_time
            for i in ibis:
                timestamps.append(timestamp)
                timestamp += pd.Timedelta(milliseconds = i)
            ibi_df = pd.DataFrame({
                'Timestamp': timestamps,
                'IBI': [i / self.fs * 1000 for i in ibis]
            })
            return ibi_df

        def estimate_ibi(prev_ibis):
            '''
            Estimate IBI based on the previous IBIs.
            
            Parameters
            ---------------------
            prev_ibis: array_like
                A list of prev_n number of previous IBIs.
            
            Returns
            ---------------------
            estimated_ibi : int
            '''
            return np.median(prev_ibis)

        def return_flag(current_ibi, prev_ibis = None):
            '''
            Return whether the current IBI is correct, short, long, or extra long based on the previous IBIs.
                Correct: 26/32 - 44/32 of the estimated IBI
                Short: < 26/32 of the estimated IBI
                Long: > 44/32 and < 54/32 of the estimated IBI
                Extra Long: > 54/32 of the estimated IBI
            
            Parameters
            ---------------------
            current_ibi: int
                current IBI value in the number of indices.
            prev_ibis: array_like, optional
                A list of prev_n number of previous IBIs.
            
            Returns
            ---------------------
            flag : str
                The flag of the current IBI: 'Correct', 'Short', 'Long', or 'Extra Long'.
            '''
            # Calculate the estimated IBI
            estimated_ibi = estimate_ibi(prev_ibis)

            # Set the acceptable/valid range of IBIs
            low = short_threshold * estimated_ibi
            high = long_threshold * estimated_ibi
            extra = extra_threshold * estimated_ibi

            # flag the ibi: correct, short, long, or extra long
            if low <= current_ibi <= high:
                flag = 'Correct'
            elif current_ibi < low:
                flag = 'Short'
            elif current_ibi > high and current_ibi < extra:
                flag = 'Long'
            else:
                flag = 'Extra Long'
            
            # print('current:', current_ibi, ' prev:', prev_ibis, ' flag:', flag)
            
            return flag
        
        def acceptance_check(corrected_ibi, prev_ibis):
            '''
            Check if the corrected IBI is acceptable (falls within the 27/32 - 42/32 of the estimated IBI).

            Parameters
            ---------------------
            corrected_ibi: int
                The corrected IBI value.
            prev_ibis: array_like
                A list of prev_n number of previous IBIs.
            
            Returns
            ---------------------
            bool
                True if the corrected IBI is within the acceptable range, False otherwise.
            '''
            # Calculate the estimated IBI
            estimated_ibi = estimate_ibi(prev_ibis)
            
            # Set the acceptable/valid range of IBIs
            low = short_threshold * estimated_ibi
            high = long_threshold * estimated_ibi

            # If the corrected value is within the range, return True
            if corrected_ibi >= low and corrected_ibi <= high:
            #if corrected_ibi <= high:
                return True
            else:
                return False

        def accept_ibi(n, correction_failed_flag = 0):
            '''
            Accept the current IBI without correction.

            Parameters
            ---------------------
            n : int
                The index of the current IBI.
            correction_failed_flag : int, optional
                Flag to indicate whether the correction failed for the current IBI; by default, 0.
                If the flag is 1, the correction failed for the current IBI.
            '''
            global prev_ibis_fifo, cnt, correction_failed
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, current_flag
            
            # Check if the previous IBI is within the limits before accepting the current IBI
            check_limits(n)

            # Fix the previous IBI
            corrected_ibis.append(prev_ibi)
            corrected_beats.append(prev_beat)
            corrected_flags.append(prev_flag)

            # Add the previous IBI to the queue
            prev_ibis_fifo.push(prev_ibi)

            # Update the previous IBI to the current IBI
            prev_ibi = current_ibi
            prev_beat = current_beat
            prev_flag = current_flag
            
            # Decrement the counter
            cnt = max(0, cnt-1)
            if DEBUGGING == True:
                print('accepted:', current_ibi, ' flag:', current_flag, ' based on ', prev_ibis_fifo.get_queue()[1:])
            # If the correction failed for the current IBI, push 1 to the correction_failed FIFO, otherwise push 0
            if correction_failed_flag == 0:
                correction_failed.push(0)
            else:
                correction_failed.push(1)
        
        def add_prev_and_current(n):
            '''
            Add the previous and current IBIs if the sum is less than 42/32 of the estimated IBI.

            Parameters
            ---------------------
            n : int
                The index of the current IBI.
            '''
            global prev_ibis_fifo, cnt
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, current_flag, correction_flags
            
            # Add the previous and current IBIs
            corrected_ibi = prev_ibi + current_ibi

            # Check if the corrected IBI is acceptable
            if acceptance_check(corrected_ibi, prev_ibis_fifo.get_queue()[1:]) == True:
                # Update the current IBI to the corrected IBI
                current_ibi = corrected_ibi
                current_beat = current_beat
                current_flag = return_flag(current_ibi, prev_ibis_fifo.get_queue()[1:])
                
                if n == 1:
                    # Update the previous IBI to the current IBI
                    prev_ibi = current_ibi
                    prev_beat = current_beat
                    prev_flag = current_flag

                else:
                    # Pull up the second previous IBI as previous IBI
                    prev_ibi = corrected_ibis[-1]
                    prev_beat = corrected_beats[-1]
                    prev_flag = corrected_flags[-1]
                    
                    # Check if the previous IBI is within the limits before accepting the current IBI
                    check_limits(n)
                    
                    # Check_limits function may update the previous IBI pulled, so update the value
                    corrected_ibis[-1] = prev_ibi
                    corrected_beats[-1] = prev_beat
                    corrected_flags[-1] = prev_flag

                    # Update the last IBI value in the queue
                    prev_ibis_fifo.change_last(prev_ibi)

                    # Update the previous IBI to the current IBI
                    prev_ibi = current_ibi
                    prev_beat = current_beat
                    prev_flag = current_flag

                # Flag that previous and current IBIs are corrected
                correction_flags[n-1] = 1
                correction_flags[n] = 1

                # Increment the counter
                cnt += 1

                if DEBUGGING == True:
                    print('added:', current_ibi, ' flag:', current_flag, ' based on ', prev_ibis_fifo.get_queue()[1:])
            else:
                if DEBUGGING == True:
                    print('acceptance check failed for adding: ', corrected_ibi)
                # If the corrected IBI is not acceptable, accept the current IBI
                accept_ibi(n, correction_failed_flag=1)
        
        def add_secondprev_and_prev(n):
            '''
            Add the second previous and previous IBIs if the sum is less than 42/32 of the estimated IBI.

            Parameters
            ---------------------
            n : int
                The index of the current IBI.
            '''
            global prev_ibis_fifo, cnt
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, current_flag, correction_flags
            
            # Add the previous and current IBIs
            corrected_ibi = corrected_ibis[-1] + prev_ibi

            # Check if the corrected IBI is acceptable
            # Use IBIs before the second previous IBI
            if acceptance_check(corrected_ibi, prev_ibis_fifo.get_queue()[:-2]) == True:
                # Update the current IBI to the corrected IBI
                
                # Pull up the second previous IBI as previous IBI
                prev_ibi = corrected_ibi
                prev_beat = prev_beat
                prev_flag = return_flag(prev_ibi, prev_ibis_fifo.get_queue()[:-2])
                
                # Check if the previous IBI is within the limits before accepting the current IBI
                check_limits(n)
                
                # Update the value
                corrected_ibis[-1] = prev_ibi
                corrected_beats[-1] = prev_beat
                corrected_flags[-1] = prev_flag

                # Update the last IBI value in the queue
                prev_ibis_fifo.change_last(prev_ibi)

                # Update the previous IBI to the current IBI
                prev_ibi = current_ibi
                prev_beat = current_beat
                prev_flag = current_flag

                # Flag that previous and current IBIs are corrected
                correction_flags[n-2] = 1
                correction_flags[n-1] = 1

                # Increment the counter
                cnt += 1

                if DEBUGGING == True:
                    print('added second prev + prev:', prev_ibi, ' flag:', prev_flag, ' based on ', prev_ibis_fifo.get_queue()[:-2])
            else:
                if DEBUGGING == True:
                    print('acceptance check failed for adding second prev + prev: ', corrected_ibi)
                # If the corrected IBI is not acceptable, accept the current IBI
                accept_ibi(n, correction_failed_flag=1)
                
        def insert_interval(n):
            '''
            Split the (previous IBI + current IBI) into multiple intervals. 
            The number of splits is determined based on the initial_hr parameter.
            
            Parameters
            ---------------------
            n : int
                The index of the current IBI.
            '''
            global prev_ibis_fifo, cnt, first_ibi
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, current_flag, correction_flags

            ## DELETE THIS LATER
            global n_insert
            
            # Calculate the number of splits
            n_split = round((prev_ibi + current_ibi) / estimate_ibi(prev_ibis_fifo.get_queue()[1:]), 0).astype(int)

            # Calculate the new IBI
            ibi = floor((prev_ibi + current_ibi) / n_split)

            # Check if the corrected IBI is acceptable
            if acceptance_check(ibi, prev_ibis_fifo.get_queue()[1:]) == True:
                # Fix inserted IBIs other than previous/current IBIs
                for i in range(n_split - 2):
                    corrected_ibis.append(ibi)
                    corrected_flags.append(return_flag(ibi, prev_ibis_fifo.get_queue()[1:]))
                    if n == 1 and i == 0:
                        corrected_beats.append(beats_ix[0] + ibi)
                    else:
                        corrected_beats.append(corrected_beats[-1] + ibi)
                    # Add to the queue
                    prev_ibis_fifo.push(ibi)

                # Update the previous IBI
                prev_ibi = ibi
                prev_beat = corrected_beats[-1] + ibi
                prev_flag = return_flag(ibi, prev_ibis_fifo.get_queue()[:-1])

                # Update the current IBI
                current_ibi = current_beat - prev_beat
                current_flag = return_flag(ibi, prev_ibis_fifo.get_queue()[1:])

                # Check if the previous IBI is within the limits
                check_limits(n)

                # Fix the previous IBI
                corrected_ibis.append(prev_ibi)
                corrected_beats.append(prev_beat)
                corrected_flags.append(prev_flag)

                # Add to the queue
                prev_ibis_fifo.push(prev_ibi)
                
                # Update the previous IBI to the current IBI
                prev_ibi = current_ibi
                prev_beat = current_beat
                prev_flag = current_flag

                # Flag that previous and current IBIs are corrected
                correction_flags[n-1] = 1
                correction_flags[n] = 1

                # Increment the counter by n_split - 1 in this case
                cnt += n_split - 1

                ## DELETE THIS LATER
                n_insert += n_split - 2

                if DEBUGGING == True:
                    print('inserted ',n_split - 2, ' intervals: ', ibi, ' flag:', current_flag, ' based on ', prev_ibis_fifo.get_queue()[1:])
            else:
                if DEBUGGING == True:
                    print('acceptance check failed for inserting: ', ibi)
                # If the corrected IBI is not acceptable, accept the current IBI
                accept_ibi(n, correction_failed_flag=1)
            

        def average_prev_and_current(n):
            '''
            Average the previous and current IBIs.

            Parameters
            ---------------------
            n : int
                The index of the current IBI.
            '''
            global prev_ibis_fifo, cnt
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, current_flag, correction_flags
            
            # Average the previous and current IBIs
            ibi = floor((prev_ibi + current_ibi) / 2)
            
            # Check if the corrected IBI is acceptable
            if acceptance_check(ibi, prev_ibis_fifo.get_queue()[1:]) == True:
                # Update the previous and current IBI
                prev_ibi = ibi
                if n == 1:
                    prev_beat = beats_ix[0] + ibi
                else:
                    prev_beat = corrected_beats[-1] + ibi
                prev_flag = return_flag(ibi, prev_ibis_fifo.get_queue()[:-1])
                current_ibi = current_beat - prev_beat
                current_flag = return_flag(ibi, prev_ibis_fifo.get_queue()[1:])

                # Check if the previous IBI is within the limits
                check_limits(n)

                # Fix the previous IBI
                corrected_ibis.append(prev_ibi)
                corrected_beats.append(prev_beat)
                corrected_flags.append(prev_flag)

                # Add to the queue
                prev_ibis_fifo.push(prev_ibi)

                # Update the previous IBI to the current IBI
                prev_ibi = current_ibi
                prev_beat = current_beat
                prev_flag = current_flag
                
                # Flag that previous and current IBIs are corrected
                correction_flags[n-1] = 1
                correction_flags[n] = 1

                # Increment the counter
                cnt += 1

                if DEBUGGING == True:
                    print('averaged:', ibi, ' flag:', current_flag, ' based on ', prev_ibis_fifo.get_queue()[1:])
            else:
                if DEBUGGING == True:
                    print('acceptance check failed for averaging: ', ibi)
                accept_ibi(n, correction_failed_flag=1)
                

        def check_limits(n):
            '''
            Check if the previous IBI (n-1) is within the limits.
            If it is longer the maximum IBI, shorten the previous IBI and lengthen the current IBI.
            If it is shorter than the minimum IBI, lengthen the previous IBI and shorten the current IBI.

            Parameters
            ---------------------
            n : int
                The index of the current IBI.
            '''
            global prev_ibis_fifo, cnt
            global corrected_ibis, corrected_beats, corrected_flags
            global prev_ibi, prev_beat, prev_flag, current_ibi, current_beat, current_flag, correction_flags
            MIN_IBI = floor(self.fs * 60 / MAX_BPM)         # minimum IBI in indices
            MAX_IBI = floor(self.fs * 60 / MIN_BPM)         # maximum IBI in indices

            # If the previous IBI is shorter than the minimum IBI, lengthen the previous IBI and shorten the current IBI
            if prev_ibi < MIN_IBI:
                remainder = MIN_IBI - prev_ibi
                prev_beat = prev_beat + remainder
                prev_ibi = MIN_IBI
                prev_flag = return_flag(prev_ibi, prev_ibis_fifo.get_queue()[:-1])
                current_ibi = current_ibi - remainder
                current_flag = return_flag(current_ibi, prev_ibis_fifo.get_queue()[1:])

                # Flag that previous and current IBIs are corrected
                correction_flags[n-1] = 1
                correction_flags[n] = 1

                # Increment the counter
                cnt += 1

                if DEBUGGING == True:
                    print('Shorter than the minimum IBI and corrected: ', prev_ibi, ' ', prev_flag, ' | ', current_ibi, ' ', current_flag)

            # If the previous IBI is longer than the maximum IBI, shorten the previous IBI and lengthen the current IBI
            elif prev_ibi > MAX_IBI:
                remainder = prev_ibi - MAX_IBI
                prev_beat = prev_beat - remainder
                prev_ibi = MAX_IBI
                prev_flag = return_flag(prev_ibi, prev_ibis_fifo.get_queue()[:-1])
                current_ibi = current_ibi + remainder
                current_flag = return_flag(current_ibi, prev_ibis_fifo.get_queue()[1:])

                # Flag that previous and current IBIs are corrected
                correction_flags[n-1] = 1
                correction_flags[n] = 1

                # Increment the counter
                cnt += 1

                if DEBUGGING == True:
                    print('Longer than the maximum IBI and corrected: ', prev_ibi, ' ', prev_flag, ' | ', current_ibi, ' ', current_flag)
            return

        short_thresholds = []
        long_thresholds = []
        extra_thresholds = []
        uncorrectable = []
        if adaptive_threshold or ignore_motion_seg:
            # Detect intervals with large motion artifacts
            motion_intervals = detect_invalid_segments_mims(mims_data = mims, threshold = mims_threshold)
        if ignore_motion_seg:
            ibi_data_segmented, segments_df = identify_uncorrectable_segments(data = ibi_data, motion_intervals = motion_intervals, motion_per_threshold = motion_per_threshold)
        for n in range(len(ibis)):
            current_ibi = ibis[n]
            current_beat = beats[n]
            if ignore_motion_seg:
                # Get the current segment
                seg = ibi_data_segmented['Segment'].iloc[n]
                # If the current segment is uncorrectable, accept the current IBI
                if segments_df.loc[segments_df['Segment'] == seg, 'Correctable'].values[0] == False:
                    # If the current segment is uncorrectable, use the default thresholds to flag IBIs
                    short_threshold = 0.75
                    long_threshold = 1.375
                    extra_threshold = 1.625
                    # Append the IBI as it is
                    corrected_ibis.append(ibis[n])
                    corrected_beats.append(beats[n])
                    corrected_flags.append(return_flag(current_ibi, prev_ibis = prev_ibis_fifo.get_queue()))
                    uncorrectable.append(True)
                    # Update the previous IBI to the current IBI
                    prev_ibi = current_ibi
                    prev_beat = current_beat
                    prev_flag = return_flag(current_ibi, prev_ibis = prev_ibis_fifo.get_queue())
                    continue
                else:
                    uncorrectable.append(False)
            # If the adaptive threshold is enabled, calculate the thresholds
            if adaptive_threshold and ibi_data is not None:
                # Get the timestamp of the first IBI
                start_time = ibi_data['Timestamp'].iloc[0]
                # Convert the corrected IBI indices to time-based DataFrame
                # This dataframe only contains the corrected previous IBIs
                ibi_df = convert_ibi_idx_to_time(start_time, corrected_ibis)
                # Calculate the adaptive thresholds based on the corrected IBIs
                thresholds = calculate_thresholds(ibi_df, motion_intervals, window_num)

                if adaptive_threshold:
                    beat_idx = beats[n]
                    # If the threshold is empty, use the default values
                    if thresholds.empty:
                        short_threshold = 0.75
                        long_threshold = 1.375
                        extra_threshold = 1.625
                    else:
                        # If not empty, adjust thresholds outside of the valid range
                        adjusted_thresholds = thresholds.apply(adjust_threshold, axis=1)
                        # Get the calculated thresholds for the current beat
                        short_threshold = adjusted_thresholds['Short threshold'].iloc[-1]
                        long_threshold = adjusted_thresholds['Long threshold'].iloc[-1]
                        extra_threshold = adjusted_thresholds['Extra threshold'].iloc[-1]
                    short_thresholds.append(short_threshold)
                    long_thresholds.append(long_threshold)
                    extra_thresholds.append(extra_threshold)
            
            elif adaptive_threshold and ibi_data is None:
                raise ValueError('The data parameter must be provided to calculate the adaptive thresholds.')
                
            # Accept the first ibi
            if n == 0:
                current_flag = return_flag(current_ibi, prev_ibis = prev_ibis_fifo.get_queue())
                # Update the previous IBI to the current IBI
                prev_ibi = current_ibi
                prev_beat = current_beat
                prev_flag = current_flag
            
            # If the counter reaches the maximum count, accept the current IBI
            #elif cnt > MAX_COUNT:
                #current_flag = return_flag(current_ibi, prev_ibis = prev_ibis_fifo.get_queue()[:-1])
                #accept_ibi(n)
                # decrement by 2 when cnt > MAX_COUNT
                #cnt -= 1
            
            else:
                current_flag = return_flag(current_ibi, prev_ibis = prev_ibis_fifo.get_queue()[:-1])
                # If the current ibi is correct
                if DEBUGGING == True:
                    print('n:', n)
                    print('prev:', prev_ibi, ' ', prev_flag, ' | current:', current_ibi, ' ', current_flag)
                if current_flag == 'Correct':
                    if prev_flag == 'Correct' or prev_flag == 'Long':
                        accept_ibi(n)                           # If the previous ibi is correct or long, then accept the current ibi
                    elif prev_flag == 'Short':
                        if n == 1:
                            add_prev_and_current(n) 
                        else:
                            if corrected_ibis[-1] > current_ibi:
                                add_prev_and_current(n)                 # If the previous ibi is short, add previous and current intervals
                            else:
                                add_secondprev_and_prev(n)
                    elif prev_flag == 'Extra Long':
                        insert_interval(n)                      # If the previous ibi is extra long, split the previous and current intervals
                # If the current ibi is short
                elif current_flag == 'Short':
                    if prev_flag == 'Correct':
                        accept_ibi(n)                           # If the previous ibi is correct, accept previous
                    elif prev_flag == 'Short':
                        add_prev_and_current(n)                 # If the previous ibi is short, add previous and current intervals
                    elif prev_flag == 'Long' or prev_flag == 'Extra Long':
                        average_prev_and_current(n)             # If the previous ibi is long or extra long, average the previous and current intervals
                # If the current ibi is long
                elif current_flag == 'Long':
                    if prev_flag == 'Correct' or prev_flag == 'Long':
                        accept_ibi(n)                           # If the previous ibi is correct or long, accept previous
                    elif prev_flag == 'Short':
                        average_prev_and_current(n)             # If the previous ibi is short, average previous and current intervals
                    elif prev_flag == 'Extra Long':
                        insert_interval(n)                      # If the previous ibi is extra long, split the previous and current intervals
                # If the current ibi is extra long
                elif current_flag == 'Extra Long':
                    if prev_flag == 'Correct' or prev_flag == 'Long' or prev_flag == 'Extra Long':
                        insert_interval(n)                      # If the previous ibi is correct, long, or extra long, split the previous and current intervals
                    elif prev_flag == 'Short':
                        average_prev_and_current(n)             # If the previous ibi is short, average previous and current intervals

            # If more than 3 corrections are made in the last prev_n IBIs, reset the FIFO
            if sum(correction_failed.get_queue()) >= 3:
                prev_ibis_fifo.reset(first_ibi)

        # Add the last beat
        corrected_ibis.append(current_ibi)
        corrected_beats.append(current_beat)
        corrected_flags.append(current_flag)

        correction_flags = np.array(correction_flags).astype(int)

        # Convert the IBIs to milliseconds
        original_ibis_ms = np.round((np.array(ibis) / self.fs) * 1000, 2)
        
        # Add the first beat and create a dataframe
        if ibi_data is not None:
            timestamp = ibi_data['Timestamp'].iloc[0]
            timestamps = []
            for ibi in original_ibis_ms:
                timestamps.append(timestamp)
                timestamp += pd.Timedelta(milliseconds = ibi)
            timestamps.append(timestamp)
            original = pd.DataFrame(
                {'Timestamp': timestamps,
                'Original IBI (ms)': np.insert(original_ibis_ms, 0, np.nan),
                'Original IBI (index)': np.insert(ibis.astype(object), 0, np.nan),
                'Original Beat': np.insert(beats, 0, beats_ix[0]),
                'Correction': np.insert(correction_flags, 0, 0)}
            )
        else:
            original = pd.DataFrame(
                {'Original IBI (ms)': np.insert(original_ibis_ms, 0, np.nan),
                'Original IBI (index)': np.insert(ibis.astype(object), 0, np.nan),
                'Original Beat': np.insert(beats, 0, beats_ix[0]),
                'Correction': np.insert(correction_flags, 0, 0)}
            )
        
        corrected_ibis_ms = np.round((np.array(corrected_ibis) / self.fs) * 1000, 2)

        corrected_ibis = np.array(corrected_ibis).astype(object)
        corrected_flags = np.array(corrected_flags).astype(object)

        if ibi_data is not None:
            timestamp = ibi_data['Timestamp'].iloc[0]
            timestamps = []
            for ibi in corrected_ibis_ms:
                timestamps.append(timestamp)
                timestamp += pd.Timedelta(milliseconds = ibi)
            timestamps.append(timestamp)
            corrected = pd.DataFrame(
                {'Timestamp': timestamps,
                'Corrected IBI (ms)': np.insert(corrected_ibis_ms, 0, np.nan),
                'Corrected IBI (index)': np.insert(corrected_ibis, 0, np.nan), 
                'Corrected Beat': np.insert(corrected_beats, 0, beats_ix[0]),
                'Flag': np.insert(corrected_flags, 0, np.nan)}
            )
        else:
            # Add the first beat and create a dataframe
            corrected = pd.DataFrame(
                {'Corrected IBI (ms)': np.insert(corrected_ibis_ms, 0, np.nan),
                'Corrected IBI (index)': np.insert(corrected_ibis, 0, np.nan), 
                'Corrected Beat': np.insert(corrected_beats, 0, beats_ix[0]),
                'Flag': np.insert(corrected_flags, 0, np.nan)}
            )

        # print('Number of inserted intervals: ', n_insert)

        if adaptive_threshold:
            thresholds_all = pd.DataFrame({
                'Short threshold': short_thresholds,
                'Long threshold': long_thresholds,
                'Extra threshold': extra_thresholds
            })
            if ignore_motion_seg:
                original['Uncorrectable'] = np.insert(uncorrectable, 0, np.nan)
            
            if ignore_motion_seg:
                return original, corrected, thresholds_all, segments_df
            else:
                return original, corrected, thresholds_all, None
        else:
            if ignore_motion_seg:
                return original, corrected, None, segments_df
            else:
                return original, corrected, None, None
    
    def get_corrected(self, beats_ix, seg_size = 60, initial_hr = 'auto', prev_n = 6, min_bpm = 40, max_bpm = 200, hr_estimate_window = 6, print_estimated_hr = True,
                        short_threshold = (24 / 32),  long_threshold = (44 / 32), extra_threshold = (52 / 32), adaptive_threshold = False, ibi_data = None, 
                        window_num = 7, mims = None, mims_threshold = 0.2, ignore_motion_seg = False, motion_per_threshold = 0.5):
        """
        Get the corrected interbeat intervals (IBIs) and beat indices.

        Parameters
        ----------
        data : pandas.DataFrame
            A data frame containing the pre-processed ECG or PPG data.
        beats_ix : array_like
            An array containing the indices of detected beats.
        seg_size : int
            The size of the segment in seconds; by default, 60.
        initial_hr : int, float, optional
            The heart rate value for the first interbeat interval (IBI) to be
            validated against; by default, 80 bpm (750 ms).
        prev_n : int, optional
            The number of preceding IBIs to validate against; by default, 6.

        Returns
        -------
        corrected : pandas.DataFrame
            A data frame containing the summary of flags in each segment.
            # and % of beats flagged
            # of Accepted/short/long/extra long flags
        """
        # Get the corrected IBIs and beat indices
        original, corrected, thresholds, segments_df = self.correct_interval(beats_ix=beats_ix, seg_size = seg_size, initial_hr=initial_hr, prev_n=prev_n, min_bpm=min_bpm, max_bpm=max_bpm, hr_estimate_window=hr_estimate_window, print_estimated_hr = print_estimated_hr,
                                                short_threshold=short_threshold, long_threshold= long_threshold, extra_threshold=extra_threshold, adaptive_threshold = adaptive_threshold, ibi_data = ibi_data, 
                                                mims = mims, mims_threshold = mims_threshold, ignore_motion_seg = ignore_motion_seg, motion_per_threshold = motion_per_threshold, window_num = window_num)

        # Get the segment number for each beat
        for row in original.iterrows():
            if ibi_data is not None:
                seg = ceil(row[1][3] / (seg_size * self.fs))
            else:
                seg = ceil(row[1][2] / (seg_size * self.fs))
            original.loc[row[0], 'Segment'] = seg
        for row in corrected.iterrows():
            if ibi_data is not None:
                seg = ceil(row[1][3] / (seg_size * self.fs))
            else:
                seg = ceil(row[1][2] / (seg_size * self.fs))
            corrected.loc[row[0], 'Segment'] = seg
        original['Segment'] = original['Segment'].astype(pd.Int64Dtype())
        corrected['Segment'] = corrected['Segment'].astype(pd.Int64Dtype())
        
        # Get the number and percentage of corrected beats in each segment
        original_seg = original.groupby('Segment')['Correction'].sum().astype(pd.Int64Dtype())
        original_seg = pd.DataFrame(original_seg.reset_index(name = '# Corrected'))
        original_seg_nbeats = original.groupby('Segment')['Correction'].count().astype(pd.Int64Dtype())
        original_seg_nbeats = pd.DataFrame(original_seg_nbeats.reset_index(name = '# Beats'))
        original_seg = original_seg.merge(original_seg_nbeats, on = 'Segment')
        original_seg['% Corrected'] = round((original_seg['# Corrected'] / original_seg['# Beats']) * 100, 2)
        original_seg.drop('# Beats', axis = 1, inplace = True)

        # Get the number of each flag (Correct/Short/Long/Extra Long) in each segment
        corrected_seg = corrected.groupby('Segment')['Flag'].value_counts().astype(pd.Int64Dtype())
        corrected_seg = pd.DataFrame(corrected_seg.reset_index(name = 'Count'))
        corrected_seg = corrected_seg.pivot(index = 'Segment', columns = 'Flag', values = 'Count').reset_index().fillna(0)
        corrected_seg.columns.name = None
        corrected_seg = corrected_seg.rename_axis(None, axis = 1)
    
        combined = pd.merge(corrected_seg, original_seg, on='Segment')
        if ignore_motion_seg:
            combined = pd.merge(combined, segments_df[['Segment', 'Motion Percentage (%)', 'Correctable']], on = 'Segment')

        return original, corrected, thresholds, combined

    def get_missing(self, data, beats_ix, seg_size = 60, min_hr = 40,
                    ts_col = None, show_progress = True):
        """
        Get the number and proportion of missing beats per segment.

        Parameters
        ----------
        data : pandas.DataFrame
            The data frame containing the pre-processed ECG or PPG data.
        beats_ix : array-like
            An array containing the indices of detected beats.
        seg_size : int
            The size of the segment in seconds; by default, 60.
        min_hr : int, float
            The minimum acceptable heart rate against which the number of
            beats in the last partial segment will be compared; by default, 40.
        ts_col : str, optional
            The name of the column containing timestamps; by default, None.
            If a string value is given, the output will contain a timestamps
            column.
        show_progress : bool, optional
            Whether to display a progress by while the function runs; by
            default, True.

        Returns
        -------
        missing : pandas.DataFrame
            A DataFrame with detected, expected, and missing numbers of
            beats per segment.
        """
        seconds = self.get_seconds(data, beats_ix, ts_col, show_progress)
        seconds.index = seconds.index.astype(int)

        n_seg = ceil(len(seconds) / seg_size)
        segments = pd.Series(np.arange(1, n_seg + 1))
        n_expected = seconds.groupby(
            seconds.index // seg_size)['Mean HR'].median()
        n_detected = seconds.groupby(
            seconds.index // seg_size)['N Beats'].sum()
        n_missing = (n_expected - n_detected).clip(lower = 0)
        perc_missing = round((n_missing / n_expected) * 100, 2)

        # Handle last partial segment of data
        last_seg_len = len(seconds) % seg_size
        if last_seg_len > 0:
            last_detected = n_detected.iloc[-1]
            last_expected_ratio = min_hr / n_expected.iloc[:-1].median()
            last_expected = last_expected_ratio * last_seg_len
            if last_expected > last_detected:
                last_n_missing = last_expected - last_detected
                last_perc_missing = round(
                    (last_n_missing / last_expected) * 100, 2)
            else:
                last_perc_missing = 0
                last_n_missing = 0
            n_expected.iloc[-1] = last_expected
            n_missing.iloc[-1] = last_n_missing
            perc_missing.iloc[-1] = last_perc_missing

        if ts_col is not None:
            timestamps = seconds.groupby(
                seconds.index // seg_size).first()[ts_col]
            missing = pd.concat([
                segments,
                timestamps,
                n_detected,
                n_expected,
                n_missing,
                perc_missing,
            ], axis = 1)
            missing.columns = [
                'Segment',
                'Timestamp',
                'N Detected',
                'N Expected',
                'N Missing',
                '% Missing',
            ]
        else:
            missing = pd.concat([
                segments,
                n_detected,
                n_expected,
                n_missing,
                perc_missing,
            ], axis = 1)
            missing.columns = [
                'Segment',
                'N Detected',
                'N Expected',
                'N Missing',
                '% Missing',
            ]
        return missing

    def get_seconds(self, data, beats_ix, ts_col = None, show_progress = True):
        """Get second-by-second HR, IBI, and beat counts from ECG or PPG data
        according to the approach by Graham (1978).

        Parameters
        ----------
        data : pd.DataFrame
            The data frame containing the pre-processed ECG or PPG data.
        beats_ix : array-like
            An array containing the indices of detected beats.
        ts_col : str, optional
            The name of the column containing timestamps; by default, None.
            If a string value is given, the output will contain a timestamps
            column.
        show_progress : bool, optional
            Whether to display a progress bar while the function runs; by
            default, True.

        Returns
        -------
        interval_data : pd.DataFrame
            A data frame containing second-by-second HR and IBI values.

        Notes
        -----
        Rows with `NaN` values in the resulting data frame `interval_data`
        denote seconds during which no beats in the data were detected.

        References
        ----------
        Graham, F. K. (1978). Constraints on measuring heart rate and period
        sequentially through real and cardiac time. Psychophysiology, 15(5),
        492–495.
        """

        df = data.copy()
        df.index = df.index.astype(int)
        df.loc[beats_ix, 'Beat'] = 1

        interval_data = []

        # Iterate over each second
        s = 1
        for i in tqdm(range(0, len(df), self.fs), disable = not show_progress):

            # Get data at the current second and evaluation window
            current_sec = df.iloc[i:(i + self.fs)]
            if i == 0:
                # Look at current and next second
                window = df.iloc[:(i + self.fs)]
            else:
                # Look at previous, current, and next second
                window = df.iloc[(i - self.fs):(min(i + self.fs, len(df)))]

            # Get mean IBI and HR values from the detected beats
            current_beats = current_sec[current_sec['Beat'] == 1].index.values
            window_beats = window[window['Beat'] == 1].index.values
            ibis = np.diff(window_beats) / self.fs * 1000
            if len(ibis) == 0:
                mean_ibi = np.nan
                mean_hr = np.nan
            else:
                mean_ibi = np.mean(ibis)
                hrs = 60000 / ibis
                r_hrs = 1 / hrs
                mean_hr = 1 / np.mean(r_hrs)

            # Append values for the current second
            if ts_col is not None:
                interval_data.append({
                    'Second': s,
                    'Timestamp': current_sec.iloc[0][ts_col],
                    'Mean HR': mean_hr,
                    'Mean IBI': mean_ibi,
                    'N Beats': len(current_beats)
                })
            else:
                interval_data.append({
                    'Second': s,
                    'Mean HR': mean_hr,
                    'Mean IBI': mean_ibi,
                    'N Beats': len(current_beats)
                })

            s += 1
        interval_data = pd.DataFrame(interval_data)
        return interval_data

    def _get_iqr(self, data):
        """Compute the interquartile range of a data array."""
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        return iqr

    def _quartile_deviation(self, data):
        """Compute the quartile deviation in the criterion beat difference
        test."""
        iqr = self._get_iqr(data)
        QD = iqr * 0.5
        return QD
    