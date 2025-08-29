from scipy.signal import butter, lfilter, hilbert
import numpy as np


class EEGFeatureExtractor:
    def __init__(self, eeg_data, sampling_rate=128):
        """
        初始化特征提取器
        :param eeg_data: EEG信号数据，形状为 (trials, channels, timepoints)
        :param sampling_rate: 采样率，默认为128Hz
        """
        self.eeg_data = eeg_data
        self.sampling_rate = sampling_rate
        self.frequency_bands = {
            'theta': (4, 7),
            'alpha': (8, 13),
            'beta': (14, 30),
            'gamma': (31, 50)
        }

    def bandpass_filter(self, data, lowcut, highcut):
        """
        带通滤波器
        :param data: 输入数据
        :param lowcut: 低截止频率
        :param highcut: 高截止频率
        :return: 滤波后的数据
        """
        nyquist = 0.5 * self.sampling_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(4, [low, high], btype='band')
        return lfilter(b, a, data, axis=-1)

    def calculate_pcc(self, data):
        """
        计算所有通道之间的皮尔逊相关系数矩阵
        :param data: 滤波后的数据，形状为 (channels, timepoints)
        :return: 皮尔逊相关系数矩阵，形状为 (channels, channels)
        """
        if np.any(np.std(data, axis=1) == 0):
            raise ValueError("One or more channels have zero variance, PCC cannot be calculated.")
        data = (data - np.mean(data, axis=1, keepdims=True)) / np.std(data, axis=1, keepdims=True)
        return np.corrcoef(data)

    def extract_PCC_features(self):
        """
        提取每个试验的PCC特征
        :return: 特征矩阵，形状为 (trials, bands, channels, channels)
        """
        trials, channels, timepoints = self.eeg_data.shape
        features = np.zeros((trials, len(self.frequency_bands), channels, channels))

        for trial in range(trials):
            for band_idx, (band_name, (low, high)) in enumerate(self.frequency_bands.items()):
                filtered_data = self.bandpass_filter(self.eeg_data[trial], low, high)
                if np.any(np.isnan(filtered_data)):
                    raise ValueError(f"Filtered data contains NaN values for trial {trial}, band {band_name}.")
                pcc_matrix = self.calculate_pcc(filtered_data)
                features[trial, band_idx] = pcc_matrix

        return features

    def calculate_plv(self, data):
        n_channels = data.shape[0]
        plv_matrix = np.zeros((n_channels, n_channels))
        analytic_signal = hilbert(data, axis=-1)
        phase_data = np.angle(analytic_signal)

        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                phase_diff = phase_data[i] - phase_data[j]
                plv_matrix[i, j] = np.abs(np.sum(np.exp(1j * phase_diff)) / len(phase_diff))
                plv_matrix[j, i] = plv_matrix[i, j]

        return plv_matrix

    def extract_PLV_features(self):
        trials, channels, timepoints = self.eeg_data.shape
        features = np.zeros((trials, len(self.frequency_bands), channels, channels))

        for trial in range(trials):
            for band_idx, (band_name, (low, high)) in enumerate(self.frequency_bands.items()):
                filtered_data = self.bandpass_filter(self.eeg_data[trial], low, high)
                if np.any(np.isnan(filtered_data)):
                    raise ValueError(f"Filtered data contains NaN values for trial {trial}, band {band_name}.")
                plv_matrix = self.calculate_plv(filtered_data)
                features[trial, band_idx] = plv_matrix

        return features
