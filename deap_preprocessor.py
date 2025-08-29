import numpy as np
import pandas as pd
import pickle


class DEAPDataProcessor:
    def __init__(self, data_folder):
        """
        初始化DEAP数据处理器
        :param data_folder: 存放DEAP预处理数据的文件夹路径
        """
        self.data_folder = data_folder
        self.eeg_channels = np.array(
            ["Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1", "P3", "P7", "PO3", "O1", "Oz", "Pz",
             "Fp2", "AF4", "Fz", "F4", "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8", "PO4", "O2"])
        self.peripheral_channels = np.array(
            ["hEOG", "vEOG", "zEMG", "tEMG", "GSR", "Respiration belt", "Plethysmograph", "Temperature"])
        self.data = None
        self.labels = None
        self.eeg_data = None

    @staticmethod
    def _read_eeg_signal_from_file(filename):
        """
        从文件加载数据
        :param filename: 文件路径
        :return: 数据内容
        """
        x = pickle._Unpickler(open(filename, 'rb'))
        x.encoding = 'latin1'
        return x.load()

    def load_data(self):
        """
        加载所有被试者的data和label数据
        """
        files = [f"{n:02}" for n in range(1, 33)]  # 构建被试编号
        data = []
        labels = []

        for file in files:
            filename = f"{self.data_folder}/s{file}.dat"
            trial = self._read_eeg_signal_from_file(filename)
            labels.append(trial['labels'])
            data.append(trial['data'])

        # 将data和labels转为数组
        self.labels = np.array(labels).reshape(1280, 4)
        self.data = np.array(data).reshape(1280, 40, 8064)

    def split_channels(self):
        """
        将数据分为EEG通道数据和外围通道数据
        """
        eeg_data = self.data[:, :32, :]  # 前 32 个通道为 EEG 数据
        peripheral_data = self.data[:, 32:, :]  # 后 8 个通道为外围信号
        self.eeg_data = eeg_data
        self.peripheral_data = peripheral_data

    def get_eeg_data(self):
        """
        获取 EEG 通道数据
        """
        return self.eeg_data

    def get_peripheral_data(self):
        """
        获取外围通道数据
        """
        return self.peripheral_data

    def preprocess_eeg_data(self):
        """
        根据论文要求预处理EEG数据：
        - 剔除前3秒的数据
        - 无重叠滑窗分割后30秒数据
        :return: 滑窗分割后的EEG数据和对应标签
        """
        # 剔除前3秒数据，只保留后30秒
        print(self.data.shape)
        eeg_data = self.data[:, :, 384:]  # 384 = 3秒 * 128Hz
        print(eeg_data.shape)
        labels = self.labels
        print(labels.shape)

        # 滑窗分割
        window_size = 384  # 3秒窗口
        step_size = 384    # 无重叠
        num_subjects, num_trials, num_timepoints = eeg_data.shape
        segments = []
        segment_labels = []

        for subject_idx in range(num_subjects):
            for trial_idx in range(num_trials):
                trial_data = eeg_data[subject_idx, trial_idx]
                trial_label = labels[subject_idx]

                for start in range(0, num_timepoints - window_size + 1, step_size):
                    segment = trial_data[:, start:start + window_size]
                    segments.append(segment)
                    segment_labels.append(trial_label)

        # 转换为 numpy 数组
        segments = np.array(segments)  # 形状: (总样本数, 通道数, 时间点数)
        segment_labels = np.array(segment_labels)  # 形状: (总样本数, 标签数)

        return segments, segment_labels

    def get_label_dataframe(self):
        """
        获取独热编码后的标签DataFrame
        :return: 标签的DataFrame
        """
        labels_encoded = [[
            1 if self.labels[i, 0] >= 5 else 0,
            1 if self.labels[i, 1] >= 5 else 0
        ] for i in range(len(self.labels))]

        labels_encoded = np.array(labels_encoded).reshape(len(self.labels), 2)
        return pd.DataFrame(data=labels_encoded, columns=["Positive Valence", "High Arousal"])
