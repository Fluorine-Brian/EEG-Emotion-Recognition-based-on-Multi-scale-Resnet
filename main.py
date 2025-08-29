"""
MSRN
"""

from deap_preprocessor import DEAPDataProcessor
from EEG_Preprocessing_FeatureExtraction import EEGFeatureExtractor
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from MSRN_model import ms_resnet_34  # 导入你的 MSRN 模型
import numpy as np
import matplotlib.pyplot as plt

"""
超参数设置
"""
num_classes = 2
learning_rate = 1e-4
sampling_rate = 128
batch_size = 64
gamma = 0.5
step_size = 100
epoch = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor = DEAPDataProcessor(data_folder="C://EEG_Emotion_Project/DEAP/data_preprocessed_python/")
processor.load_data()
processor.split_channels()
# print(processor.labels.shape)
labels_encoded = processor.get_label_dataframe().values
# print(labels_encoded.shape)
eeg_data = processor.get_eeg_data()
# print(eeg_data.shape)
# eeg_data = processor.get_eeg_data()
# labels_encoded = processor.get_label_dataframe().values

eeg_data = eeg_data[:, :, 4224:]

# 滑窗分割
window_size = 384  # 3秒窗口，3840/10 = 384
step_size = 384  # 无重叠
num_subjects, num_trials, num_timepoints = eeg_data.shape  # 1280, 32, 3840
segments = []
segment_labels = []

for subject_idx in range(num_subjects):
    trial_data = eeg_data[subject_idx]
    trial_label = labels_encoded[subject_idx]
    # print(trial_data.shape)  # 应该是(32, 3840)
    # print(trial_label.shape)  # 应该是(2,)

    for start in range(0, num_timepoints - window_size + 1, step_size):
        # 提取滑动窗口数据，每个窗口数据大小是(32, 384)
        segment = trial_data[:, start:start + window_size]
        segments.append(segment)
        segment_labels.append(trial_label)

# 转换为 numpy 数组
segments = np.array(segments)  # 形状: (12800, 32, 384)
segment_labels = np.array(segment_labels)  # 形状: (12800, 2)
eeg_data = segments
labels_encoded = segment_labels

scaler = StandardScaler()
eeg_data = scaler.fit_transform(eeg_data.reshape(-1, eeg_data.shape[-1])).reshape(eeg_data.shape)
# print(eeg_data)
# print(labels_encoded.shape)

# # 提取 PCC 特征
# feature_extractor = EEGFeatureExtractor(eeg_data, sampling_rate=sampling_rate)
# features = feature_extractor.extract_PCC_features()
# print(features.shape)

# 提取 PLV 特征
feature_extractor = EEGFeatureExtractor(eeg_data, sampling_rate=sampling_rate)
features = feature_extractor.extract_PLV_features()
print(features.shape)
print(features)
#
# # 被试编号
# subjects = np.repeat(np.arange(32), 400)  # 每个被试有 40 个试验，共 32 个被试
#
# # 用于记录每次交叉验证的测试准确度
# all_accuracies = []
#
# # LOSO 交叉验证
# logo = LeaveOneGroupOut()
#
#
# # 训练模型函数，增加保存最佳模型的逻辑
# def train_model(model, train_loader, criterion, optimizer, scheduler, epochs, patience=10):
#     model.train()
#     epoch_losses = []
#     best_accuracy = 0.0  # 初始化最佳准确度
#     epochs_without_improvement = 0  # 记录连续多少个epoch没有改进
#     best_model_state = None  # 用于保存最佳模型
#
#     for epoch in range(epochs):
#         total_loss = 0
#         for X_batch, y_batch in train_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#
#             # 前向传播
#             outputs = model(X_batch)
#             loss = criterion(outputs, y_batch)
#
#             # 反向传播
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#
#         # 更新学习率
#         scheduler.step()
#         avg_loss = total_loss / len(train_loader)
#         epoch_losses.append(avg_loss)
#
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for X_batch, y_batch in test_loader:
#                 X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#
#                 outputs = model(X_batch)
#                 _, predicted = torch.max(outputs, 1)
#                 correct += (predicted == y_batch).sum().item()
#                 total += y_batch.size(0)
#
#         accuracy = correct / total
#
#         # 早停策略：如果准确度没有提高，则增加计数
#         if accuracy > best_accuracy:
#             best_accuracy = accuracy
#             epochs_without_improvement = 0  # 重置计数器
#             best_model_state = model.state_dict()  # 保存当前最好的模型
#         else:
#             epochs_without_improvement += 1
#
#         # 如果连续 patience 个 epoch 没有提高准确度，则提前结束训练
#         if epochs_without_improvement >= patience:
#             print(f"Early stopping at epoch {epoch + 1} due to no improvement.")
#             break
#
#         # 打印当前epoch的损失和准确度
#         print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Train Accuracy: {accuracy:.4f}")
#
#     # 恢复最好的模型
#     model.load_state_dict(best_model_state)
#     return epoch_losses
#
#
# # 测试模型函数
# def test_model(model, test_loader):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for X_batch, y_batch in test_loader:
#             X_batch, y_batch = X_batch.to(device), y_batch.to(device)
#
#             outputs = model(X_batch)
#             _, predicted = torch.max(outputs, 1)
#             correct += (predicted == y_batch).sum().item()
#             total += y_batch.size(0)
#
#     accuracy = correct / total
#     print(f"Test Accuracy: {accuracy}")
#     return accuracy
#
#
# # 主训练循环
# # 主训练循环
# for train_idx, test_idx in logo.split(features, labels_encoded, groups=subjects):
#     # 数据划分
#     X_train, X_test = features[train_idx], features[test_idx]
#     y_train, y_test = labels_encoded[train_idx, 1], labels_encoded[test_idx, 1]  # 使用 0:Valence 标签; 1:Arousal标签
#
#     # 转换为 PyTorch 张量
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.long)
#     y_test_tensor = torch.tensor(y_test, dtype=torch.long)
#
#     # 构造 DataLoader
#     train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
#     test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     # 初始化模型
#     model = ms_resnet_34().to(device)
#
#     # 设置优化器和调度器
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
#
#     # 损失函数
#     criterion = nn.CrossEntropyLoss()
#
#     print(f"Training on all subjects except subject {test_idx[0] / 400 + 1}")
#     # 训练模型，得到最佳模型
#     train_loss = train_model(model, train_loader, criterion, optimizer, scheduler, epochs=epoch)
#
#     # 测试模型
#     accuracy = test_model(model, test_loader)
#
#     # 保存准确度和模型参数
#     all_accuracies.append(accuracy)
#
# # 计算总体准确度和标准差
# overall_accuracy = np.mean(all_accuracies)
# accuracy_std = np.std(all_accuracies)
#
# print(f"Overall Accuracy: {overall_accuracy:.4f}")
# print(f"Standard Deviation: {accuracy_std:.4f}")
#
# # 绘制 LOSO 准确度分布
# plt.figure(figsize=(10, 6))
# plt.bar(range(1, 33), all_accuracies, color='blue', alpha=0.7)
# plt.axhline(overall_accuracy, color='red', linestyle='--', label=f"Mean Accuracy: {overall_accuracy:.4f}")
# plt.fill_between(range(1, 33),
#                  overall_accuracy - accuracy_std,
#                  overall_accuracy + accuracy_std,
#                  color='red',
#                  alpha=0.2,
#                  label=f"±1 Std Dev: {accuracy_std:.4f}")
# plt.title("LOSO Accuracy for Each Subject")
# plt.xlabel("Subject")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.show()
