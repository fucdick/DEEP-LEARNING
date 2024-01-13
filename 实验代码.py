import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt  # Added for visualization

# 定义类别标签映射
label_dict = {'aloe': 0, 'burger': 1, 'cabbage': 2, 'candied_fruits': 3, 'carrots': 4, 'chips': 5,
              'chocolate': 6, 'drinks': 7, 'fries': 8, 'grapes': 9, 'gummies': 10, 'ice-cream': 11,
              'jelly': 12, 'noodles': 13, 'pickles': 14, 'pizza': 15, 'ribs': 16, 'salmon': 17,
              'soup': 18, 'wings': 19}
label_dict_inv = {v: k for k, v in label_dict.items()}
print(label_dict_inv)


# 定义音频数据集类
class AudioDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = self.features[idx]
        sample = sample.squeeze(0) if sample.size(0) == 1 else sample
        sample = sample.squeeze(1) if sample.size(1) == 1 else sample
        return sample, self.labels[idx]


# 从Librosa获取特征和标签
def extract_features_and_labels(parent_dir, sub_dirs, max_file=10, max_audio_length=500):
    features, labels = [], []
    for sub_dir in sub_dirs:
        for fn in tqdm(glob.glob(os.path.join(parent_dir, sub_dir, "*.wav"))[:max_file]):
            label_name = os.path.basename(os.path.dirname(fn))
            labels.append(label_dict[label_name])

            waveform, sample_rate = torchaudio.load(fn)
            mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)

            if mel_spec.shape[2] > max_audio_length:
                mel_spec = mel_spec[:, :, :max_audio_length]
            else:
                mel_spec = torch.nn.functional.pad(mel_spec, (0, max_audio_length - mel_spec.shape[2]))

            features.append(mel_spec)

    return features, labels


# 加载训练数据
parent_dir = './train_sample/train_sample/'
sub_dirs = list(label_dict.keys())
print(sub_dirs)
X, Y = extract_features_and_labels(parent_dir, sub_dirs, max_file=100)

# 转换特征和标签为PyTorch张量
features_tensor = torch.stack(X)
labels_tensor = torch.tensor(Y)
print(features_tensor.shape)
print(labels_tensor.shape)

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(features_tensor, labels_tensor, random_state=1,
                                                    stratify=labels_tensor)
print('训练集的大小', len(X_train))
print('测试集的大小', len(X_test))

# 构建数据加载器
train_dataset = AudioDataset(X_train, Y_train)
test_dataset = AudioDataset(X_test, Y_test)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)


# 定义音频分类模型
class AudioCNN(nn.Module):
    def __init__(self):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(128 * 32 * 125, 512)
        self.fc2 = nn.Linear(512, 20)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))
        x = self.pool(torch.tanh(self.conv2(x)))

        x = self.dropout(x)
        x = x.view(-1, 128 * 32 * 125)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


# 初始化模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} for training")
model = AudioCNN().to(device)
model.load_state_dict(torch.load('cnn.pth'))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 30
train_loss_list = []  # Added for visualization
train_accuracy_list = []  # Added for visualization

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in train_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(1))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = correct / total
    train_loss_list.append(running_loss / len(train_loader))
    train_accuracy_list.append(accuracy)

    torch.save(model.state_dict(), 'cnn.pth')
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy}")


# Visualize the training process
def plot_loss_accuracy(train_loss_list, train_accuracy_list):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracy_list, label='Training Accuracy', color='orange')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


plot_loss_accuracy(train_loss_list, train_accuracy_list)


# 测试模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs.unsqueeze(1))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy on test set: {(correct / total) * 100:.2f}%")


# 提取测试集特征
def extract_features_test(test_dir, max_audio_length=500):
    features = []
    for fn in tqdm(glob.glob(os.path.join(test_dir, "*.wav"))):
        waveform, sample_rate = torchaudio.load(fn)
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)

        if mel_spec.shape[2] > max_audio_length:
            mel_spec = mel_spec[:, :, :max_audio_length]
        else:
            mel_spec = torch.nn.functional.pad(mel_spec, (0, max_audio_length - mel_spec.shape[2]))

        features.append(mel_spec)
    return features


# 加载并预测测试集
X_test = extract_features_test('./test_a/test_a/')
X_test_tensor = torch.stack(X_test)
predictions = []

with torch.no_grad():
    model.eval()
    for test_input in X_test_tensor:
        test_input = test_input.to(device)
        test_output = model(test_input.unsqueeze(0))
        _, predicted = torch.max(test_output, 1)
        predictions.append(predicted.item())

# 保存预测结果到CSV文件
predicted_labels = [label_dict_inv[pred] for pred in predictions]
result_df = pd.DataFrame(
    {'name': [os.path.basename(file) for file in glob.glob('./test_a/test_a/*.wav')], 'label': predicted_labels})
result_df.to_csv('submit.csv', index=False)
