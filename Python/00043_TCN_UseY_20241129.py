import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import StandardScaler
import json

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
else:
    print("Using CPU")

# 从 config.json 文件中读取路径
config_path = 'config.json'  # 配置文件的路径

# 读取配置文件
with open(config_path, 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

# 获取路径
train_data_path = config['train_data_path']
estimate_data_path = config['estimate_data_path']
targets_column = config['Y_true_column']
observations_column = config['Y_observations_column']

# 根据 train_data_path 自动生成模型和归一化器的保存路径
train_base_name = os.path.splitext(os.path.basename(train_data_path))[0]  # 获取训练集文件名（不含扩展名）
script_dir = os.path.dirname(os.path.abspath(__file__))

# 将保存路径设置为脚本文件所在的同一目录
model_save_path = os.path.join(script_dir, f'{train_base_name}_TCN.pth')
prediction_save_path = os.path.join(script_dir, f'{train_base_name}Y_TCN.csv')
scaler_save_path_X = os.path.join(script_dir, f'{train_base_name}_TCN_scaler_X.gz')
scaler_save_path_y = os.path.join(script_dir, f'{train_base_name}_TCN_scaler_y.gz')

# 定义 TCN 模型类（与训练时相同）
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self, x):
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                     dilation=dilation_size, padding=padding, dropout=dropout)]
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, input_size=1, output_size=1, num_channels=[32]*4, kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.fc = nn.Linear(num_channels[-1], output_size)
    
    def forward(self, x):
        # x 的形状: (batch_size, channels, seq_length)
        y = self.tcn(x)
        y = y[:, :, -1]  # 取最后一个时间步
        y = self.fc(y)
        return y

# 加载模型
model = TCNModel().to(device)
model.load_state_dict(torch.load(model_save_path))
model.eval()

# 加载归一化器
scaler_X = joblib.load(scaler_save_path_X)
scaler_y = joblib.load(scaler_save_path_y)

# 加载测试数据
test_df = pd.read_csv(estimate_data_path, header=None, sep='\s+')
test_observations = test_df.iloc[:, observations_column].values
test_targets = test_df.iloc[:, targets_column].values

# 对测试数据进行归一化
test_observations = scaler_X.transform(test_observations.reshape(-1, 1)).flatten()
test_targets = scaler_y.transform(test_targets.reshape(-1, 1)).flatten()

# 构建测试数据的输入序列
sequence_length = 25
X_test = []
y_test = []

for i in range(len(test_observations) - sequence_length):
    X_test.append(test_observations[i:i + sequence_length])
    y_test.append(test_targets[i + sequence_length])

X_test = np.array(X_test)
y_test = np.array(y_test)

# 转换为 PyTorch 张量并移动到设备
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)  # (样本数, 1, 序列长度)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# 进行预测
with torch.no_grad():
    outputs = model(X_test)
    test_loss = nn.MSELoss()(outputs.squeeze(), y_test).item()

print(f'Test Loss: {test_loss:.4f}')

# 反归一化预测结果
predictions = outputs.cpu().numpy().flatten()
y_test_cpu = y_test.cpu().numpy().flatten()

predictions_inverse = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
y_test_inverse = scaler_y.inverse_transform(y_test_cpu.reshape(-1, 1)).flatten()

# 保存预测结果到 CSV 文件
prediction_df = pd.DataFrame({
    'True Value': y_test_inverse,
    'Predicted Value': predictions_inverse
})
prediction_df.to_csv(prediction_save_path, index=False)

print(f'Prediction results saved to {prediction_save_path}')
