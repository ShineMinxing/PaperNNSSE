import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torchdiffeq
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
train_base_name = os.path.splitext(os.path.basename(train_data_path))[0]
script_dir = os.path.dirname(os.path.abspath(__file__))

# 将保存路径设置为脚本文件所在的同一目录
model_save_path = os.path.join(script_dir, f'{train_base_name}_NeuralODE.pth')
prediction_save_path = os.path.join(script_dir, f'{train_base_name}Y_NeuralODE.csv')
scaler_save_path_X = os.path.join(script_dir, f'{train_base_name}_NeuralODE_scaler_X.gz')
scaler_save_path_y = os.path.join(script_dir, f'{train_base_name}_NeuralODE_scaler_y.gz')

# 定义神经 ODE 模型类（与训练时相同）
class ODEFunc(nn.Module):
    def __init__(self, hidden_size, input_size):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size + input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.inputs = None  # 存储输入序列

    def forward(self, t, h):
        # 获取对应时间步的输入
        idx = min(int(t.item() * (sequence_length - 1)), sequence_length - 1)
        x_t = self.inputs[:, idx, :]  # (batch_size, input_size)
        h_and_x = torch.cat([h, x_t], dim=1)  # (batch_size, hidden_size + input_size)
        return self.net(h_and_x)

class NeuralODEModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralODEModel, self).__init__()
        self.input_fc = nn.Linear(input_size, hidden_size)
        self.ode_func = ODEFunc(hidden_size, input_size)
        self.output_fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: (batch_size, sequence_length, feature_size)
        batch_size = x.size(0)
        sequence_length = x.size(1)
        self.ode_func.inputs = x  # 将输入序列传递给 ODEFunc
        x0 = self.input_fc(x[:, 0, :])  # 初始条件
        t = torch.linspace(0, 1, steps=sequence_length).to(device)

        # x0 形状：(batch_size, hidden_size)
        # t 形状：(sequence_length,)
        out = torchdiffeq.odeint(self.ode_func, x0, t, method='rk4')
        # out 形状：(sequence_length, batch_size, hidden_size)
        out = out[-1]  # 取最后一个时间步的输出，形状：(batch_size, hidden_size)
        out = self.output_fc(out)  # (batch_size, output_size)
        return out

# 加载模型
input_size = 1
hidden_size = 50
output_size = 1
sequence_length = 25  # 与训练时相同

model = NeuralODEModel(input_size, hidden_size, output_size).to(device)
model.load_state_dict(torch.load(model_save_path, map_location=device))
model.eval()

# 加载归一化器
scaler_X = joblib.load(scaler_save_path_X)
scaler_y = joblib.load(scaler_save_path_y)

# 加载测试数据
test_df = pd.read_csv(estimate_data_path, header=None, sep='\s+')
test_observations = test_df.iloc[:, observations_column].values # 第四列：当前观测量
test_targets = test_df.iloc[:, targets_column].values      # 第二列：2步后的真实数据

# 对测试数据进行归一化
test_observations_normalized = scaler_X.transform(test_observations.reshape(-1, 1)).flatten()
test_targets_normalized = scaler_y.transform(test_targets.reshape(-1, 1)).flatten()

# 构建输入序列
X_test = []
y_test = []

for i in range(len(test_observations_normalized) - sequence_length):
    X_test.append(test_observations_normalized[i:i + sequence_length])
    y_test.append(test_targets_normalized[i + sequence_length])

X_test = np.array(X_test)
y_test = np.array(y_test)

# 转换为张量并移动到设备
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(device)  # (样本数, 序列长度, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)

# 进行预测
with torch.no_grad():
    outputs = model(X_test_tensor)
    test_loss = nn.MSELoss()(outputs.squeeze(), y_test_tensor).item()

print(f'Test Loss: {test_loss:.4f}')

# 反归一化预测结果
predictions_normalized = outputs.cpu().numpy().flatten()
y_test_normalized = y_test_tensor.cpu().numpy().flatten()

predictions = scaler_y.inverse_transform(predictions_normalized.reshape(-1, 1)).flatten()
y_test_original = scaler_y.inverse_transform(y_test_normalized.reshape(-1, 1)).flatten()

# 保存预测结果到 CSV 文件
prediction_df = pd.DataFrame({
    'True Value': y_test_original,
    'Predicted Value': predictions
})
prediction_df.to_csv(prediction_save_path, index=False)

print(f'Prediction results saved to {prediction_save_path}')
