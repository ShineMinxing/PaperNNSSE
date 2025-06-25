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
test_data_path = config['test_data_path']
targets_column = config['X_true_column']
observations_column = config['X_observations_column']

# 根据 train_data_path 自动生成模型和归一化器的保存路径
train_base_name = os.path.splitext(os.path.basename(train_data_path))[0]
script_dir = os.path.dirname(os.path.abspath(__file__))

# 将保存路径设置为脚本文件所在的同一目录
model_save_path = os.path.join(script_dir, f'{train_base_name}_NeuralODE.pth')
prediction_save_path = os.path.join(script_dir, f'{train_base_name}_NeuralODE.csv')
scaler_save_path_X = os.path.join(script_dir, f'{train_base_name}_NeuralODE_scaler_X.gz')
scaler_save_path_y = os.path.join(script_dir, f'{train_base_name}_NeuralODE_scaler_y.gz')

# 加载训练数据
train_df = pd.read_csv(train_data_path, header=None, sep='\s+')
train_observations = train_df.iloc[:, observations_column].values # 第四列：当前观测量
train_targets = train_df.iloc[:, targets_column].values      # 第二列：2步后的真实数据

# 对数据进行归一化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

train_observations = scaler_X.fit_transform(train_observations.reshape(-1, 1)).flatten()
train_targets = scaler_y.fit_transform(train_targets.reshape(-1, 1)).flatten()

# 保存归一化器
joblib.dump(scaler_X, scaler_save_path_X)
joblib.dump(scaler_y, scaler_save_path_y)

# 构建输入序列和对应的目标值
sequence_length = 25
X_train = []
y_train = []

for i in range(len(train_observations) - sequence_length):
    X_train.append(train_observations[i:i + sequence_length])
    y_train.append(train_targets[i + sequence_length])  # 预测2步后的数据

X_train = np.array(X_train)
y_train = np.array(y_train)

# 转换为 PyTorch 张量并移动到设备
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)  # (样本数, 序列长度, 1)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)

# 创建数据加载器
from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 加载测试数据
test_df = pd.read_csv(test_data_path, header=None, sep='\s+')
test_observations = test_df.iloc[:, observations_column].values # 第四列：当前观测量
test_targets = test_df.iloc[:, targets_column].values      # 第二列：2步后的真实数据

# 对测试数据进行与训练数据相同的归一化
test_observations = scaler_X.transform(test_observations.reshape(-1, 1)).flatten()
test_targets = scaler_y.transform(test_targets.reshape(-1, 1)).flatten()

# 构建测试数据的输入序列
X_test = []
y_test = []
for i in range(len(test_observations) - sequence_length):
    X_test.append(test_observations[i:i + sequence_length])
    y_test.append(test_targets[i + sequence_length])

X_test = np.array(X_test)
y_test = np.array(y_test)

# 转换为 PyTorch 张量并移动到设备
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# 定义神经 ODE 模型（使用受控神经 ODE）
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

model = NeuralODEModel(input_size=1, hidden_size=50, output_size=1).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 初始化最佳测试损失
best_test_loss = float('inf')

# 训练模型
num_epochs = 50  # 增加训练轮数
for epoch in range(num_epochs):
    # 训练模式
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)

    train_loss = train_loss / len(train_loader.dataset)

    # 测试模式
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        test_loss = criterion(outputs.squeeze(), y_test).item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')

    # 检查并保存效果最好的模型和预测结果
    if test_loss < best_test_loss:
        best_test_loss = test_loss

        # 保存当前模型状态
        torch.save(model.state_dict(), model_save_path)

        # 保存预测结果（反归一化）
        predictions = outputs.cpu().numpy().flatten()
        y_test_cpu = y_test.cpu().numpy().flatten()

        # 反归一化
        predictions_inverse = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        y_test_inverse = scaler_y.inverse_transform(y_test_cpu.reshape(-1, 1)).flatten()

        prediction_df = pd.DataFrame({
            'True Value': y_test_inverse,
            'Predicted Value': predictions_inverse
        })
        prediction_df.to_csv(prediction_save_path, index=False)

        print(f'New best model saved with test loss: {best_test_loss:.4f}')
