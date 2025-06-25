import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
config_path = 'config.json'
with open(config_path, 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

train_data_path = config['train_data_path']
estimate_data_path = config['estimate_data_path']
targets_column = config['X_true_column']
observations_column = config['X_observations_column']

train_base_name = os.path.splitext(os.path.basename(train_data_path))[0]
script_dir = os.path.dirname(os.path.abspath(__file__))
model_save_path = os.path.join(script_dir, f'{train_base_name}_OnlineTransformer.pth')
prediction_save_path = os.path.join(script_dir, f'{train_base_name}X_OnlineTransformer.csv')
scaler_save_path_X = os.path.join(script_dir, f'{train_base_name}_OnlineTransformer_scaler_X.gz')
scaler_save_path_y = os.path.join(script_dir, f'{train_base_name}_OnlineTransformer_scaler_y.gz')


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model, device=device)
        position = torch.arange(0, max_len, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=64, hidden_size=128, num_layers=2, nhead=8, output_size=1, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_fc = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, src):
        # src: (seq_len, batch_size, 1)
        src = self.input_fc(src)  # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[-1, :, :]  # 取最后一个时间步的输出 (batch_size, d_model)
        output = self.decoder(output) # (batch_size, 1)
        return output

# 加载模型和归一化器
model = TransformerModel().to(device)
model.load_state_dict(torch.load(model_save_path))
model.eval()

scaler_X = joblib.load(scaler_save_path_X)
scaler_y = joblib.load(scaler_save_path_y)

# 加载测试数据
test_df = pd.read_csv(estimate_data_path, header=None, sep='\s+')
test_observations = test_df.iloc[:, observations_column].values
test_targets = test_df.iloc[:, targets_column].values

# 定义在线更新函数
def online_update(model, optimizer, input_seq, target_value, sequence_length=25, epochs=1):
    """
    对模型进行一次在线更新。
    input_seq: shape (25,) 原始未归一化输入序列
    target_value: 标量，对应的目标值
    """
    model.train()
    criterion = nn.MSELoss()

    # 归一化输入和目标
    input_norm = scaler_X.transform(input_seq.reshape(-1, 1)).flatten()  # (25,)
    target_norm = scaler_y.transform(np.array(target_value).reshape(-1,1)).flatten() # (1,)

    # 转换为tensor并移动设备
    X_new = torch.tensor(input_norm, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(2)  # (25,1,1)
    # 模型期望(seq_len, batch_size, feature_size)，已经是(25,1,1)
    y_new = torch.tensor(target_norm, dtype=torch.float32, device=device).unsqueeze(1) # (1,1)

    # 单样本训练，可以迭代多次epoch
    for ep in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_new)  # (1,1)
        loss = criterion(outputs.squeeze(), y_new.squeeze())
        loss.backward()
        optimizer.step()

    model.eval()

# 准备预测与在线更新的循环
# 假设我们要从某个起点开始循环预测，比如从 i_start 开始预测 i+3 的值
# 必须确保 i-27 >= 0 且 i+3 < len(test_targets), 并且保证有足够的数据
# 假设从 i=30 开始进行预测和在线更新 (根据你数据的实际长度调整)
i_start = 30
i_end = len(test_observations) - 3  # 确保 i+3 不超出范围

# 定义优化器，用于在线更新
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

predictions_list = []
true_values_list = []
indices_list = []

for i in range(i_start, i_end):
    # 在线更新数据：用 i-27 到 i-3 (共25步) 的数据来预测 i 的值，并进行在线更新
    if i-27 < 0:
        # 数据不够进行在线更新，跳过
        continue
    input_for_update = test_observations[i-27:i-2]  # i-27到i-3共25步输入
    target_for_update = test_targets[i]             # 对应输出为 i 时刻的值

    # 在线更新
    online_update(model, optimizer, input_for_update, target_for_update, sequence_length=25, epochs=1)

    # 在线更新后进行预测：用 i-24 到 i 的数据预测 i+3
    if i+3 >= len(test_observations):
        break
    input_for_prediction = test_observations[i-24:i+1]  # 共25步
    # 归一化预测输入
    input_pred_norm = scaler_X.transform(input_for_prediction.reshape(-1,1)).flatten()
    X_pred = torch.tensor(input_pred_norm, dtype=torch.float32, device=device).unsqueeze(1).unsqueeze(2)  # (25,1,1)

    with torch.no_grad():
        pred_output = model(X_pred)  # (1,1)
    pred_norm = pred_output.cpu().numpy().flatten()
    pred_value = scaler_y.inverse_transform(pred_norm.reshape(-1,1)).flatten()[0]

    true_value = test_targets[i+3]

    predictions_list.append(pred_value)
    true_values_list.append(true_value)
    indices_list.append(i+3)

# 将结果保存

# 保存预测结果到 CSV 文件
prediction_df = pd.DataFrame({
    'True Value': true_values_list,
    'Predicted Value': predictions_list
})
prediction_df.to_csv(prediction_save_path, index=False)

print(f'Prediction with online updates saved to {prediction_save_path}')
