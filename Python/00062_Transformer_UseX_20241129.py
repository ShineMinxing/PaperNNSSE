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
config_path = 'config.json'  # 配置文件的路径

# 读取配置文件
with open(config_path, 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

# 获取路径
train_data_path = config['train_data_path']
estimate_data_path = config['estimate_data_path']
targets_column = config['X_true_column']
observations_column = config['X_observations_column']

# 根据 train_data_path 自动生成模型和归一化器的保存路径
train_base_name = os.path.splitext(os.path.basename(train_data_path))[0]  # 获取训练集文件名（不含扩展名）
script_dir = os.path.dirname(os.path.abspath(__file__))

# 将保存路径设置为脚本文件所在的同一目录
model_save_path = os.path.join(script_dir, f'{train_base_name}_Transformer.pth')
prediction_save_path = os.path.join(script_dir, f'{train_base_name}X_Transformer.csv')
scaler_save_path_X = os.path.join(script_dir, f'{train_base_name}_Transformer_scaler_X.gz')
scaler_save_path_y = os.path.join(script_dir, f'{train_base_name}_Transformer_scaler_y.gz')

# 定义 Transformer 模型类（与训练时相同）
class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=64, hidden_size=128, num_layers=2, nhead=8, output_size=1, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.input_size = input_size
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_fc = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, src):
        # src shape: (seq_len, batch_size, feature_size)
        src = self.input_fc(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[-1, :, :]  # 取最后一个时间步的输出
        output = self.decoder(output)
        return output

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
        pe = pe.unsqueeze(1)  # (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 加载模型
model = TransformerModel().to(device)
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
test_observations_normalized = scaler_X.transform(test_observations.reshape(-1, 1)).flatten()
test_targets_normalized = scaler_y.transform(test_targets.reshape(-1, 1)).flatten()

# 构建输入序列
sequence_length = 25
X_test = []
y_test = []

for i in range(len(test_observations_normalized) - sequence_length):
    X_test.append(test_observations_normalized[i:i + sequence_length])
    y_test.append(test_targets_normalized[i + sequence_length])

X_test = np.array(X_test)
y_test = np.array(y_test)

# 转换为张量并移动到设备
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2).permute(1, 0, 2).to(device)  # (seq_len, batch_size, feature_size)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)  # (batch_size, 1)

# 进行预测
with torch.no_grad():
    outputs = model(X_test_tensor)
    test_loss = nn.MSELoss()(outputs.squeeze(), y_test_tensor.squeeze()).item()

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
