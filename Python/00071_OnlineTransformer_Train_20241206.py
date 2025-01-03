import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from sklearn.preprocessing import StandardScaler
import json
from torch.utils.data import TensorDataset, DataLoader

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f'Using GPU: {torch.cuda.get_device_name(0)}')
else:
    print("Using CPU")

# 从 config.json 文件中读取路径
config_path = 'config.json'  # 配置文件的路径
with open(config_path, 'r', encoding='utf-8') as config_file:
    config = json.load(config_file)

train_data_path = config['train_data_path']
test_data_path = config['test_data_path']
targets_column = config['X_true_column']
observations_column = config['X_observations_column']

train_base_name = os.path.splitext(os.path.basename(train_data_path))[0]
script_dir = os.path.dirname(os.path.abspath(__file__))

model_save_path = os.path.join(script_dir, f'{train_base_name}_OnlineTransformer.pth')
prediction_save_path = os.path.join(script_dir, f'{train_base_name}_OnlineTransformer.csv')
scaler_save_path_X = os.path.join(script_dir, f'{train_base_name}_OnlineTransformer_scaler_X.gz')
scaler_save_path_y = os.path.join(script_dir, f'{train_base_name}_OnlineTransformer_scaler_y.gz')

# 定义位置编码类
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_size=1, d_model=64, hidden_size=128, num_layers=2, nhead=8, output_size=1, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.input_fc = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, src):
        src = self.input_fc(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output[-1, :, :]  # 取最后一个时间步的输出
        output = self.decoder(output)
        return output

# 加载数据
train_df = pd.read_csv(train_data_path, header=None, sep='\s+')
train_observations = train_df.iloc[:, observations_column].values 
train_targets = train_df.iloc[:, targets_column].values

# 对数据进行归一化
scaler_X = StandardScaler()
scaler_y = StandardScaler()

train_observations = scaler_X.fit_transform(train_observations.reshape(-1, 1)).flatten()
train_targets = scaler_y.fit_transform(train_targets.reshape(-1, 1)).flatten()

# 保存归一化器
joblib.dump(scaler_X, scaler_save_path_X)
joblib.dump(scaler_y, scaler_save_path_y)

sequence_length = 25
X_train = []
y_train = []

for i in range(len(train_observations) - sequence_length):
    X_train.append(train_observations[i:i+sequence_length])
    y_train.append(train_targets[i+sequence_length])

X_train = np.array(X_train)
y_train = np.array(y_train)

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(2).permute(1, 0, 2).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

# 定义模型、损失函数和优化器
model = TransformerModel().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

train_dataset = TensorDataset(X_train.permute(1,0,2), y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

test_df = pd.read_csv(test_data_path, header=None, sep='\s+')
test_observations = test_df.iloc[:, observations_column].values
test_targets = test_df.iloc[:, targets_column].values

test_observations = scaler_X.transform(test_observations.reshape(-1, 1)).flatten()
test_targets = scaler_y.transform(test_targets.reshape(-1, 1)).flatten()

X_test = []
y_test = []
for i in range(len(test_observations)-sequence_length):
    X_test.append(test_observations[i:i+sequence_length])
    y_test.append(test_targets[i+sequence_length])

X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(2).permute(1,0,2).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to(device)

num_epochs = 10
best_test_loss = float('inf')

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        loss = criterion(outputs.squeeze(), y_test.squeeze()).item()
    return loss, outputs

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.permute(1,0,2).to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch.squeeze())
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(1)
    train_loss = train_loss / len(train_loader.dataset)
    
    test_loss, test_outputs = evaluate(model, X_test, y_test)
    print(f'Epoch [{epoch+1}/{num_epochs}], Training Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
    
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        torch.save(model.state_dict(), model_save_path)
        predictions = test_outputs.cpu().numpy().flatten()
        y_test_cpu = y_test.cpu().numpy().flatten()
        predictions_inverse = scaler_y.inverse_transform(predictions.reshape(-1, 1)).flatten()
        y_test_inverse = scaler_y.inverse_transform(y_test_cpu.reshape(-1, 1)).flatten()
        prediction_df = pd.DataFrame({
            'True Value': y_test_inverse,
            'Predicted Value': predictions_inverse
        })
        prediction_df.to_csv(prediction_save_path, index=False)
        print(f'New best model saved with test loss: {best_test_loss:.4f}')

print("Training finished.")