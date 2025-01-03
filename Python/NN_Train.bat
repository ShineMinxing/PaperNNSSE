@echo off
:: 设置 Python 解释器的路径
set PYTHON_EXEC=python

:: 直接运行每个脚本
echo Running 00011_RNN_Train_20241129.py...
%PYTHON_EXEC% 00011_RNN_Train_20241129.py
if errorlevel 1 exit /b 1

echo Running 00021_LSTM_Train_20241129.py...
%PYTHON_EXEC% 00021_LSTM_Train_20241129.py
if errorlevel 1 exit /b 1

echo Running 00031_GRU_Train_20241129.py...
%PYTHON_EXEC% 00031_GRU_Train_20241129.py
if errorlevel 1 exit /b 1

echo Running 00041_TCN_Train_20241129.py...
%PYTHON_EXEC% 00041_TCN_Train_20241129.py
if errorlevel 1 exit /b 1

echo Running 00051_NeuralODE_Train_20241129.py...
%PYTHON_EXEC% 00051_NeuralODE_Train_20241129.py
if errorlevel 1 exit /b 1

echo Running 00061_Transformer_Train_20241129.py...
%PYTHON_EXEC% 00061_Transformer_Train_20241129.py
if errorlevel 1 exit /b 1

echo Running 00071_OnlineTransformer_Train_20241206.py...
%PYTHON_EXEC% 00071_OnlineTransformer_Train_20241206.py
if errorlevel 1 exit /b 1

echo All scripts executed successfully!
