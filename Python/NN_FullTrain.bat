@echo off
:: 设置 Python 解释器的路径
set PYTHON_EXEC=python

:: 直接运行每个脚本
echo Running 00014_RNN_FullTrain_20241129.py...
%PYTHON_EXEC% 00014_RNN_FullTrain_20241129.py
if errorlevel 1 exit /b 1

echo Running 00024_LSTM_FullTrain_20241129.py...
%PYTHON_EXEC% 00024_LSTM_FullTrain_20241129.py
if errorlevel 1 exit /b 1

echo Running 00034_GRU_FullTrain_20241129.py...
%PYTHON_EXEC% 00034_GRU_FullTrain_20241129.py
if errorlevel 1 exit /b 1

echo Running 00044_TCN_FullTrain_20241129.py...
%PYTHON_EXEC% 00044_TCN_FullTrain_20241129.py
if errorlevel 1 exit /b 1

echo Running 00054_NeuralODE_FullTrain_20241129.py...
%PYTHON_EXEC% 00054_NeuralODE_FullTrain_20241129.py
if errorlevel 1 exit /b 1

echo Running 00064_Transformer_FullTrain_20241129.py...
%PYTHON_EXEC% 00064_Transformer_FullTrain_20241129.py
if errorlevel 1 exit /b 1

echo Running 00074_OnlineTransformer_Train_20241206.py...
%PYTHON_EXEC% 00071_OnlineTransformer_Train_20241206.py
if errorlevel 1 exit /b 1

echo All scripts executed successfully!
