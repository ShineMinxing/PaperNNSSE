@echo off
:: 设置 Python 解释器的路径
set PYTHON_EXEC=python

:: 直接运行每个脚本
echo Running 00012_RNN_UseX_20241129.py...
%PYTHON_EXEC% 00012_RNN_UseX_20241129.py
if errorlevel 1 exit /b 1

echo Running 00022_LSTM_UseX_20241129.py...
%PYTHON_EXEC% 00022_LSTM_UseX_20241129.py
if errorlevel 1 exit /b 1

echo Running 00032_GRU_UseX_20241129.py...
%PYTHON_EXEC% 00032_GRU_UseX_20241129.py
if errorlevel 1 exit /b 1

echo Running 00042_TCN_UseX_20241129.py...
%PYTHON_EXEC% 00042_TCN_UseX_20241129.py
if errorlevel 1 exit /b 1

echo Running 00052_NeuralODE_UseX_20241129.py...
%PYTHON_EXEC% 00052_NeuralODE_UseX_20241129.py
if errorlevel 1 exit /b 1

echo Running 00062_Transformer_UseX_20241129.py...
%PYTHON_EXEC% 00062_Transformer_UseX_20241129.py
if errorlevel 1 exit /b 1

echo Running 00072_OnlineTransformer_UseX_20241206.py...
%PYTHON_EXEC% 00072_OnlineTransformer_UseX_20241206.py
if errorlevel 1 exit /b 1

echo All scripts executed successfully!
