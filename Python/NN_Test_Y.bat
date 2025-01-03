@echo off
:: 设置 Python 解释器的路径
set PYTHON_EXEC=python

:: 直接运行每个脚本
echo Running 00013_RNN_UseY_20241129.py...
%PYTHON_EXEC% 00013_RNN_UseY_20241129.py
if errorlevel 1 exit /b 1

echo Running 00023_LSTM_UseY_20241129.py...
%PYTHON_EXEC% 00023_LSTM_UseY_20241129.py
if errorlevel 1 exit /b 1

echo Running 00033_GRU_UseY_20241129.py...
%PYTHON_EXEC% 00033_GRU_UseY_20241129.py
if errorlevel 1 exit /b 1

echo Running 00043_TCN_UseY_20241129.py...
%PYTHON_EXEC% 00043_TCN_UseY_20241129.py
if errorlevel 1 exit /b 1

echo Running 00053_NeuralODE_UseY_20241129.py...
%PYTHON_EXEC% 00053_NeuralODE_UseY_20241129.py
if errorlevel 1 exit /b 1

echo Running 00063_Transformer_UseY_20241129.py...
%PYTHON_EXEC% 00063_Transformer_UseY_20241129.py
if errorlevel 1 exit /b 1

echo Running 00073_OnlineTransformer_UseY_20241206.py...
%PYTHON_EXEC% 00073_OnlineTransformer_UseY_20241206.py
if errorlevel 1 exit /b 1

echo All scripts executed successfully!
