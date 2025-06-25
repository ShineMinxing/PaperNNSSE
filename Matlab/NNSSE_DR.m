%% NNSSE for double reflector platform trace

close all; clear; clc;

% Configuration
DATA_ROWS = 4799;
READ_DATA_COLUMNS = 7;
WRITE_DATA_COLUMNS = 21;
State_Dimension = 4;
MODEL_DataWrite = [6,7;8,9;10,11;12,13;14,15;16,17;18,19;20,21;22,23;];

% Allocate space
ReadFileData = zeros(DATA_ROWS, READ_DATA_COLUMNS);
WriteFileData = zeros(DATA_ROWS, WRITE_DATA_COLUMNS);
EstimatedState = zeros(DATA_ROWS, State_Dimension); 

StateColumns = [2,3];
ObserColumns = [4,5];

% Read observation data
FullScriptPath = mfilename('fullpath');
[CurrentDir, ~, ~] = fileparts(FullScriptPath);
ParentDir = fileparts(CurrentDir);
INPUT_PATH = fullfile(ParentDir, 'DRData', ['Trace', num2str(DATA_ROWS), '.txt']);
ReadFileData = load(INPUT_PATH);

WriteFileData(:, 1) = ReadFileData(:, 1); % Timestamp
WriteFileData(:, 2:5) = ReadFileData(:, [StateColumns,ObserColumns]); % Timestamp
Time = [];

tic;
% Initialize estimator structure 1
EstiamtorIndex = 1;
disp(['Start Estiamtor ',num2str(EstiamtorIndex)]);
StateSpaceModel1_ = struct();
StateSpaceModel1_ = StateSpaceModel1(StateSpaceModel1_);
% Estimation
for i = 1:DATA_ROWS
    StateSpaceModel1_.CurrentObservation = ReadFileData(i, ObserColumns)';
    StateSpaceModel1_ = StateSpaceModel1_.EstimatorPort(StateSpaceModel1_);
    WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,:)) = StateSpaceModel1_.EstimatedState([1,4]);
end
Time = [Time,toc];
tic;


% Initialize estimator structure 2
EstiamtorIndex = 2;
disp(['Start Estiamtor ',num2str(EstiamtorIndex)]);
StateSpaceModel107_ = struct();
StateSpaceModel107_ = StateSpaceModel107(StateSpaceModel107_);
% Estimation
for i = 1:DATA_ROWS
    StateSpaceModel107_.CurrentObservation = ReadFileData(i, ObserColumns)';
    StateSpaceModel107_ = StateSpaceModel107_.EstimatorPort(StateSpaceModel107_);
    WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,:)) = StateSpaceModel107_.EstimatedState([1,4]);
end
Time = [Time,toc];
tic;


% Initialize estimator structure 3
% X Y data are seperately handled
EstiamtorIndex = 3;
disp(['Start Estiamtor ',num2str(EstiamtorIndex)]);
StateSpaceModel101_1 = struct();
StateSpaceModel101_1.CurrentObservation = ReadFileData(1, ObserColumns(1));
StateSpaceModel101_1 = StateSpaceModel101(StateSpaceModel101_1);
% Estimation
for i = 1:DATA_ROWS
    StateSpaceModel101_1.CurrentObservation = ReadFileData(i, ObserColumns(1))';
    StateSpaceModel101_1 = StateSpaceModel101_1.EstimatorPort(StateSpaceModel101_1);
    WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,1)) = StateSpaceModel101_1.PredictedState;
end
StateSpaceModel101_2 = struct();
StateSpaceModel101_2.CurrentObservation = ReadFileData(1, ObserColumns(2));
StateSpaceModel101_2 = StateSpaceModel101(StateSpaceModel101_2);
% Estimation
for i = 1:DATA_ROWS
    StateSpaceModel101_2.CurrentObservation = ReadFileData(i, ObserColumns(2))';
    StateSpaceModel101_2 = StateSpaceModel101_2.EstimatorPort(StateSpaceModel101_2);
    WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,2)) = StateSpaceModel101_2.PredictedState;
end
Time = [Time,toc];
tic;


% Initialize estimator structure 4
EstiamtorIndex = 4;
disp(['Start Estiamtor ',num2str(EstiamtorIndex)]);
StateSpaceModel102_1 = struct();
StateSpaceModel102_1.CurrentObservation = ReadFileData(1, ObserColumns(1));
StateSpaceModel102_1 = StateSpaceModel102(StateSpaceModel102_1);
% Estimation
for i = 1:DATA_ROWS
    StateSpaceModel102_1.CurrentObservation = ReadFileData(i, ObserColumns(1))';
    StateSpaceModel102_1 = StateSpaceModel102_1.EstimatorPort(StateSpaceModel102_1);
    WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,1)) = StateSpaceModel102_1.PredictedState;
    if any(isinf(WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,1))) | isnan(WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,1))), 'all')
        break;
    end
end
StateSpaceModel102_2 = struct();
StateSpaceModel102_2.CurrentObservation = ReadFileData(1, ObserColumns(2));
StateSpaceModel102_2 = StateSpaceModel102(StateSpaceModel102_2);
% Estimation
for i = 1:DATA_ROWS
    StateSpaceModel102_2.CurrentObservation = ReadFileData(i, ObserColumns(2))';
    StateSpaceModel102_2 = StateSpaceModel102_2.EstimatorPort(StateSpaceModel102_2);
    WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,2)) = StateSpaceModel102_2.PredictedState;
    if any(isinf(WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,2))) | isnan(WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,2))), 'all')
        break;
    end
end
Time = [Time,toc];
tic;


% Initialize estimator structure 5
EstiamtorIndex = 5;
disp(['Start Estiamtor ',num2str(EstiamtorIndex)]);
StateSpaceModel103_1 = struct();
StateSpaceModel103_1.CurrentObservation = ReadFileData(1, ObserColumns(1));
StateSpaceModel103_1 = StateSpaceModel103(StateSpaceModel103_1);
% Estimation
for i = 1:DATA_ROWS
    StateSpaceModel103_1.CurrentObservation = ReadFileData(i, ObserColumns(1))';
    StateSpaceModel103_1 = StateSpaceModel103_1.EstimatorPort(StateSpaceModel103_1);
    WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,1)) = StateSpaceModel103_1.PredictedState;
end
StateSpaceModel103_2 = struct();
StateSpaceModel103_2.CurrentObservation = ReadFileData(1, ObserColumns(2));
StateSpaceModel103_2 = StateSpaceModel103(StateSpaceModel103_2);
% Estimation
for i = 1:DATA_ROWS
    StateSpaceModel103_2.CurrentObservation = ReadFileData(i, ObserColumns(2))';
    StateSpaceModel103_2 = StateSpaceModel103_2.EstimatorPort(StateSpaceModel103_2);
    WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,2)) = StateSpaceModel103_2.PredictedState;
end
Time = [Time,toc];
tic;


% Initialize estimator structure 6
EstiamtorIndex = 6;
disp(['Start Estiamtor ',num2str(EstiamtorIndex)]);
StateSpaceModel104_1 = struct();
StateSpaceModel104_1.CurrentObservation = ReadFileData(1, ObserColumns(1));
StateSpaceModel104_1 = StateSpaceModel104(StateSpaceModel104_1);
% Estimation
for i = 1:DATA_ROWS
    StateSpaceModel104_1.CurrentObservation = ReadFileData(i, ObserColumns(1))';
    StateSpaceModel104_1 = StateSpaceModel104_1.EstimatorPort(StateSpaceModel104_1);
    WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,1)) = StateSpaceModel104_1.PredictedState;
end
StateSpaceModel104_2 = struct();
StateSpaceModel104_2.CurrentObservation = ReadFileData(1, ObserColumns(2));
StateSpaceModel104_2 = StateSpaceModel104(StateSpaceModel104_2);
% Estimation
for i = 1:DATA_ROWS
    StateSpaceModel104_2.CurrentObservation = ReadFileData(i, ObserColumns(2))';
    StateSpaceModel104_2 = StateSpaceModel104_2.EstimatorPort(StateSpaceModel104_2);
    WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,2)) = StateSpaceModel104_2.PredictedState;
end
Time = [Time,toc];
tic;


% Initialize estimator structure 7
EstiamtorIndex = 7;
disp(['Start Estiamtor ',num2str(EstiamtorIndex)]);
StateSpaceModel105_1 = struct();
StateSpaceModel105_1.CurrentObservation = ReadFileData(1, ObserColumns(1));
StateSpaceModel105_1 = StateSpaceModel105(StateSpaceModel105_1);
% Estimation
for i = 1:DATA_ROWS
    StateSpaceModel105_1.CurrentObservation = ReadFileData(i, ObserColumns(1))';
    StateSpaceModel105_1 = StateSpaceModel105_1.EstimatorPort(StateSpaceModel105_1);
    WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,1)) = StateSpaceModel105_1.PredictedState;
end
StateSpaceModel105_2 = struct();
StateSpaceModel105_2.CurrentObservation = ReadFileData(1, ObserColumns(2));
StateSpaceModel105_1 = StateSpaceModel105(StateSpaceModel105_1);
% Estimation
for i = 1:DATA_ROWS
    StateSpaceModel105_1.CurrentObservation = ReadFileData(i, ObserColumns(2))';
    StateSpaceModel105_1 = StateSpaceModel105_1.EstimatorPort(StateSpaceModel105_1);
    WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,2)) = StateSpaceModel105_1.PredictedState;
end
Time = [Time,toc];
tic;


% Initialize estimator structure 8
EstiamtorIndex = 8;
disp(['Start Estiamtor ',num2str(EstiamtorIndex)]);
StateSpaceModel106_1 = struct();
StateSpaceModel106_1.CurrentObservation = ReadFileData(1, ObserColumns(1));
StateSpaceModel106_1 = StateSpaceModel106(StateSpaceModel106_1);
% Estimation
for i = 1:DATA_ROWS
    StateSpaceModel106_1.CurrentObservation = ReadFileData(i, ObserColumns(1))';
    StateSpaceModel106_1 = StateSpaceModel106_1.EstimatorPort(StateSpaceModel106_1);
    WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,1)) = StateSpaceModel106_1.PredictedState;
end
StateSpaceModel106_2 = struct();
StateSpaceModel106_2.CurrentObservation = ReadFileData(1, ObserColumns(2));
StateSpaceModel106_2 = StateSpaceModel106(StateSpaceModel106_2);
% Estimation
for i = 1:DATA_ROWS
    StateSpaceModel106_2.CurrentObservation = ReadFileData(i, ObserColumns(2))';
    StateSpaceModel106_2 = StateSpaceModel106_2.EstimatorPort(StateSpaceModel106_2);
    WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,2)) = StateSpaceModel106_2.PredictedState;
end
Time = [Time,toc];
tic;

% Initialize estimator structure 9
EstiamtorIndex = 9;
disp(['Start Estiamtor ',num2str(EstiamtorIndex)]);
StateSpaceModel108_1 = struct();
StateSpaceModel108_1.CurrentObservation = ReadFileData(1, ObserColumns(1));
StateSpaceModel108_1 = StateSpaceModel108(StateSpaceModel108_1);
% Estimation
for i = 1:DATA_ROWS
    StateSpaceModel108_1.CurrentObservation = ReadFileData(i, ObserColumns(1))';
    StateSpaceModel108_1 = StateSpaceModel108_1.EstimatorPort(StateSpaceModel108_1);
    WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,1)) = StateSpaceModel108_1.PredictedState;
end
StateSpaceModel108_2 = struct();
StateSpaceModel108_2.CurrentObservation = ReadFileData(1, ObserColumns(2));
StateSpaceModel108_2 = StateSpaceModel108(StateSpaceModel108_2);
% Estimation
for i = 1:DATA_ROWS
    StateSpaceModel108_2.CurrentObservation = ReadFileData(i, ObserColumns(2))';
    StateSpaceModel108_2 = StateSpaceModel108_2.EstimatorPort(StateSpaceModel108_2);
    WriteFileData(i, MODEL_DataWrite(EstiamtorIndex,2)) = StateSpaceModel108_2.PredictedState;
end
Time = [Time,toc];

Time

% Output Results
fullOutputFilePath = fullfile(CurrentDir,'EstimationResult/Data_1/EstimationResult_.txt');
fileID = fopen(fullOutputFilePath, 'w');
if fileID == -1
    error('Unable to create file: %s', fullOutputFilePath);
end
[dataRows, dataCols] = size(WriteFileData);
for i = 1:dataRows
    fprintf(fileID, '%.6f ', WriteFileData(i, 1:dataCols));
    fprintf(fileID, '\n');
end
fclose(fileID);
fprintf('Estimation data has been written to %s\n', fullOutputFilePath);
fullOutputFilePath = fullfile(CurrentDir,'EstimationResult/Data_2/EstimationResult_.txt');
fileID = fopen(fullOutputFilePath, 'w');
if fileID == -1
    error('Unable to create file: %s', fullOutputFilePath);
end
[dataRows, dataCols] = size(WriteFileData);
for i = 1:dataRows
    fprintf(fileID, '%.6f ', WriteFileData(i, 1:dataCols));
    fprintf(fileID, '\n');
end
fclose(fileID);
fprintf('Estimation data has been written to %s\n', fullOutputFilePath);

% Draw Picture
PLOTLINE = 1:DATA_ROWS;

figure(1)
hold on;grid on;
plot(WriteFileData(PLOTLINE, 2),'k')
plot(WriteFileData(PLOTLINE, 4),'ko')
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(1,1)),'b')
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(2,1)), 'color', [0.6350 0.0780 0.1840])
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(3,1)),'r')
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(4,1)),'g')
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(5,1)),'c')
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(6,1)),'m')
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(7,1)), 'color', [0.8500 0.3250 0.0980])
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(8,1)), 'color', [0.4660 0.6740 0.1880])
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(9,1)), 'color', [0.6350 0.0780 0.1840])
ylim([1700,2600])
legend('True trace', 'Observation', 'UAM LKE', 'UAM UKE', 'NNSSE UKE', 'NNSSE PE', 'NNSSE EKE', 'NNSSE 551', 'NNSSE 10101', 'NNSSE Tanh', 'NNSSE 5551')
title('Yaw')

figure(2)
hold on;grid on;
plot(WriteFileData(PLOTLINE, 3),'k')
plot(WriteFileData(PLOTLINE, 5),'ko')
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(1,2)), 'b')
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(2,2)), 'color', [0.6350 0.0780 0.1840])
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(3,2)), 'r')
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(4,2)), 'g')
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(5,2)), 'c')
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(6,2)), 'm')
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(7,2)), 'color', [0.8500 0.3250 0.0980])
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(8,2)), 'color', [0.4660 0.6740 0.1880])
plot(WriteFileData(PLOTLINE, MODEL_DataWrite(9,2)), 'color', [0.4940 0.1840 0.5560])
ylim([4500,7200])
legend('True trace', 'Observation', 'UAM LKE', 'UAM UKE', 'NNSSE UKE', 'NNSSE PE', 'NNSSE EKE', 'NNSSE 551', 'NNSSE 10101', 'NNSSE Tanh', 'NNSSE 5551')
title('Pitch')

AccumationStart = 200;
PathTemp1= [CurrentDir,'/EstimationResult/Data_1'];
PathTemp2 = '/Trace10000X_';

EstimatorError = abs(WriteFileData - WriteFileData(:,2));
SumEstimatorError = EstimatorError((end-DATA_ROWS+1):end,:);
for i = AccumationStart:length(SumEstimatorError(:,1))
    SumEstimatorError(i,:) = SumEstimatorError(i-1,:) + SumEstimatorError(i,:);
end

figure(3)
hold on;grid on;
plot(SumEstimatorError(:, 4),'ko')
plot(SumEstimatorError(:, MODEL_DataWrite(1,1)),'b')
plot(SumEstimatorError(:, MODEL_DataWrite(2,1)), 'color', [0.6350 0.0780 0.1840])
plot(SumEstimatorError(:, MODEL_DataWrite(3,1)),'r-.')
plot(SumEstimatorError(:, MODEL_DataWrite(4,1)),'g')
plot(SumEstimatorError(:, MODEL_DataWrite(5,1)),'c')
plot(SumEstimatorError(:, MODEL_DataWrite(6,1)),'m-.')
plot(SumEstimatorError(:, MODEL_DataWrite(7,1)), 'color', [0.8500 0.3250 0.0980])
plot(SumEstimatorError(:, MODEL_DataWrite(8,1)), 'color', [0.4660 0.6740 0.1880])
plot(SumEstimatorError(:, MODEL_DataWrite(9,1)), 'color', [0.4940 0.1840 0.5560])
legend('Observation', 'UAM LKE', 'UAM UKE', 'NNSSE UKE', 'NNSSE PE', 'NNSSE EKE', 'NNSSE 551', 'NNSSE 10101', 'NNSSE Tanh', 'NNSSE 5551', 'location', 'NorthWest')
% ylim([0,7e4])
title('Accumulated Error on X, Proposed estimtors')


NerveNetResult = zeros(length(SumEstimatorError(:,1)),7);
Temp = readmatrix([PathTemp1,PathTemp2,'RNN.csv']);
NerveNetResult(26:end,1) = Temp(:,2);
Temp = readmatrix([PathTemp1,PathTemp2,'LSTM.csv']);
NerveNetResult(26:end,2) = Temp(:,2);
Temp = readmatrix([PathTemp1,PathTemp2,'GRU.csv']);
NerveNetResult(26:end,3) = Temp(:,2);
Temp = readmatrix([PathTemp1,PathTemp2,'TCN.csv']);
NerveNetResult(26:end,4) = Temp(:,2);
Temp = readmatrix([PathTemp1,PathTemp2,'Transformer.csv']);
NerveNetResult(26:end,5) = Temp(:,2);
Temp = readmatrix([PathTemp1,PathTemp2,'NeuralODE.csv']);
NerveNetResult(26:end,6) = Temp(:,2);
Temp = readmatrix([PathTemp1,PathTemp2,'OnlineTransformer.csv']);
NerveNetResult(26:end,7) = [zeros(8,1);Temp(:,2)];

NerveNetError = abs(NerveNetResult - WriteFileData((end-DATA_ROWS+1):end,2));
SumNerveNetError = NerveNetError;
for i = AccumationStart:length(NerveNetError(:,1))
    SumNerveNetError(i,:) = SumNerveNetError(i-1,:) + SumNerveNetError(i,:);
end

figure(4)
hold on;grid on;
plot(SumEstimatorError(:, MODEL_DataWrite(3,1)),'k-.')
plot(SumEstimatorError(:, MODEL_DataWrite(6,1)),'r-.')
plot(SumNerveNetError(:, 1),'b')
plot(SumNerveNetError(:, 2),'r')
plot(SumNerveNetError(:, 3),'g')
plot(SumNerveNetError(:, 4),'c')
plot(SumNerveNetError(:, 5),'color', [0.8500 0.3250 0.0980])
plot(SumNerveNetError(:, 6),'color', [0.4660 0.6740 0.1880])
plot(SumNerveNetError(:, 7),'color', [0.4940 0.1840 0.5560])
legend('NNSSE UKE', 'NNSSE 551', 'RNN', 'LSTM', 'GRU', 'TCN', 'Transformer', 'NeuralODE', 'OnlineTransformer', 'location', 'NorthWest')
title('Accumulated Error on X, Compare with Neural Networks')

figure(5)
hold on; grid on;
plot([zeros(25,1);Temp(1:end, 1)],'m')
plot(NerveNetResult(:, 1),'b')
plot(NerveNetResult(:, 2),'r')
plot(NerveNetResult(:, 3),'g')
plot(NerveNetResult(:, 4),'c')
plot(NerveNetResult(:, 5),'color', [0.8500 0.3250 0.0980])
plot(NerveNetResult(:, 6),'color', [0.4660 0.6740 0.1880])
plot(NerveNetResult(:, 7),'color', [0.4940 0.1840 0.5560])
legend('True trace', 'RNN', 'LSTM', 'GRU', 'TCN', 'Transformer', 'NeuralODE', 'OnlineTransformer')
title('Estimation result on X')


PathTemp2 = '/Trace10000Y_';

EstimatorError = abs(WriteFileData - WriteFileData(:,3));
SumEstimatorError = EstimatorError((end-DATA_ROWS+1):end,:);
for i = AccumationStart:length(SumEstimatorError(:,1))
    SumEstimatorError(i,:) = SumEstimatorError(i-1,:) + SumEstimatorError(i,:);
end

figure(6)
hold on;grid on;
plot(SumEstimatorError(:, 5),'ko')
plot(SumEstimatorError(:, MODEL_DataWrite(1,2)),'b')
plot(SumEstimatorError(:, MODEL_DataWrite(2,2)), 'color', [0.6350 0.0780 0.1840])
plot(SumEstimatorError(:, MODEL_DataWrite(3,2)),'r-.')
plot(SumEstimatorError(:, MODEL_DataWrite(4,2)),'g')
plot(SumEstimatorError(:, MODEL_DataWrite(5,2)),'c')
plot(SumEstimatorError(:, MODEL_DataWrite(6,2)),'m-.')
plot(SumEstimatorError(:, MODEL_DataWrite(7,2)), 'color', [0.8500 0.3250 0.0980])
plot(SumEstimatorError(:, MODEL_DataWrite(8,2)), 'color', [0.4660 0.6740 0.1880])
plot(SumEstimatorError(:, MODEL_DataWrite(9,2)), 'color', [0.4940 0.1840 0.5560])
legend('Observation', 'UAM LKE', 'UAM UKE', 'NNSSE UKE', 'NNSSE PE', 'NNSSE EKE', 'NNSSE 551', 'NNSSE 10101', 'NNSSE Tanh', 'NNSSE 5551', 'location', 'NorthWest')
% ylim([0,3e4])
title('Accumulated Error on Y, Proposed estimtors')

NerveNetResult = zeros(length(SumEstimatorError(:,1)),6);
Temp = readmatrix([PathTemp1,PathTemp2,'RNN.csv']);
NerveNetResult(26:end,1) = Temp(:,2);
Temp = readmatrix([PathTemp1,PathTemp2,'LSTM.csv']);
NerveNetResult(26:end,2) = Temp(:,2);
Temp = readmatrix([PathTemp1,PathTemp2,'GRU.csv']);
NerveNetResult(26:end,3) = Temp(:,2);
Temp = readmatrix([PathTemp1,PathTemp2,'TCN.csv']);
NerveNetResult(26:end,4) = Temp(:,2);
Temp = readmatrix([PathTemp1,PathTemp2,'Transformer.csv']);
NerveNetResult(26:end,5) = Temp(:,2);
Temp = readmatrix([PathTemp1,PathTemp2,'NeuralODE.csv']);
NerveNetResult(26:end,6) = Temp(:,2);
Temp = readmatrix([PathTemp1,PathTemp2,'OnlineTransformer.csv']);
NerveNetResult(26:end,7) = [zeros(8,1);Temp(:,2)];


NerveNetError = abs(NerveNetResult - WriteFileData((end-DATA_ROWS+1):end,3));
SumNerveNetError = NerveNetError;
for i = AccumationStart:length(NerveNetError(:,1))
    SumNerveNetError(i,:) = SumNerveNetError(i-1,:) + SumNerveNetError(i,:);
end

figure(7)
hold on;grid on;
plot(SumEstimatorError(:, MODEL_DataWrite(3,2)),'r-.')
plot(SumEstimatorError(:, MODEL_DataWrite(6,2)),'m-.')
plot(SumNerveNetError(:, 1),'b')
plot(SumNerveNetError(:, 2),'r')
plot(SumNerveNetError(:, 3),'g')
plot(SumNerveNetError(:, 4),'c')
plot(SumNerveNetError(:, 5),'color', [0.8500 0.3250 0.0980])
plot(SumNerveNetError(:, 6),'color', [0.4660 0.6740 0.1880])
plot(SumNerveNetError(:, 7),'color', [0.4940 0.1840 0.5560])
legend('NNSSE UKE', 'NNSSE 551', 'RNN', 'LSTM', 'GRU', 'TCN', 'Transformer', 'NeuralODE', 'OnlineTransformer', 'location', 'NorthWest')
title('Accumulated Error on Y, Compare with Neural Networks (trained with X Data)')

figure(8)
hold on; grid on;
plot([zeros(25,1);Temp(1:end, 1)],'m')
plot(NerveNetResult(:, 1),'b')
plot(NerveNetResult(:, 2),'r')
plot(NerveNetResult(:, 3),'g')
plot(NerveNetResult(:, 4),'c')
plot(NerveNetResult(:, 5),'color', [0.8500 0.3250 0.0980])
plot(NerveNetResult(:, 6),'color', [0.4660 0.6740 0.1880])
plot(NerveNetResult(:, 7),'color', [0.4940 0.1840 0.5560])
legend('True trace', 'RNN', 'LSTM', 'GRU', 'TCN', 'Transformer', 'NeuralODE', 'OnlineTransformer')
title('Estimation result on Y')