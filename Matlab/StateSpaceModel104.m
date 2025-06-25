function StateSpaceModelN = StateSpaceModel104(StateSpaceModelN)
    % 初始化结构体中的变量
    StateSpaceModelN.PortName = 'Tested Model 1 v0.00';
    StateSpaceModelN.PortIntroduction = 'NNSSE for UKF, simulate a recurrent neural network as 5-5-1';

    StateSpaceModelN.Scale = 1;
    StateSpaceModelN.PredictStep = 3;
    StateSpaceModelN.InputLayer = 5;
    StateSpaceModelN.MiddleLayer = StateSpaceModelN.InputLayer;
    StateSpaceModelN.OutpuLayer = 1;
    StateSpaceModelN.NodeCount = StateSpaceModelN.InputLayer + StateSpaceModelN.PredictStep;
    StateSpaceModelN.WeightCount = StateSpaceModelN.InputLayer * StateSpaceModelN.MiddleLayer + StateSpaceModelN.MiddleLayer * StateSpaceModelN.OutpuLayer;
    StateSpaceModelN.Nx = StateSpaceModelN.NodeCount + StateSpaceModelN.WeightCount;
    StateSpaceModelN.Nz = 1;
    StateSpaceModelN.Intervel = 0.005;
    StateSpaceModelN.PredictTime = StateSpaceModelN.PredictStep * StateSpaceModelN.Intervel;

    StateSpaceModelN.Matrix_Q = diag([ones(1,StateSpaceModelN.NodeCount), 0.0000001*ones(1, StateSpaceModelN.WeightCount)]);
    StateSpaceModelN.Matrix_R = eye(StateSpaceModelN.Nz);
    StateSpaceModelN.Matrix_P = StateSpaceModelN.Matrix_Q;

    if ~isfield(StateSpaceModelN, 'CurrentObservation')
        error('"CurrentObservation" needs to be set before initiate StateSpaceModel104');
    end
    
    StateSpaceModelN.EstimatedState = [StateSpaceModelN.CurrentObservation*ones(StateSpaceModelN.NodeCount,1); sqrt(2/(5+1))*randn(StateSpaceModelN.WeightCount,1)];
    StateSpaceModelN.PredictedState = zeros(StateSpaceModelN.Nz, 1);
    StateSpaceModelN.PredictedObservation = zeros(StateSpaceModelN.Nz, 1);
    
    % 定义结构体中的函数句柄
    StateSpaceModelN.StateTransitionEquation = @(In_State, StateSpaceModelN) StateSpaceModel104StateTransitionFunction(In_State, StateSpaceModelN);
    StateSpaceModelN.ObservationEquation = @(In_State, StateSpaceModelN) StateSpaceModel104ObservationFunction(In_State, StateSpaceModelN);
    StateSpaceModelN.PredictionEquation = @(In_State, StateSpaceModelN) StateSpaceModel104PredictionFunction(In_State, StateSpaceModelN);
    StateSpaceModelN.EstimatorPort = @(StateSpaceModelN) StateSpaceModel104EstimatorPort(StateSpaceModelN);
    StateSpaceModelN.EstimatorPortTermination = @() StateSpaceModel104EstimatorPortTermination();
end

% 定义各个函数的实现
function [Out_State, StateSpaceModelN] = StateSpaceModel104StateTransitionFunction(In_State, StateSpaceModelN)
    InputLayer = In_State((StateSpaceModelN.PredictStep+1):StateSpaceModelN.NodeCount)/StateSpaceModelN.Scale ;
    Layer12Weight = In_State((StateSpaceModelN.NodeCount+1):(StateSpaceModelN.NodeCount+StateSpaceModelN.InputLayer * StateSpaceModelN.MiddleLayer));
    Layer23Weight = In_State((StateSpaceModelN.NodeCount+StateSpaceModelN.InputLayer * StateSpaceModelN.MiddleLayer+1):end);
    MiddleLayer = zeros(StateSpaceModelN.MiddleLayer,1);
    for i =1:StateSpaceModelN.MiddleLayer
        MiddleLayer(i) = InputLayer'*Layer12Weight(((i-1)*StateSpaceModelN.MiddleLayer+1):i*StateSpaceModelN.MiddleLayer);
        MiddleLayer(i) = MiddleLayer(i);
    end
    OutputLayer = MiddleLayer'*Layer23Weight*StateSpaceModelN.Scale;
    Out_State = zeros(StateSpaceModelN.Nx,1);
    Out_State(1) = OutputLayer;
    Out_State(2 : StateSpaceModelN.NodeCount) = In_State(1:(StateSpaceModelN.NodeCount - 1));
    Out_State((StateSpaceModelN.NodeCount + 1) : end) = In_State((StateSpaceModelN.NodeCount + 1) : end);
end

function [Out_Observation, StateSpaceModelN] = StateSpaceModel104ObservationFunction(In_State, StateSpaceModelN)
    Out_Observation = In_State(1);
end

function [Out_PredictedState, StateSpaceModelN] = StateSpaceModel104PredictionFunction(In_State, StateSpaceModelN)
    InputLayer = In_State((StateSpaceModelN.PredictStep+1):StateSpaceModelN.NodeCount)/StateSpaceModelN.Scale ;
    Layer12Weight = In_State((StateSpaceModelN.NodeCount+1):(StateSpaceModelN.NodeCount+StateSpaceModelN.InputLayer * StateSpaceModelN.MiddleLayer));
    Layer23Weight = In_State((StateSpaceModelN.NodeCount+StateSpaceModelN.InputLayer * StateSpaceModelN.MiddleLayer+1):end);
    MiddleLayer = zeros(StateSpaceModelN.MiddleLayer,1);
    for i =1:StateSpaceModelN.MiddleLayer
        MiddleLayer(i) = InputLayer'*Layer12Weight(((i-1)*StateSpaceModelN.MiddleLayer+1):i*StateSpaceModelN.MiddleLayer);
        MiddleLayer(i) = MiddleLayer(i);
    end
    Out_PredictedState = MiddleLayer'*Layer23Weight*StateSpaceModelN.Scale;
end

function StateSpaceModelN = StateSpaceModel104EstimatorPort(StateSpaceModelN)
    StateSpaceModelN = Estimator3002(StateSpaceModelN);
end

function StateSpaceModel104EstimatorPortTermination()
    fprintf('EstimatorPort terminated.\n');
end