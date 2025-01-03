function StateSpaceModelN = StateSpaceModel103(StateSpaceModelN)
    % 初始化结构体中的变量
    StateSpaceModelN.PortName = 'Tested Model 1 v0.00';
    StateSpaceModelN.PortIntroduction = 'NNSSE for EKF';

    StateSpaceModelN.Np = 25;
    StateSpaceModelN.PredictStep = 3;
    StateSpaceModelN.Nx = 2 * StateSpaceModelN.Np + StateSpaceModelN.PredictStep;
    StateSpaceModelN.Nz = 1;
    StateSpaceModelN.Intervel = 0.005;
    StateSpaceModelN.PredictTime = StateSpaceModelN.PredictStep * StateSpaceModelN.Intervel;

    StateSpaceModelN.Matrix_Q = diag([ones(1,StateSpaceModelN.Np + StateSpaceModelN.PredictStep), 0.00001*ones(1, StateSpaceModelN.Np)]);
    StateSpaceModelN.Matrix_R = eye(StateSpaceModelN.Nz);
    StateSpaceModelN.Matrix_P = StateSpaceModelN.Matrix_Q;

    if ~isfield(StateSpaceModelN, 'CurrentObservation')
        error('"CurrentObservation" needs to be set before initiate StateSpaceModel103');
    end
    
    Cw=-2/(StateSpaceModelN.Np-1)/(StateSpaceModelN.Np-2);
    StateSpaceModelN.EstimatedState = [StateSpaceModelN.CurrentObservation*ones(StateSpaceModelN.Np+StateSpaceModelN.PredictStep,1);...
        1+2/(StateSpaceModelN.Np-1);Cw*ones(StateSpaceModelN.Np-2,1);0];
    StateSpaceModelN.PredictedState = zeros(StateSpaceModelN.Nz, 1);
    StateSpaceModelN.PredictedObservation = zeros(StateSpaceModelN.Nz, 1);
    
    % 定义结构体中的函数句柄
    StateSpaceModelN.StateTransitionEquation = @(In_State, StateSpaceModelN) StateSpaceModel103StateTransitionFunction(In_State, StateSpaceModelN);
    StateSpaceModelN.StateTransitionDiffEquation = @(In_State, StateSpaceModelN) StateSpaceModel103StateTransitionDiffFunction(In_State, StateSpaceModelN);
    StateSpaceModelN.ObservationEquation = @(In_State, StateSpaceModelN) StateSpaceModel103ObservationFunction(In_State, StateSpaceModelN);
    StateSpaceModelN.ObservationDiffEquation = @(In_State, StateSpaceModelN) StateSpaceModel103ObservationDiffFunction(In_State, StateSpaceModelN);
    StateSpaceModelN.PredictionEquation = @(In_State, StateSpaceModelN) StateSpaceModel103PredictionFunction(In_State, StateSpaceModelN);
    StateSpaceModelN.EstimatorPort = @(StateSpaceModelN) StateSpaceModel103EstimatorPort(StateSpaceModelN);
    StateSpaceModelN.EstimatorPortTermination = @() StateSpaceModel103EstimatorPortTermination();
end

% 定义各个函数的实现
function [Out_State, StateSpaceModelN] = StateSpaceModel103StateTransitionFunction(In_State, StateSpaceModelN)
    Out_State = zeros(StateSpaceModelN.Nx,1);
    Out_State(1) = In_State((StateSpaceModelN.PredictStep + 1):(StateSpaceModelN.PredictStep + StateSpaceModelN.Np))' * In_State((StateSpaceModelN.PredictStep + StateSpaceModelN.Np + 1) : end);
    Out_State(2 : (StateSpaceModelN.PredictStep + StateSpaceModelN.Np)) = In_State(1:(StateSpaceModelN.PredictStep + StateSpaceModelN.Np - 1));
    Out_State((StateSpaceModelN.PredictStep + StateSpaceModelN.Np + 1) : end) = In_State((StateSpaceModelN.PredictStep + StateSpaceModelN.Np + 1) : end);
end

function [Out_State, StateSpaceModelN] = StateSpaceModel103StateTransitionDiffFunction(In_State, StateSpaceModelN)
    Out_State = zeros(StateSpaceModelN.Nx, StateSpaceModelN.Nx);

    idx_a = (StateSpaceModelN.PredictStep + 1):(StateSpaceModelN.PredictStep + StateSpaceModelN.Np);  % 第一段索引
    idx_b = (StateSpaceModelN.PredictStep + StateSpaceModelN.Np + 1):StateSpaceModelN.Nx;             % 第二段索引

    Out_State(1 , idx_a) = In_State(idx_b);
    Out_State(1 , idx_b) = In_State(idx_a);

     for i = 2:(StateSpaceModelN.PredictStep  + StateSpaceModelN.Np)
        Out_State(i, i - 1) = 1;
    end

    for i = (StateSpaceModelN.PredictStep  + StateSpaceModelN.Np + 1):StateSpaceModelN.Nx
        Out_State(i, i) = 1;
    end
end

function [Out_Observation, StateSpaceModelN] = StateSpaceModel103ObservationFunction(In_State, StateSpaceModelN)
    Out_Observation = In_State((StateSpaceModelN.PredictStep + 1):(StateSpaceModelN.PredictStep + StateSpaceModelN.Np))' * In_State((StateSpaceModelN.PredictStep + StateSpaceModelN.Np + 1) : end);
end

function [Out_Observation, StateSpaceModelN] = StateSpaceModel103ObservationDiffFunction(In_State, StateSpaceModelN)
    Out_Observation = zeros(1, StateSpaceModelN.Nx);
    idx_a = (StateSpaceModelN.PredictStep + 1):(StateSpaceModelN.PredictStep + StateSpaceModelN.Np);  % 第一段索引
    idx_b = (StateSpaceModelN.PredictStep + StateSpaceModelN.Np + 1):StateSpaceModelN.Nx;             % 第二段索引

    Out_Observation(1 , idx_a) = In_State(idx_b);
    Out_Observation(1 , idx_b) = In_State(idx_a);
end

function [Out_PredictedState, StateSpaceModelN] = StateSpaceModel103PredictionFunction(In_State, StateSpaceModelN)
    Out_PredictedState = In_State(1:StateSpaceModelN.Np)' * In_State((StateSpaceModelN.PredictStep + StateSpaceModelN.Np + 1) : end);
end

function StateSpaceModelN = StateSpaceModel103EstimatorPort(StateSpaceModelN)
    StateSpaceModelN = Estimator3004(StateSpaceModelN);
end

function StateSpaceModel103EstimatorPortTermination()
    fprintf('EstimatorPort terminated.\n');
end