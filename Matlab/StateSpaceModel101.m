function StateSpaceModelN = StateSpaceModel101(StateSpaceModelN)
    % 初始化结构体中的变量
    StateSpaceModelN.PortName = 'Tested Model 1 v0.00';
    StateSpaceModelN.PortIntroduction = 'NNSSE for UKF';

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
        error('"CurrentObservation" needs to be set before initiate StateSpaceModel101');
    end
    
    Cw=-2/(StateSpaceModelN.Np-1)/(StateSpaceModelN.Np-2);
    StateSpaceModelN.EstimatedState = [StateSpaceModelN.CurrentObservation*ones(StateSpaceModelN.Np+StateSpaceModelN.PredictStep,1);...
        1+2/(StateSpaceModelN.Np-1);Cw*ones(StateSpaceModelN.Np-2,1);0];
    StateSpaceModelN.PredictedState = zeros(StateSpaceModelN.Nz, 1);
    StateSpaceModelN.PredictedObservation = zeros(StateSpaceModelN.Nz, 1);
    
    % 定义结构体中的函数句柄
    StateSpaceModelN.StateTransitionEquation = @(In_State, StateSpaceModelN) StateSpaceModel101StateTransitionFunction(In_State, StateSpaceModelN);
    StateSpaceModelN.ObservationEquation = @(In_State, StateSpaceModelN) StateSpaceModel101ObservationFunction(In_State, StateSpaceModelN);
    StateSpaceModelN.PredictionEquation = @(In_State, StateSpaceModelN) StateSpaceModel101PredictionFunction(In_State, StateSpaceModelN);
    StateSpaceModelN.EstimatorPort = @(StateSpaceModelN) StateSpaceModel101EstimatorPort(StateSpaceModelN);
    StateSpaceModelN.EstimatorPortTermination = @() StateSpaceModel101EstimatorPortTermination();
end

% 定义各个函数的实现
function [Out_State, StateSpaceModelN] = StateSpaceModel101StateTransitionFunction(In_State, StateSpaceModelN)
    Out_State = zeros(StateSpaceModelN.Nx,1);
    Out_State(1) = In_State((StateSpaceModelN.PredictStep + 1):(StateSpaceModelN.PredictStep + StateSpaceModelN.Np))' * In_State((StateSpaceModelN.PredictStep + StateSpaceModelN.Np + 1) : end);
    Out_State(2 : (StateSpaceModelN.PredictStep + StateSpaceModelN.Np)) = In_State(1:(StateSpaceModelN.PredictStep + StateSpaceModelN.Np - 1));
    Out_State((StateSpaceModelN.PredictStep + StateSpaceModelN.Np + 1) : end) = In_State((StateSpaceModelN.PredictStep + StateSpaceModelN.Np + 1) : end);
end

function [Out_Observation, StateSpaceModelN] = StateSpaceModel101ObservationFunction(In_State, StateSpaceModelN)
    Out_Observation = In_State(1);
end

function [Out_PredictedState, StateSpaceModelN] = StateSpaceModel101PredictionFunction(In_State, StateSpaceModelN)
    Out_PredictedState = In_State(1:StateSpaceModelN.Np)' * In_State((StateSpaceModelN.PredictStep + StateSpaceModelN.Np + 1) : end);
end

function StateSpaceModelN = StateSpaceModel101EstimatorPort(StateSpaceModelN)
    StateSpaceModelN = Estimator3002(StateSpaceModelN);
end

function StateSpaceModel101EstimatorPortTermination()
    fprintf('EstimatorPort terminated.\n');
end