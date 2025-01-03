function StateSpaceModelN = StateSpaceModel109(StateSpaceModelN)
    % 初始化结构体中的变量
    StateSpaceModelN.PortName = 'Tested Model 1 v0.00';
    StateSpaceModelN.PortIntroduction = 'Accurate Sin Model';

    StateSpaceModelN.Nx = 4;
    StateSpaceModelN.Nz = 2;
    StateSpaceModelN.PredictStep = 3;
    StateSpaceModelN.Intervel = 0.005;
    StateSpaceModelN.PredictTime = StateSpaceModelN.PredictStep * StateSpaceModelN.Intervel;

    W1 = 2*pi;
    A1 = 10;
    A2 = 10;
    dT = 0.005;
    X0 = [A1*sin(W1*dT);
        W1*A1*cos(W1*dT);
        A2*sin(W1*dT);
        W1*A2*cos(W1*dT)];
    A = zeros(4,4);
    A(1,1) = cos(W1*dT);
    A(1,2) = sin(W1*dT)/W1;
    A(2,1) = -W1*sin(W1*dT);
    A(2,2) = cos(W1*dT);
    A(3,3) = cos(W1*dT);
    A(3,4) = sin(W1*dT)/W1;
    A(4,3) = -W1*sin(W1*dT);
    A(4,4) = cos(W1*dT);

    StateSpaceModelN.EstimatedState = X0;
    StateSpaceModelN.PredictedState = zeros(StateSpaceModelN.Nx, 1);
    StateSpaceModelN.CurrentObservation = zeros(StateSpaceModelN.Nz, 1);
    StateSpaceModelN.PredictedObservation = zeros(StateSpaceModelN.Nz, 1);
    StateSpaceModelN.Matrix_F = A;
    StateSpaceModelN.Matrix_G = eye(StateSpaceModelN.Nx);
    StateSpaceModelN.Matrix_B = zeros(StateSpaceModelN.Nx,1);
    StateSpaceModelN.Matrix_H = zeros(StateSpaceModelN.Nz, StateSpaceModelN.Nx);
    Indices = sub2ind(size(StateSpaceModelN.Matrix_H), [1,2],[1,3]);
    StateSpaceModelN.Matrix_H(Indices) = 1;

    StateSpaceModelN.Matrix_Q = eye(StateSpaceModelN.Nx);
    StateSpaceModelN.Matrix_R = eye(StateSpaceModelN.Nz);
    StateSpaceModelN.Matrix_P = StateSpaceModelN.Matrix_Q;

    StateSpaceModelN.Int_Par = 1;
    StateSpaceModelN.Double_Par = 1;

    % 定义结构体中的函数句柄
    StateSpaceModelN.StateTransitionEquation = @(In_State, StateSpaceModelN) StateSpaceModel109StateTransitionFunction(In_State, StateSpaceModelN);
    StateSpaceModelN.ObservationEquation = @(In_State, StateSpaceModelN) StateSpaceModel109ObservationFunction(In_State, StateSpaceModelN);
    StateSpaceModelN.PredictionEquation = @(In_State, StateSpaceModelN) StateSpaceModel109PredictionFunction(In_State, StateSpaceModelN);
    StateSpaceModelN.EstimatorPort = @(StateSpaceModelN) StateSpaceModel109EstimatorPort(StateSpaceModelN);
    StateSpaceModelN.EstimatorPortTermination = @() StateSpaceModel109EstimatorPortTermination();
end

% 定义各个函数的实现
function [Out_State, StateSpaceModelN] = StateSpaceModel109StateTransitionFunction(In_State, StateSpaceModelN)
    Out_State = StateSpaceModelN.Matrix_F * In_State;
end

function [Out_Observation, StateSpaceModelN] = StateSpaceModel109ObservationFunction(In_State, StateSpaceModelN)
    Out_Observation = StateSpaceModelN.Matrix_H * In_State;
end

function [Out_PredictedState, StateSpaceModelN] = StateSpaceModel109PredictionFunction(In_State, StateSpaceModelN)
    W1 = 5;
    dT = StateSpaceModelN.PredictTime;
    A = zeros(4,4);
    A(1,1) = cos(W1*dT);
    A(1,2) = sin(W1*dT)/W1;
    A(2,1) = -W1*sin(W1*dT);
    A(2,2) = cos(W1*dT);
    A(3,3) = cos(W1*dT);
    A(3,4) = sin(W1*dT)/W1;
    A(4,3) = -W1*sin(W1*dT);
    A(4,4) = cos(W1*dT);
    Out_PredictedState =  A * In_State;
end

function StateSpaceModelN = StateSpaceModel109EstimatorPort(StateSpaceModelN)
    StateSpaceModelN = Estimator3001(StateSpaceModelN);
end

function StateSpaceModel109EstimatorPortTermination()
    fprintf('EstimatorPort terminated.\n');
end