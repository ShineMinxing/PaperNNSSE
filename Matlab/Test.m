close all;clc;clear;

%% 轨迹生成

T=0.01; %采样周期
N=10/T; %采样点数
Nc=N;
X=zeros(8,N);          %真实值
X(:,1)=[-100,50,0,0,200,20,0,0];
Z=zeros(2,N);          %观测值
Z(:,1)=[X(1,1),X(5,1)];
Q0=eye(8).*repmat([1;0.1;0.1;0.1;1;0.1;0.1;0.1],1,8);%过程噪声方差
R0=5*eye(2);              %观测噪声方差
F=[1,T,T^2/2,T^3/3,0,0,0,0;
   0,1,T,T^2/2,0,0,0,0;
   0,0,1,T,0,0,0,0;
   0,0,0,1,0,0,0,0;
   0,0,0,0,1,T,T^2/2,T^3/3;
   0,0,0,0,0,1,T,T^2/2;
   0,0,0,0,0,0,1,T;
   0,0,0,0,0,0,0,1];
H=[1,0,0,0,0,0,0,0;0,0,0,0,1,0,0,0];
B=[1;0;0;0;1;0;0;0]*randsrc(N/Nc,1)';
B(5,:)=B(5,:).*randsrc(N/Nc,1)';
for t=2:N
    X(:,t)=F*X(:,t-1)+sqrt(Q0)*randn(8,1)+B(:,floor((t-1)/Nc)+1);
    Z(:,t)=H*X(:,t)+sqrt(R0)*randn(2,1);
end


%% 轨迹导入
% data0 = load('D:\华为云盘\研究\研究进程\22_05\data-k.txt');
% T = 0.02;
% N = length(data0(:,1));
% data  = data0(1,:);
% for i=2:N
%     if(data0(i-1,9)~=data0(i,9))
%         data=[data;data0(i,:)];
%     end
% end
% N = length(data(:,1));
% X=zeros(4,N);
% X(1,:)=data(:,7)'+data(:,9)';
% X(5,:)=data(:,8)'+data(:,10)';
% Z = [data(:,1)';data(:,2)'];

%%参数设置
Q0=eye(8).*repmat([0.3;0.1;0.01;0.001;0.3;0.1;0.01;0.001],1,8);%过程噪声方差
R0=[2,0;0,8];  
Err_Ob=zeros(2,N);
Err_Kp=zeros(2,N);
Err_Kv=zeros(2,N);
Err_Ka=zeros(2,N);
Err_Kaa=zeros(2,N);
Err_K2=zeros(2,N);
Err_K3=zeros(2,N);
Err_K4=zeros(2,N);
Err_Kv4=zeros(2,N);

for i=1:N
    Err_Ob(:,i)=RMS(X(:,i),Z(:,i));
end

%% kalman滤波仅位置
F=[1,0;0,1];
H=[1,0;0,1];
Length=2;
Xkp=zeros(Length,N);%估计状态
Xkf=zeros(Length,N);%估计状态
Xkp(:,1)=[X(1:Length/2,1);X(5:(4+Length/2),1)];
Xkf(:,1)=[X(1:Length/2,1);X(5:(4+Length/2),1)];
P0=eye(Length);
Q = Q0([1:Length/2,(4+1):(4+Length/2)],[1:Length/2,(4+1):(4+Length/2)]);
R = R0;
for i=2:N
    Xkp(:,i)=F*Xkf(:,i-1);%状态预测
    P1=F*P0*F'+Q;%协方差预测
    K=P1*H'*inv(H*P1*H'+R);%计算Kalman增益
    Xkf(:,i)=Xkp(:,i)+K*(Z(:,i)-H*Xkp(:,i));%状态更新
    P0=(eye(Length)-K*H)*P1;%协方差更新
end
for i=1:N
    Err_Kp(:,i)=RMS(X(:,i),Xkp([1,1+Length/2],i));
end

%% kalman滤波位置速度
F=[1,T,0,0;0,1,0,0;0,0,1,T;0,0,0,1];
H=[1,0,0,0;0,0,1,0];
Length=4;
Xkv=zeros(Length,N);%估计状态
Xkf=zeros(Length,N);%估计状态
Xkv(:,1)=[X(1:Length/2,1);X(5:(4+Length/2),1)];
Xkf(:,1)=[X(1:Length/2,1);X(5:(4+Length/2),1)];
P0=eye(Length);
Q = Q0([1:Length/2,(4+1):(4+Length/2)],[1:Length/2,(4+1):(4+Length/2)]);
R = R0;
for i=2:N
    Xkv(:,i)=F*Xkf(:,i-1);%状态预测
    P1=F*P0*F'+Q;%协方差预测
    K=P1*H'*inv(H*P1*H'+R);%计算Kalman增益
    Xkf(:,i)=Xkv(:,i)+K*(Z(:,i)-H*Xkv(:,i));%状态更新
    P0=(eye(Length)-K*H)*P1;%协方差更新
end
for i=1:N
    Err_Kv(:,i)=RMS(X(:,i),Xkv([1,1+Length/2],i));
end

%% kalman滤波位置速度加速度
F=[1,T,T^2/2,0,0,0;
    0,1,T,0,0,0;
    0,0,1,0,0,0;
    0,0,0,1,T,T^2/2;
    0,0,0,0,1,T;
    0,0,0,0,0,1];
H=[1,0,0,0,0,0;0,0,0,1,0,0];
Length=6;
Xka=zeros(Length,N);%估计状态
Xkf=zeros(Length,N);%估计状态
Xka(:,1)=[X(1:Length/2,1);X(5:(4+Length/2),1)];
Xkf(:,1)=[X(1:Length/2,1);X(5:(4+Length/2),1)];
P0=eye(Length);
Q = Q0([1:Length/2,(4+1):(4+Length/2)],[1:Length/2,(4+1):(4+Length/2)]);
R = R0;
for i=2:N
    Xka(:,i)=F*Xkf(:,i-1);%状态预测
    P1=F*P0*F'+Q;%协方差预测
    K=P1*H'*inv(H*P1*H'+R);%计算Kalman增益
    Xkf(:,i)=Xka(:,i)+K*(Z(:,i)-H*Xka(:,i));%状态更新
    P0=(eye(Length)-K*H)*P1;%协方差更新
end
for i=1:N
    Err_Ka(:,i)=RMS(X(:,i),Xka([1,1+Length/2],i));
end

%% kalman滤波位置速度加速度加加速度
F=[1,T,T^2/2,T^3/3,0,0,0,0;
   0,1,T,T^2/2,0,0,0,0;
   0,0,1,T,0,0,0,0;
   0,0,0,1,0,0,0,0;
   0,0,0,0,1,T,T^2/2,T^3/3;
   0,0,0,0,0,1,T,T^2/2;
   0,0,0,0,0,0,1,T;
   0,0,0,0,0,0,0,1];
H=[1,0,0,0,0,0,0,0;0,0,0,0,1,0,0,0];
Length=8;
Xkaa=zeros(Length,N);%估计状态
Xkf=zeros(Length,N);%估计状态
Xkaa(:,1)=[X(1:Length/2,1);X(5:(4+Length/2),1)];
Xkf(:,1)=[X(1:Length/2,1);X(5:(4+Length/2),1)];
P0=eye(Length);
Q = Q0([1:Length/2,(4+1):(4+Length/2)],[1:Length/2,(4+1):(4+Length/2)]);
R = R0;
for i=2:N
    Xkaa(:,i)=F*Xkf(:,i-1);%状态预测
    P1=F*P0*F'+Q;%协方差预测
    K=P1*H'*inv(H*P1*H'+R);%计算Kalman增益
    Xkf(:,i)=Xkaa(:,i)+K*(Z(:,i)-H*Xkaa(:,i));%状态更新
    P0=(eye(Length)-K*H)*P1;%协方差更新
end
for i=1:N
    Err_Kaa(:,i)=RMS(X(:,i),Xkaa([1,1+Length/2],i));
end

%% kalman滤波双状态
F=[2,-1,0,0;1,0,0,0;0,0,2,-1;0,0,1,0];
H=eye(4);
Length=4;
Z0=[];
for i=1:Length/2
    Z0=[Z0;[Z(1,1)*ones(1,i-1),Z(1,1:end+1-i)]];
end
for i=1:Length/2
    Z0=[Z0;[Z(2,1)*ones(1,i-1),Z(2,1:end+1-i)]];
end
Xk2=zeros(Length,N);%估计状态
Xkf=zeros(Length,N);%估计状态
Xk2(:,1)=Z0(:,1);
Xkf(:,1)=Z0(:,1);
P0=eye(Length);
Q = Q0([1:Length/2,(4+1):(4+Length/2)],[1:Length/2,(4+1):(4+Length/2)]);
R = R0(1,1)*eye(Length);
for i=2:N
    Xk2(:,i)=F*Xkf(:,i-1);%状态预测
    P1=F*P0*F'+Q;%协方差预测
    K=P1*H'*inv(H*P1*H'+R);%计算Kalman增益
    Xkf(:,i)=Xk2(:,i)+K*(Z0(:,i)-H*Xk2(:,i));%状态更新
    P0=(eye(Length)-K*H)*P1;%协方差更新
    for j=2:Length/2
        R(j,j) = P0(j-1,j-1);
        R(Length/2+j,Length/2+j) = P0(Length/2+j-1,Length/2+j-1);
    end
end
for i=1:N
    Err_K2(:,i)=RMS(X(:,i),Xk2([1,1+Length/2],i));
end

%% kalman滤波三状态
F=[3/2,0,-1/2,0,0,0;1,0,0,0,0,0;0,1,0,0,0,0;0,0,0,3/2,0,-1/2;0,0,0,1,0,0;0,0,0,0,1,0];
H=eye(6);
Length=6;
Z0=[];
for i=1:Length/2
    Z0=[Z0;[Z(1,1)*ones(1,i-1),Z(1,1:end+1-i)]];
end
for i=1:Length/2
    Z0=[Z0;[Z(2,1)*ones(1,i-1),Z(2,1:end+1-i)]];
end
Xk3=zeros(Length,N);%估计状态
Xkf=zeros(Length,N);%估计状态
Xk3(:,1)=Z0(:,1);
Xkf(:,1)=Z0(:,1);
P0=eye(Length);
Q = Q0([1:Length/2,(4+1):(4+Length/2)],[1:Length/2,(4+1):(4+Length/2)]);
R = R0(1,1)*eye(Length);
for i=2:N
    Xk3(:,i)=F*Xkf(:,i-1);%状态预测
    P1=F*P0*F'+Q;%协方差预测
    K=P1*H'*inv(H*P1*H'+R);%计算Kalman增益
    Xkf(:,i)=Xk3(:,i)+K*(Z0(:,i)-H*Xk3(:,i));%状态更新
    P0=(eye(Length)-K*H)*P1;%协方差更新
    for j=2:Length/2
        R(j,j) = P0(j-1,j-1);
        R(Length/2+j,Length/2+j) = P0(Length/2+j-1,Length/2+j-1);
    end
end
for i=1:N
    Err_K3(:,i)=RMS(X(:,i),Xk3([1,1+Length/2],i));
end

%% kalman滤波四状态
F=[4/3,0,0,-1/3,0,0,0,0;
    1,0,0,0,0,0,0,0;
    0,1,0,0,0,0,0,0;
    0,0,1,0,0,0,0,0;
    0,0,0,0,4/3,0,0,-1/3;
    0,0,0,0,1,0,0,0;
    0,0,0,0,0,1,0,0;
    0,0,0,0,0,0,1,0,];
H=eye(8);
Length=8;
Z0=[];
for i=1:Length/2
    Z0=[Z0;[Z(1,1)*ones(1,i-1),Z(1,1:end+1-i)]];
end
for i=1:Length/2
    Z0=[Z0;[Z(2,1)*ones(1,i-1),Z(2,1:end+1-i)]];
end
Xk4=zeros(Length,N);%估计状态
Xkf=zeros(Length,N);%估计状态
Xk4(:,1)=Z0(:,1);
Xkf(:,1)=Z0(:,1);
P0=eye(Length);
Q = Q0([1:Length/2,(4+1):(4+Length/2)],[1:Length/2,(4+1):(4+Length/2)]);
R = R0(1,1)*eye(Length);
for i=2:N
    Xk4(:,i)=F*Xkf(:,i-1);%状态预测
    P1=F*P0*F'+Q;%协方差预测
    K=P1*H'*inv(H*P1*H'+R);%计算Kalman增益
    Xkf(:,i)=Xk4(:,i)+K*(Z0(:,i)-H*Xk4(:,i));%状态更新
    P0=(eye(Length)-K*H)*P1;%协方差更新
    for j=2:Length/2
        R(j,j) = P0(j-1,j-1);
        R(Length/2+j,Length/2+j) = P0(Length/2+j-1,Length/2+j-1);
    end
end
for i=1:N
    Err_K4(:,i)=RMS(X(:,i),Xk4([1,1+Length/2],i));
end

%% kalman滤波四状态变系数
%此系数是否可使用智能方法训练？
F=[3/2,-1/6,-1/6,-1/6,0,0,0,0;
    1,0,0,0,0,0,0,0;
    0,1,0,0,0,0,0,0;
    0,0,1,0,0,0,0,0;
    0,0,0,0,3/2,-1/6,-1/6,-1/6;
    0,0,0,0,1,0,0,0;
    0,0,0,0,0,1,0,0;
    0,0,0,0,0,0,1,0,];
% F=[13/9,-1/9,-1/9,-2/9,0,0,0,0;
%     1,0,0,0,0,0,0,0;
%     0,1,0,0,0,0,0,0;
%     0,0,1,0,0,0,0,0;
%     0,0,0,0,13/9,-1/9,-1/9,-2/9,;
%     0,0,0,0,1,0,0,0;
%     0,0,0,0,0,1,0,0;
%     0,0,0,0,0,0,1,0,];
H=eye(8);
Length=8;
Z0=[];
for i=1:Length/2
    Z0=[Z0;[Z(1,1)*ones(1,i-1),Z(1,1:end+1-i)]];
end
for i=1:Length/2
    Z0=[Z0;[Z(2,1)*ones(1,i-1),Z(2,1:end+1-i)]];
end
Xkv4=zeros(Length,N);%估计状态
Xkf=zeros(Length,N);%估计状态
Xkv4(:,1)=Z0(:,1);
Xkf(:,1)=Z0(:,1);
P0=eye(Length);
Q = Q0([1:Length/2,(4+1):(4+Length/2)],[1:Length/2,(4+1):(4+Length/2)]);
R = R0(1,1)*eye(Length);
for i=2:N
    Xkv4(:,i)=F*Xkf(:,i-1);%状态预测
    P1=F*P0*F'+Q;%协方差预测
    K=P1*H'*inv(H*P1*H'+R);%计算Kalman增益
    Xkf(:,i)=Xkv4(:,i)+K*(Z0(:,i)-H*Xkv4(:,i));%状态更新
    P0=(eye(Length)-K*H)*P1;%协方差更新
    for j=2:Length/2
        R(j,j) = P0(j-1,j-1);
        R(Length/2+j,Length/2+j) = P0(Length/2+j-1,Length/2+j-1);
    end
end
for i=1:N
    Err_Kv4(:,i)=RMS(X(:,i),Xkv4([1,1+Length/2],i));
end
%% kalman滤波四状态线性回归系数
F=[1.2668,-0.0152,0.0103,-0.2618,0,0,0,0;
    1,0,0,0,0,0,0,0;
    0,1,0,0,0,0,0,0;
    0,0,1,0,0,0,0,0;
    0,0,0,0,1.2668,-0.0152,0.0103,-0.2618,;
    0,0,0,0,1,0,0,0;
    0,0,0,0,0,1,0,0;
    0,0,0,0,0,0,1,0,];
% F=[13/9,-1/9,-1/9,-2/9,0,0,0,0;
%     1,0,0,0,0,0,0,0;
%     0,1,0,0,0,0,0,0;
%     0,0,1,0,0,0,0,0;
%     0,0,0,0,13/9,-1/9,-1/9,-2/9,;
%     0,0,0,0,1,0,0,0;
%     0,0,0,0,0,1,0,0;
%     0,0,0,0,0,0,1,0,];
H=eye(8);
Length=8;
Z0=[];
for i=1:Length/2
    Z0=[Z0;[Z(1,1)*ones(1,i-1),Z(1,1:end+1-i)]];
end
for i=1:Length/2
    Z0=[Z0;[Z(2,1)*ones(1,i-1),Z(2,1:end+1-i)]];
end
Xkr4=zeros(Length,N);%估计状态
Xkf=zeros(Length,N);%估计状态
Xkr4(:,1)=Z0(:,1);
Xkf(:,1)=Z0(:,1);
P0=eye(Length);
Q = Q0([1:Length/2,(4+1):(4+Length/2)],[1:Length/2,(4+1):(4+Length/2)]);
R = R0(1,1)*eye(Length);
for i=2:N
    Xkr4(:,i)=F*Xkf(:,i-1);%状态预测
    P1=F*P0*F'+Q;%协方差预测
    K=P1*H'*inv(H*P1*H'+R);%计算Kalman增益
    Xkf(:,i)=Xkr4(:,i)+K*(Z0(:,i)-H*Xkr4(:,i));%状态更新
    P0=(eye(Length)-K*H)*P1;%协方差更新
    for j=2:Length/2
        R(j,j) = P0(j-1,j-1);
        R(Length/2+j,Length/2+j) = P0(Length/2+j-1,Length/2+j-1);
    end
end
for i=1:N
    Err_Kr4(:,i)=RMS(X(:,i),Xkr4([1,1+Length/2],i));
end
%% kalman滤波四状态实时线性回归系数
F=[1.2668,-0.0152,0.0103,-0.2618,0,0,0,0;
    1,0,0,0,0,0,0,0;
    0,1,0,0,0,0,0,0;
    0,0,1,0,0,0,0,0;
    0,0,0,0,1.2668,-0.0152,0.0103,-0.2618,;
    0,0,0,0,1,0,0,0;
    0,0,0,0,0,1,0,0;
    0,0,0,0,0,0,1,0,];
H=[1,0,0,0,0,0,0,0;0,0,0,0,1,0,0,0];
Length=8;
RLength = 50;
Z0 = Z;
Xktr4=zeros(Length,N);%估计状态
Xkf=zeros(Length,N);%估计状态
Xktr4(:,1)=[ones(4,1)*Z0(1,1);ones(4,1)*Z0(2,1)];
Xkf(:,1)=Xktr4(:,1);
P0=eye(Length);
Q = Q0([1:Length/2,(4+1):(4+Length/2)],[1:Length/2,(4+1):(4+Length/2)]);
R = R0(1,1)*eye(2);
for i=2:N
    Xktr4(:,i)=F*Xkf(:,i-1);%状态预测
    P1=F*P0*F'+Q;%协方差预测
    K=P1*H'*inv(H*P1*H'+R);%计算Kalman增益
    Xkf(:,i)=Xktr4(:,i)+K*(Z0(:,i)-H*Xktr4(:,i));%状态更新
    P0=(eye(Length)-K*H)*P1;%协方差更新
    if (i>(RLength+5))
        ReX = regress(Z0(1,(i-RLength):i)',[Z0(1,(i-1-RLength):(i-1));Z0(1,(i-2-RLength):(i-2));Z0(1,(i-3-RLength):(i-3));Z0(1,(i-4-RLength):(i-4))]');
        ReY = regress(Z0(2,(i-RLength):i)',[Z0(2,(i-1-RLength):(i-1));Z0(2,(i-2-RLength):(i-2));Z0(2,(i-3-RLength):(i-3));Z0(2,(i-4-RLength):(i-4))]');
        F(1,1:4) = ReX;
        F(5,5:8) = ReY;
    end
end
for i=1:N
    Err_Ktr4(:,i)=RMS(X(:,i),Xktr4([1,1+Length/2],i));
end
%% 绘图
figure(1)
hold on;box on;
% plot(X(1,:),'-k');
% plot(Z(1,:),'k.');
% plot(Xkp(1,:),'-b.');
% plot(Xkv(1,:),'-r.');
% plot(Xka(1,:),'-y.');
% plot(Xkaa(1,:),'-g.');
% plot(Xk2(1,:),'-b+');
% plot(Xk3(1,:),'-r+');
% plot(Xk4(1,:),'-y+');
% plot(Xkv4(1,:),'-g+');
% plot(Xktr4(1,:),'-mo');

% plot(-X(5,:)+max(X(5,:))+min(X(5,:)),'-k');
% plot(Z(2,:),'k.');
% plot(Xkp(2,:),'-b.');
% plot(Xkv(3,:),'-r.');
% plot(Xka(4,:),'-y.');
% plot(Xkaa(5,:),'-g.');
% plot(Xk2(3,:),'-b+');
% plot(Xk3(4,:),'-r+');
% plot(Xk4(5,:),'-y+');
% plot(Xkv4(5,:),'-g+');

plot(X(1,:),X(5,:),'-k');
plot(Z(1,:),Z(2,:),'k.');
plot(Xkp(1,:),Xkp(2,:),':b+');
plot(Xkv(1,:),Xkv(3,:),':r+');
plot(Xka(1,:),Xka(4,:),':c+');
plot(Xkaa(1,:),Xkaa(5,:),':g+');
plot(Xk2(1,:),Xk2(3,:),'-b*');
plot(Xk3(1,:),Xk3(4,:),'-r*');
plot(Xk4(1,:),Xk4(5,:),'-c*');
plot(Xkv4(1,:),Xkv4(5,:),'-go');
plot(Xkr4(1,:),Xkr4(5,:),'-ro');
plot(Xktr4(1,:),Xktr4(5,:),'-mo');
% legend('真实轨迹','观测值','位置预测','位置速度预测','位置速度加速度预测','位置速度加速度加加速度预测','双位置预测','三位置预测','四位置预测','四位置变系数预测','四位置线性回归系数预测','Location','SouthEast');
% xlabel('横坐标X/m');
% ylabel('纵坐标Y/m');
legend('True Trace','Observation','Estimator with p','Estimator with p,v','Estimator with p,v,a','Estimator with p,v,a,aa','E2P','E3P','E4P','E4PVW','E4PRW','E4PTRW','Location','SouthEast');
xlabel('X/m');
ylabel('Y/m');
axes('Position',[0.18,0.5,0.35,0.4]); % 生成子图 最简单的方式
hold on;box on;
plot(X(1,200:220),X(5,200:220),'-k');
plot(Z(1,200:220),Z(2,200:220),'k.');
plot(Xkp(1,200:220),Xkp(2,200:220),':b+');
plot(Xkv(1,200:220),Xkv(3,200:220),':r+');
plot(Xka(1,200:220),Xka(4,200:220),':c+');
plot(Xkaa(1,200:220),Xkaa(5,200:220),':g+');
plot(Xk2(1,200:220),Xk2(3,200:220),'-b*');
plot(Xk3(1,200:220),Xk3(4,200:220),'-r*');
plot(Xk4(1,200:220),Xk4(5,200:220),'-c*');
plot(Xkv4(1,200:220),Xkv4(5,200:220),'-go');
plot(Xkr4(1,200:220),Xkr4(5,200:220),'-ro');
plot(Xktr4(1,200:220),Xktr4(5,200:220),'-o');

figure(2)
hold on;box on;
AErr_Ob=sqrt(Err_Ob(1,:).^2+Err_Ob(2,:).^2);
AErr_Kp=sqrt(Err_Kp(1,:).^2+Err_Kp(2,:).^2);
AErr_Kv=sqrt(Err_Kv(1,:).^2+Err_Kv(2,:).^2);
AErr_Ka=sqrt(Err_Ka(1,:).^2+Err_Ka(2,:).^2);
AErr_Kaa=sqrt(Err_Kaa(1,:).^2+Err_Kaa(2,:).^2);
AErr_K2=sqrt(Err_K2(1,:).^2+Err_K2(2,:).^2);
AErr_K3=sqrt(Err_K3(1,:).^2+Err_K3(2,:).^2);
AErr_K4=sqrt(Err_K4(1,:).^2+Err_K4(2,:).^2);
AErr_Kv4=sqrt(Err_Kv4(1,:).^2+Err_Kv4(2,:).^2);
AErr_Kr4=sqrt(Err_Kr4(1,:).^2+Err_Kr4(2,:).^2);
AErr_ktr4=sqrt(Err_Ktr4(1,:).^2+Err_Ktr4(2,:).^2);
for i=2:N
    AErr_Ob(i) = AErr_Ob(i)+AErr_Ob(i-1);
    AErr_Kp(i) = AErr_Kp(i)+AErr_Kp(i-1);
    AErr_Kv(i) = AErr_Kv(i)+AErr_Kv(i-1);
    AErr_Ka(i) = AErr_Ka(i)+AErr_Ka(i-1);
    AErr_Kaa(i) = AErr_Kaa(i)+AErr_Kaa(i-1);
    AErr_K2(i) = AErr_K2(i)+AErr_K2(i-1);
    AErr_K3(i) = AErr_K3(i)+AErr_K3(i-1);
    AErr_K4(i) = AErr_K4(i)+AErr_K4(i-1);
    AErr_Kv4(i) = AErr_Kv4(i)+AErr_Kv4(i-1);
    AErr_Kr4(i) = AErr_Kr4(i)+AErr_Kr4(i-1);
    AErr_ktr4(i) = AErr_ktr4(i)+AErr_ktr4(i-1);
end
% plot(20*log(AErr_Ob),'-k');'观测误差',
plot(20*log(AErr_Kp),':b+');
plot(20*log(AErr_Kv),':r+');
plot(20*log(AErr_Ka),':c+');
plot(20*log(AErr_Kaa),':g+');
plot(20*log(AErr_K2),'-b*');
plot(20*log(AErr_K3),'-r*');
plot(20*log(AErr_K4),'-c*');
plot(20*log(AErr_Kv4),'-go');
plot(20*log(AErr_Kr4),'-ro');
plot(20*log(AErr_ktr4),'-mo');
% legend('位置预测误差','位置速度预测误差','位置速度加速度预测误差','位置速度加速度加加速度预测误差','双位置预测误差','三位置预测误差','四位置预测误差','四位置变系数预测误差','四位置线性回归系数预测误差','四位置实时线性回归系数预测误差','Location','Southwest');
% xlabel('仿真次数');
% ylabel('累计误差');
legend('Estimator with p','Estimator with p,v','Estimator with p,v,a','Estimator with p,v,a,aa','E2P','E3P','E4P','E4PVW','E4PRW','E4PTRW','Location','Southwest');
xlabel('Simulation Steps');
ylabel('Accumulation Error');

axes('Position',[0.6,0.15,0.3,0.4]); % 生成子图 最简单的方式
hold on;box on;
plot(990:1000,20*log(AErr_Kp(990:1000)),':b+');
plot(990:1000,20*log(AErr_Kv(990:1000)),':r+');
plot(990:1000,20*log(AErr_Ka(990:1000)),':c+');
plot(990:1000,20*log(AErr_Kaa(990:1000)),':g+');
plot(990:1000,20*log(AErr_K2(990:1000)),'-b*');
plot(990:1000,20*log(AErr_K3(990:1000)),'-r*');
plot(990:1000,20*log(AErr_K4(990:1000)),'-c*');
plot(990:1000,20*log(AErr_Kv4(990:1000)),'-go');
plot(990:1000,20*log(AErr_Kr4(990:1000)),'-ro');
plot(990:1000,20*log(AErr_ktr4(990:1000)),'-mo');
function dist=RMS(X1,X2)
    dist(1)=abs((X1(1)-X2(1))^2);
    dist(2)=abs((X1(5)-X2(2))^2);
end
