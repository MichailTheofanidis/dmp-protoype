%% Learn the DMP parameters of the Inverse Model Output and adjust the control input to reach the target

close all
clear all
clc

%% Loard the data

load('Data/Data_1.mat');

%% Load the time vector
convert=1000;

% time=Data.DMP.time./convert;
time=Data.time./convert;

time=time-time(1);
dt=time(2)-time(1);

%% Load the joint trajectories
jnt_id=2;

target=-2.4898; % New goal position

q=Data.q(:,jnt_id)
dq=vel(q)./dt;
ddq=vel(dq)./dt;

%% Perform the DMP operation
[par]=Train_DMP(q,dq,ddq,time);
[res_1]=forward_DMP(par,q(end));

%% Start adjusting the parameters with Reinforcement Learning

[res_2]=forward_DMP(par,target); % Get the new state

% Initial actions
w_a=par.w_x';

% Initialize the action array
flag=0;
samples=10;
sampling_rate=0.5;

actions=ones(par.ng,samples);

counter=0;
while flag==0
    
    % Set the expolration parameter
    expl=zeros(par.ng,samples);
    %expl=normrnd(3,10,[par.ng,samples]);
    
    for i=1:samples
        for j=1:par.ng
            %expl(j,i)=normrnd(mean(par.psV_x(j,:)*par.w_x(j)),std2(par.psV_x(j,:)*par.w_x(j)));
            expl(j,i)=normrnd(0,std2(par.psV_x(j,:)*par.w_x(j)));
        end
    end
          
    % Sample all the possible actions
    for i=1:samples
       
        actions(:,i)=(w_a+expl(:,i));
        
    end
    
    % Generate new rollouts
    ydd_xr=zeros(samples,length(par.time));
    yd_xr=zeros(samples,length(par.time));
    y_xr=zeros(samples,length(par.time));
    
    for i=1:samples
        
        ydd_x_r=0;
        yd_x_r=0;
        y_x_r=par.x(1);

        for j=1:length(par.time)

            psum_x=0;
            pdiv_x=0;

            for z=1:par.ng

                % Normalizing the center of gaussians
                psum_x=psum_x+par.psV_x(z,j)*actions(z,i);
                pdiv_x=pdiv_x+par.psV_x(z,j);

            end

            % Generate the new trajectories according to the new control input
            f_replay_x(j)=(psum_x/pdiv_x)*par.stime(j)*(target-par.x(1));

            % Apply the formula tau*ydd= K(g-x)-Dv+(g-x0)f
            ydd_x_r=(par.K*(target-y_x_r)-(par.D*yd_x_r)+(target-par.x(1))*f_replay_x(j))/par.tau;
            yd_x_r= yd_x_r+ (ydd_x_r*par.dt)/par.tau;
            y_x_r= y_x_r+ (yd_x_r*par.dt)/par.tau;

            ydd_xr(i,j)=ydd_x_r;
            yd_xr(i,j)=yd_x_r;
            y_xr(i,j)=y_x_r;

        end
    end
    
    % Estimate the Q values
    Q=zeros(1,samples);
    
    for i=1:samples
        sum=0;
        for j=1:length(time)

            sum=sum+Myreward(target,y_xr(i,j),par.time(j),par.tau);
            
            Q(i)=sum;
            
        end
        
    end
    
    % Sample the highest Q values to aptade the action parameters
    high=floor(sampling_rate*samples);
    [Q_sort,I] = sort(Q,'descend');
    
    % Update the action parameters
    sumQ=0;
    sumQ_up=0;
    for i=1:high
        sumQ=sumQ+Q_sort(i);
        
        sumQ_up=expl(:,I(i)).*Q_sort(i);
        
    end
    
    y_xr(I(1),end)
    %w_a=w_a-(sumQ_up)./(sumQ);
    w_a=w_a+(sumQ_up)./(sumQ);
    
    counter=counter+1;
    
    if(abs(y_xr(I(1),end)-target)<0.01)
        flag=1;
    end
    
end

%% Plotting functions

fig=1;

figure(fig)
grid on
plot(time,par.stime,'b')

cg=jet(par.ng); %colors of the gaussians

figure(fig+1)
grid on
hold on
for i=1:par.ng
    plot(par.stime,par.psV_x(i,:),'defaultAxesColorOrder',cg(i,:))
end
hold off

figure(fig+2)
grid on
hold on
for i=1:par.ng
    plot(par.stime,par.psV_x(i,:)*par.w_x(i),'defaultAxesColorOrder',cg(i,:))
end
hold off

%Plot joint histories
figure(fig+3)
grid on
hold on
plot(time,par.x,'b')
plot(time,res_1.y_xr,'r')
plot(time,res_2.y_xr,'k')
for i=1:samples
    plot(time,y_xr(i,:),'k')
end
plot(time,y_xr(I(1),:),'r')
hold off

% Plot the final gaussians
figure(fig+4)
grid on
hold on
for i=1:samples
    plot(par.stime,par.psV_x(i,:)*par.w_x(i),'defaultAxesColorOrder',cg(i,:))
end
hold off

%% FUNCTIONS %%

%% My reward function
function [rwd]=Myreward(target,position,time,tau)

w1=0.9; % weights of the two rewards
threshold=0.01; % threshold for final time
temp=target-position; % distance between the target and current position

if(abs(time-tau)<0.01)
    rwd=(w1)*exp(-sqrt(temp*temp'));
else
    rwd=(1-w1)*exp(-sqrt(temp*temp'))/tau;
end

end

%% Locally Weighted Regression
function [par]=Train_DMP(q,dq,ddq,time)

%% Learn the Dynamic Motion Primitive
par.x=q; %position
par.dx=dq; % velocity
par.ddx=ddq; %acceleration
par.time=time; %time vector
par.dt=par.time(2)-par.time(1);

x0=par.x(1); %initial state
par.gx=par.x(end); %final state

par.ng=50; % Number of Gaussians
h=1;
par.h=ones(1,par.ng)*(h); % width of the Gaussians
par.s=1; % Init of phase
par.as=5; % Decay of s phase var
par.tau=max(time); % Time scaling
par.K=4000; % K gain
par.D=100; % D gain

par.stime=zeros(1,length(time)); % vector to store s
par.s_g=zeros(1,length(time)); % vector to store s with the goal
par.ftarget_x=zeros(1,length(time)); % vector to store f target

%% Compute ftarget
for i=1:length(time)
    
    par.stime(i)=exp((-1*par.as*time(i))/par.tau);
    
    % fdemonstration=ftarget
    par.ftarget_x(i)= (-1*par.K*(par.gx-par.x(i))+par.D*par.dx(i)+par.tau*par.ddx(i))/(par.gx-x0);
    par.s_g(i)=par.stime(i)*(par.gx-x0);
    
end

%% Centers of gaussians
incr=(par.stime(1)-par.stime(end))/(par.ng-1);
c=par.stime(end):incr:par.stime(1);
lrc=fliplr(c);
ctime=(-1*par.tau*log(lrc))/par.as;
d=diff(c);
par.c=c/d(1); % normalize for exp correctness

%% Place the gaussians
par.psV_x=zeros(par.ng,length(par.stime));

for i=1:par.ng
    
    for j=1:length(par.stime)
        par.psV_x(i,j)=psiF(par.h,par.c,par.stime(j)/d(1),i);
    end
    
    %Perform the Regression
    par.w_x(i)=(transpose(par.s_g')*diag(par.psV_x(i,:))*transpose(par.ftarget_x))/(transpose(par.s_g')*diag(par.psV_x(i,:))*par.s_g');
    
end


end

%% Code that Generates a DMP
function [res]=forward_DMP(par,target)

f_replay_x=zeros(1,length(par.time));

ydd_x_r=0;
yd_x_r=0;
y_x_r=par.x(1);

for j=1:length(par.time)
    
    psum_x=0;
    pdiv_x=0;
    
    for i=1:par.ng
        
        % Normalizing the center of gaussians
        psum_x=psum_x+par.psV_x(i,j)*par.w_x(i);
        pdiv_x=pdiv_x+par.psV_x(i,j);
        
    end
    
    % Generate the new trajectories according to the new control input
    f_replay_x(j)=(psum_x/pdiv_x)*par.stime(j)*(target-par.x(1));
       
    % Apply the formula tau*ydd= K(g-x)-Dv+(g-x0)f
    ydd_x_r=(par.K*(target-y_x_r)-(par.D*yd_x_r)+(target-par.x(1))*f_replay_x(j))/par.tau;
    yd_x_r= yd_x_r+(ydd_x_r*par.dt)/par.tau;
    y_x_r= y_x_r+(yd_x_r*par.dt)/par.tau;
    
    res.ydd_xr(j)=ydd_x_r;
    res.yd_xr(j)=yd_x_r;
    res.y_xr(j)=y_x_r;
    
end

end

%% My Gaussian function
function r=psiF(h, c, s, i)
r=exp(-h(i)*(s-c(i))^2);
end

%% Function that calculates velocities
function [dq]=vel(q)

s=size(q);
dq=zeros(s(1),s(2));

for j=1:s(2)
    dq(:,j)=[0;diff(q(:,j))];
end

end