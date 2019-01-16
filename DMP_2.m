%% Learn the DMP parameters of the Inverse Model Output and adjust the control input to reach the target

clear all
clc

%% Load the data of the inverse Models

id=3;
name='DMP_';
name_id=strcat(name,num2str(id));
full_name=strcat(name_id,'.mat');
Data=load(full_name);

convert=1000;

%% Load the time vector
time=Data.DMP.time./convert;
time=time-time(1);
dt=diff(time);
dt=dt(1);

%% Load the joint trajectories
q=Data.DMP.q;
dq=vel(q)./dt;
ddq=vel(dq)./dt;
s=size(q);

%% Get the DMP parameters for every joint trajectory 
for i=1:s(2)
    par(i)=DMP_par(q(:,i),dq(:,i),ddq(:,i),time,dt);
end

%% Get the DMP parameters for every joint trajectory 
for i=1:s(2)
    result(i)=DMP_Generate(q(:,i),dq(:,i),ddq(:,i),time,dt,q(end,i),par(i));
end 

%% Get the Cartesian History
cartesian=zeros(s(1),3);

for i=1:s(1)
    %joints=[result(1).y_xr(i) result(2).y_xr(i) 0 result(3).y_xr(i) result(4).y_xr(i) 0 0];
    joints=[q(i,1) q(i,2) 0 q(i,3) q(i,4) 0 0];
    [TeR,TrR,TR]=getSawyerFK_R(joints);
    [x,y,z]=MyTransl(TeR);
    cartesian(i,:)=[x,y,z];    
end

%% Reinforcement Learning

%% Initialize RL parameters

learnRate = 0.9; %learning rate
epsilon = 0.5; %initial value of exploration
epsilonDecay = 0.98; % Decay factor per iteration.
discount = 0.9; % Future vs present value
successRate = 1; % Inject some noise
win = 100;  % Bonus at the goal state
start = [q(end,1) q(end,2) q(end,3) q(end,4)]; % Initial state

maxEpi = 200; % Number of Episodes.
maxit = 100; % Interactions per Episodes.
res=10; % resolution of the joint angles

%% RL variables

% Generate action list
actions = [0, -1, 1];

% Generate a state list
temp=linspace(-pi,q(end,1),res);
q1 = [temp linspace(q(end,1),pi,res)];

temp=linspace(-pi,q(end,2),res);
q2 = [temp linspace(q(end,2),pi,res)];

temp=linspace(-pi,q(end,3),res);
q3 = [temp linspace(q(end,3),pi,res)];

temp=linspace(-pi,q(end,4),res);
q4 = [temp linspace(q(end,4),pi,res)];

states=zeros(length(q1)*length(q2)*length(q3)*length(q4),4);
index=1;
for a=1:length(q1)
    for b = 1:length(q2)
        for c= 1:length(q3)
            for d= 1:length(q4)
                states(index,1)=q1(a);
                states(index,2)=q2(b);
                states(index,3)=q3(c);
                states(index,4)=q4(d);
                index=index+1;
            end
        end
    end
end

% Initialize the reward function
R=zeros(1,length(states));
for i=1:length(states)
    [TeR,~,~]=getSawyerFK_R([states(i,1) states(i,2) 0 states(i,3) states(i,4) 0 0]);
    [x,y,z]=MyTransl(TeR);
    [R(i)]=Reward(Data.DMP.target,[x,y,z]);
end

% Initialize Q value
Q = repmat(R,length(actions));

% Matrix that contains best actions per state
V = zeros(size(states,1),1);

% Find the state with the best reward
index=find(R==max(R));

%% Get the DMP parameters for the most optimum state
for i=1:s(2)
    new(i)=DMP_Generate(q(:,i),dq(:,i),ddq(:,i),time,dt,states(index(1),i),par(i));
end

%% Get the Cartesian History
new_cartesian=zeros(s(1),3);

for i=1:s(1)
    joints=[new(1).y_xr(i) new(2).y_xr(i) 0 new(3).y_xr(i) new(4).y_xr(i) 0 0];
    [TeR,TrR,TR]=getSawyerFK_R(joints);
    [x,y,z]=MyTransl(TeR);
    new_cartesian(i,:)=[x,y,z];    
end

%% Start Training

% message for debugging purposes
disp('Training Started')

% Number of episodes
% for episodes = 1:1%maxEpi
% 
%     z1 = start; % Initial state
%     
%     for g = 1:1%maxit
%         
%        %% PICK AN ACTION
%         
%         % Interpolate the state 
%         [~,sIdx] = min(sum((states - repmat(z1,[size(states,1),1])).^2,2));
%         
%         % Choose an action:
%         if (rand()>epsilon || episodes == maxEpi) && rand()<=successRate
%             [~,aIdx] = max(Q(:,sIdx)); % Pick the action the Q matrix thinks is best
%         else
%             aIdx = randi(length(actions),1); % Random action
%         end
%         
%         T = actions(aIdx); % Final action
         
         %% Calculate the new state of the system
%         
%         % Step the dynamics forward with our new action choice
%         % RK4 Loop - Numerical integration
%         for i = 1:substeps
%             k1 = Dynamics(z1,T);
%             k2 = Dynamics(z1+dt/2*k1,T);
%             k3 = Dynamics(z1+dt/2*k2,T);
%             k4 = Dynamics(z1+dt*k3,T);
%             
%             z2 = z1 + dt/6*(k1 + 2*k2 + 2*k3 + k4);
%             % All states wrapped to 2pi
%             if z2(1)>pi
%                 z2(1) = -pi + (z2(1)-pi);
%             elseif z2(1)<-pi
%                 z2(1) = pi - (-pi - z2(1));
%             end
%         end
%         
%         z1 = z2; % Old state = new state
%         
%         
%         %% UPDATE Q-MATRIX
%         
%         % End condition for an episode
%         if norm(z2)<0.01 % If we've reached upright with no velocity (within some margin), end this episode.
%             success = true;
%             bonus = winBonus; % Give a bonus for getting there.
%         else
%             bonus = 0;
%             success = false;
%         end
%         
%         [~,snewIdx] = min(sum((states - repmat(z1,[size(states,1),1])).^2,2)); % Interpolate again to find the new state the system is closest to.
%         
%         if episodes ~= maxEpi % On the last iteration, stop learning and just execute. Otherwise...
%             % Update Q
%             Q(sIdx,aIdx) = Q(sIdx,aIdx) + learnRate * ( R(snewIdx) + discount*max(Q(snewIdx,:)) - Q(sIdx,aIdx) + bonus );
%             
%             % Lets break this down:
%             %
%             % We want to update our estimate of the global value of being
%             % at our previous state s and taking action a. We have just
%             % tried this action, so we have some information. Here are the terms:
%             %   1) Q(sIdx,aIdx) AND later -Q(sIdx,aIdx) -- Means we're
%             %      doing a weighting of old and new (see 2). Rewritten:
%             %      (1-alpha)*Qold + alpha*newStuff
%             %   2) learnRate * ( ... ) -- Scaling factor for our update.
%             %      High learnRate means that new information has great weight.
%             %      Low learnRate means that old information is more important.
%             %   3) R(snewIdx) -- the reward for getting to this new state
%             %   4) discount * max(Q(snewIdx,:)) -- The estimated value of
%             %      the best action at the new state. The discount means future
%             %      value is worth less than present value
%             %   5) Bonus - I choose to give a big boost of it's reached the
%             %      goal state. Optional and not really conventional.
%         end
%         
%         % Decay the odds of picking a random action vs picking the
%         % estimated "best" action. I.e. we're becoming more confident in
%         % our learned Q.
%         epsilon = epsilon*epsilonDecay;
        
        % End this episode if we reach the goal.
%         if success
%             break;
%         end   
%     end
%     
% end


%% Plotting functions

% Plot joint histories
figure(1);
subplot(4,1,1)
hold on
grid on
plot(time,q(:,1),'b')
plot(time,result(1).y_xr,'--r')
plot(time,new(1).y_xr,'--k')
hold off
subplot(4,1,2)
hold on
grid on
plot(time,q(:,2),'b')
plot(time,result(2).y_xr,'--r')
plot(time,new(2).y_xr,'--k')
hold off
subplot(4,1,3)
hold on
grid on
plot(time,q(:,3),'b')
plot(time,result(3).y_xr,'--r')
plot(time,new(3).y_xr,'--k')
hold off
subplot(4,1,4)
hold on
grid on
plot(time,q(:,4),'b')
plot(time,result(4).y_xr,'--r')
plot(time,new(3).y_xr,'--k')
hold off

% Plot cartesian histories
figure(2)
hold on
plot3(cartesian(:,1), cartesian(:,2), cartesian(:,3),'r')
plot3(Data.DMP.target(1),Data.DMP.target(2),Data.DMP.target(3),'*r')
plot3(new_cartesian(:,1), new_cartesian(:,2), new_cartesian(:,3),'k')
grid on
hold off

%% Define the reward function
function [R]=Reward(target,current)
    
    temp=target-current;
    R=-sqrt(temp*temp');

end

%% Function that returns the parameters of a DMP 
function [par]=DMP_par(x,dx,ddx,time,dt)
    
%% Initialize the parameters 

x=x; %position
time=time-time(1); %time vector
dt=dt(1); %time step
dx=dx; % velocity
ddx=ddx; % acceleration
x0=x(1); % initial state
gx=x(end); % final state

par.ng=100; % Number of Gaussians
h=1;
par.h=ones(1,par.ng)*(h); % width of the Gaussians
par.s=1; % Init of phase
par.as=1; % Decay of s phase var
par.tau=max(time); % Time scaling
par.K=200; % K gain
par.D=10; % D gain

%% Compute ftarget

stime=[];
sE_x=[];

for i=1:length(time)
    
    t=time(i);
    s=exp((-1*par.as*t)/par.tau);
    stime=[stime s];
    
    % fdemonstration=ftarget
    ftarget_x(i)= (-1*par.K*(gx-x(i))+par.D*dx(i)+par.tau*ddx(i))/(gx-x0);
    
    sE_x=[sE_x; s*(gx-x0)];
    
end

% centers of gaussian are placed even in s time
% gaussian centers are distributed evenly in s axis.
incr=(max(stime)-min(stime))/(par.ng-1);
c=min(stime):incr:max(stime);
lrc=fliplr(c);
ctime=(-1*par.tau*log(lrc))/par.as;
d=diff(c);
c=c/d(1); % normalize for exp correctness
par.c=c;

% Regression
for i=1:par.ng
    psV_x=[];
    for j=1:length(time)
        psV_x=[psV_x psiF(par.h,par.c,stime(j)/d(1),i)];
    end
    %Locally Weighted Learning
    w_x(i)=(transpose(sE_x)*diag(psV_x)*transpose(ftarget_x))/(transpose(sE_x)*diag(psV_x)*sE_x);
end

%% Store the results
par.time=time;
par.stime=stime;
par.ftarget_x=ftarget_x;
par.w_x=w_x;
par.x0=x0;
par.gx=gx;
par.d1=d(1);
par.dt=dt;
par.ctime=ctime;

end

%% Function that generates a DMP 
function [result]=DMP_Generate(x,dx,ddx,time,dt,target,par)

% Arrays to store the estimated trajectories
f_replay_x=[];
fr_x_zeros=[];

ydd_x_r=0;
yd_x_r=0;
y_x_r=par.x0;
dtx=par.dt;

for j=1:length(par.time)
    psum_x=0;
    pdiv_x=0;
    for i=1:par.ng

        % Normalizing the center of gaussians
        psum_x=psum_x+psiF(par.h, par.c, par.stime(j)/par.d1,i)*par.w_x(i);
        pdiv_x=pdiv_x+psiF(par.h, par.c, par.stime(j)/par.d1,i);
        
    end

    % Generate the new trajectories according to the new control input
    f_replay_x(j)=(psum_x/pdiv_x)*par.stime(j)*(target-par.x0);
    
    if(j>1)
        if sign(f_replay_x(j-1))~=sign(f_replay_x(j))
            fr_x_zeros=[fr_x_zeros j-1];
        end
    end
    
    % Apply the formula tau*ydd= K(g-x)-Dv+(g-x0)f
    ydd_x_r=(par.K*(target-y_x_r)-(par.D*yd_x_r)+(target-par.x0)*f_replay_x(j))/par.tau;
    yd_x_r= yd_x_r+ (ydd_x_r*dtx)/par.tau;
    y_x_r= y_x_r+ (yd_x_r*dtx)/par.tau;
    
    ydd_xr(j)=ydd_x_r;
    yd_xr(j)=yd_x_r;
    y_xr(j)=y_x_r;
    
end

%% Store the estimated DMP
result.ydd_xr=ydd_xr;
result.yd_xr=yd_xr;
result.y_xr=y_xr;
result.fr_x_zeros=fr_x_zeros;
result.f_replay_x=f_replay_x;

end

%% Function that calculates velocities
function [dq]=vel(q)

s=size(q);
dq=zeros(s(1),s(2));

for j=1:s(2)
    dq(:,j)=[0;diff(q(:,j))];
end

end

%% My Gaussian function
function r=psiF(h, c, s, i)
r=exp(-h(i)*(s-c(i))^2);
end
