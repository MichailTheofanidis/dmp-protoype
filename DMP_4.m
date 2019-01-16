%% Sawyer DMP Simulation

clc
clearvars -except jointPub
close all

%% Subscribe
% sub1 = rossubscriber('/inverse_model_prediction');
% sub2 = rossubscriber('/inverse_model_flag');

%% Variables to slow down
target=[0.3 -0.4 -0.07]; % for DMP 1
%target=[-0.3 -0.4 -0.07]; % for DMP 2
%target=[0.3 -0.6 -0.07]; % for DMP 3
%target=[-0.3 -0.6 -0.07]; % for DMP 4

% counter=0;
% p=[];
% q=[];
% flag=1;
% 
% while(flag==1)
%     
%     
%     %% Increase the counter
%     counter=counter+1;
%     
%     %% Receive Data
%     data=receive(sub1);
%     d=receive(sub2);
%     
%     %% Assign the Joints
%     joints=[data.Position(1) data.Position(2) 0 data.Position(3) data.Position(4) 0 0];
%     
%     %% Update the flag
%     flag=str2num(d.Data)
%     
%     if flag==1
%         
%         %% Extract the time vector
%         dt=data.Header.Stamp.Sec/1e+09;
% 
%         if counter==1
%             time(counter)=0;
%         else
%             time(counter)=time(counter-1)+dt;
%         end
%         
%         %% Calculate the real position and store it
%         [TeR,TrR,TR]=getSawyerFK_R(joints);
%         disp(TeR)
%         [x, y, z]=MyTransl(TeR);
%         
%         %% Store the positions and joint angles
%         q(counter,:)=[joints(1), joints(2), joints(4), joints(5)];
%         p(counter,:)=[x, y, z];
%         
%         %% Display the Robot
%         cla
%         figure(1)
%         hold on
%         plot3(target(1), target(2), target(3),'*r')
%         plot3(p(1:counter,1), p(1:counter,2), p(1:counter,3),'r')
%         plotSawyer(TR)
%         grid on
%         drawnow
%     end
%     
% end
% 
% %% Pause the simulation 
% pause()

%% Close all the figures
close all

%% Load the Data
load('Data.mat');

%% Create a DMP from the predicted trajectory

%% Modification parameters
convert=1000;
int_dim=200;

%% Modify the time vector
time=interp1([0:1:length(Data.time)-1],Data.time,linspace(0,length(Data.time)-1,int_dim))./convert;
time=time-time(1);
dt=time(2)-time(1);

%% Load the joint trajectories
s=size(Data.q);

q=interp1([0:1:length(Data.q)-1],Data.q,linspace(0,length(Data.q)-1,int_dim));
dq=vel(q)./dt;
ddq=vel(dq)./dt;

%% Perform the DMP operation

for i=1:s(2)
    
    [par(i)]=Train_DMP(q(:,i),dq(:,i),ddq(:,i),time);
    [res(i)]=forward_DMP(par(i),q(end,i));
    
end

%% Perform the learned DMP
learned_positions=zeros(int_dim,3);

for i=1:int_dim
    joints=[res(1).y_xr(i) res(2).y_xr(i) 0 res(3).y_xr(i) res(4).y_xr(i) 0 0];
    [TeR,TrR,TR]=getSawyerFK_R(joints);
    [x,y,z]=MyTransl(TeR);
    learned_positions(i,:)=[x,y,z];   
end

%% Use RL to find the new Target

% Generate the exploration space for the target
step=5;
temp=linspace(pi/6,joints(end,1),step);
q1 = [temp linspace(q(end,1),-pi,step)];

temp=linspace(pi/6,joints(end,2),step);
q2 = [temp linspace(q(end,2),pi/3,step)];

temp=linspace(pi/6,joints(end,3),step);
q3 = [temp linspace(q(end,3),pi/3,step)];

temp=linspace(pi/6,joints(end,4),step);
q4 = [temp linspace(q(end,4),pi/3,step)];

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
    [R(i)]=RewardIK(Data.target,[x,y,z]);
end

% Find the state with the best reward
index=find(R==max(R));

% Define the joint targets
joint_targets=states(index,:);

%%

% Generate the exploration space for the target
step=7;
range=7;
temp=linspace(joint_targets(1)-deg2rad(range),joint_targets(1),step);
q1 = [temp linspace(q(end,1),joint_targets(1)+deg2rad(range),step)];

temp=linspace(joint_targets(2)-deg2rad(range),joint_targets(2),step);
q2 = [temp linspace(q(end,2),joint_targets(2)+deg2rad(range),step)];

temp=linspace(joint_targets(3)-deg2rad(range),joint_targets(3),step);
q3 = [temp linspace(q(end,3),joint_targets(3)+deg2rad(range),step)];

temp=linspace(joint_targets(4)-deg2rad(range),joint_targets(4),step);
q4 = [temp linspace(q(end,4),joint_targets(4)+deg2rad(range),step)];

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
    [R(i)]=RewardIK(Data.target,[x,y,z]);
end

% Find the state with the best reward
index=find(R==max(R));

% Define the joint targets
joint_targets=states(index,:);

%%


% Forward Pass to get the new States 
for i=1:s(2)
    
    [res2(i)]=forward_DMP(par(i),joint_targets(i));
    
end

% Results of the forward Pass in cartesian space
estimated_positions=zeros(int_dim,3);

for i=1:int_dim
    joints=[res2(1).y_xr(i) res2(2).y_xr(i) 0 res2(3).y_xr(i) res2(4).y_xr(i) 0 0];
    [TeR,TrR,TR]=getSawyerFK_R(joints);
    [x,y,z]=MyTransl(TeR);
    estimated_positions(i,:)=[x,y,z];   
end

%% RL to optimize the joint trajectories

y_final=zeros(s(2),int_dim);
counter=0;
for i=1:s(2)
    
    % Initial actions
    w_a=par(i).w_x';

    % Initialize the action array
    flag=0;
    samples=10;
    sampling_rate=0.5;

    actions=ones(par(i).ng,samples);
    
    while flag==0

        % Set the expolration parameter
        expl=zeros(par(i).ng,samples);

        for z=1:samples
            for j=1:par(i).ng
                expl(j,z)=normrnd(0,std2(par(i).psV_x(j,:)*par(i).w_x(j)));
            end
        end

        % Sample all the possible actions
        for j=1:samples

            actions(:,j)=(w_a+expl(:,j));

        end

        % Generate new rollouts
        ydd_xr=zeros(samples,length(par(i).time));
        yd_xr=zeros(samples,length(par(i).time));
        y_xr=zeros(samples,length(par(i).time));

        for g=1:samples

            ydd_x_r=0;
            yd_x_r=0;
            y_x_r=par(i).x(1);

            for j=1:length(par(i).time)

                psum_x=0;
                pdiv_x=0;

                for z=1:par(i).ng

                    % Normalizing the center of gaussians
                    psum_x=psum_x+par(i).psV_x(z,j)*actions(z,i);
                    pdiv_x=pdiv_x+par(i).psV_x(z,j);

                end

                % Generate the new trajectories according to the new control input
                f_replay_x(j)=(psum_x/pdiv_x)*par(i).stime(j)*(joint_targets(i)-par(i).x(1));

                % Apply the formula tau*ydd= K(g-x)-Dv+(g-x0)f
                ydd_x_r=(par(i).K*(joint_targets(i)-y_x_r)-(par(i).D*yd_x_r)+(joint_targets(i)-par(i).x(1))*f_replay_x(j))/par(i).tau;
                yd_x_r= yd_x_r+ (ydd_x_r*par(i).dt)/par(i).tau;
                y_x_r= y_x_r+ (yd_x_r*par(i).dt)/par(i).tau;

                ydd_xr(g,j)=ydd_x_r;
                yd_xr(g,j)=yd_x_r;
                y_xr(g,j)=y_x_r;

            end
        end

        % Estimate the Q values
        Q=zeros(1,samples);

        for z=1:samples
            sum=0;
            for j=1:length(time)

                sum=sum+Myreward(joint_targets(i),y_xr(z,j),par(i).time(j),par(i).tau);

                Q(z)=sum;

            end

        end

        % Sample the highest Q values to aptade the action parameters
        high=floor(sampling_rate*samples);
        [Q_sort,I] = sort(Q,'descend');

        % Update the action parameters
        sumQ=0;
        sumQ_up=0;
        for j=1:high
            sumQ=sumQ+Q_sort(j);

            sumQ_up=expl(:,I(j)).*Q_sort(j);

        end

        y_xr(I(1),end);
        w_a=w_a+(sumQ_up)./(sumQ);

        counter=counter+1;
        
        if(abs(y_xr(I(1),end)-joint_targets(i))<0.01)
            flag=1;
            y_final(i,:)=y_xr(I(1),:);
        end

    end
    
end

%% Cartesian results of RL
final_positions=zeros(int_dim,3);

for i=1:int_dim
    joints=[y_final(1,i) y_final(2,i) 0 y_final(3,i) y_final(4,i) 0 0];
    [TeR,TrR,TR]=getSawyerFK_R(joints);
    [x,y,z]=MyTransl(TeR);
    final_positions(i,:)=[x,y,z];   
end

%% Ploting functions

fig=1;

figure(fig)
grid on
hold on
for i=1:s(2)
    plot(par(i).time,par(i).stime)
end
hold off

figure(fig+1)
grid on
for i=1:s(2)
    cg=jet(par(i).ng); %colors of the gaussians
    for j=1:par(i).ng
        subplot(s(2),1,i)
        hold on
        plot(par(i).stime,par(i).psV_x(j,:),'defaultAxesColorOrder',cg(i,:))
        hold off
    end
end

figure(fig+2)
grid on
for i=1:s(2)
    cg=jet(par(i).ng); %colors of the gaussians
    for j=1:par(i).ng
        subplot(s(2),1,i)
        hold on
        plot(par(i).stime,par(i).psV_x(j,:)*par(i).w_x(j),'defaultAxesColorOrder',cg(i,:))
        hold off
    end
end

figure(fig+3)
for i=1:s(2)
    subplot(s(2),1,i)
    grid on
    hold on
    plot(time,par(i).x,'b')
    plot(time,res(i).y_xr,'r')
    plot(time,res2(i).y_xr,'k')
    plot(time,y_final(i,:),'k')
    hold off
end

figure(fig+4)
grid on
hold on
axis equal
plot3(target(1),target(2),target(3),'*r')
plot3(learned_positions(:,1), learned_positions(:,2), learned_positions(:,3),'b')
plot3(estimated_positions(:,1), estimated_positions(:,2), estimated_positions(:,3),'k')
plot3(Data.p(:,1),Data.p(:,2),Data.p(:,3),'--b')
plot3(final_positions(:,1),final_positions(:,2),final_positions(:,3),'c')
hold off



%% Functions

%% Define the reward function
function [R]=RewardIK(target,current)
    
    temp=target-current;
    R=-sqrt(temp*temp');

end

%% My reward function
function [rwd]=Myreward(target,position,time,tau)

    w1=0.9; % weights of the two rewards
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

par.ng=30; % Number of Gaussians
h=1;
par.h=ones(1,par.ng)*(h); % width of the Gaussians
par.s=1; % Init of phase
par.as=6; % Decay of s phase var
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
fr_x_zeros=zeros(1,length(par.time));

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
    yd_x_r= yd_x_r+ (ydd_x_r*par.dt)/par.tau;
    y_x_r= y_x_r+ (yd_x_r*par.dt)/par.tau;
    
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



