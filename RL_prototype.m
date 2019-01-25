%% Tuning DMP parameters with Reinforcement Learning
%% by Michail Theofanidis

clear all
clc

%% Import data from demonstration

% id of the data
id=9;
jnt_id=1;
name='../Demos/demo';
name=strcat(name,num2str(id));
name=strcat(name,'.txt');
Demo=importdata(name);

%% Reinforcement Learning parameters
goal=2.6;
samples=10;
rate=0.5;

%% Create the joint and time vector
data=Demo.data(:,2:8);
t=Demo.data(:,1);

% Normalize time vector
t=linspace(0,t(end),length(t));

% Select the position vector
q=data(:,jnt_id);

%% Filtering parameters
r=50;
blends=10;

%% Compute velocity and acceleration
q=smooth(q);
dq=vel(q,t);
ddq=vel(dq,t);

%% Perform filtering    
fq=PolyTraj(q,dq,t,blends);
fdq=vel(fq,t);
fddq=vel(fdq,t);
    
%% Initialize the dmps
dmp=Dmp();

%% Calculate the phase of the dmp
s=dmp.phase(t);

%% Calculate the spread out gaussians
psV=dmp.distributions(s);

%% Perform immitation learning on all degrees of freedom
[ftarget,w]=dmp.immitate(fq,fdq,fddq,t,s,psV);

%% Perform forward pass of the DMP
[x,dx,ddx]=dmp.generate(w,fq(1),fq(end),t,s,psV);

%% Adapt the DMP with Reinforcement Learning
[x_r,dx_r,ddx_r]=dmp.adapt(w',fq(1),goal,t,s,psV,samples,rate);

%% Ploting functions

figure(1)
hold on
grid on
plot(t,q,'b')
plot(t,fq,'--b')
plot(t,x,'--r')
plot(t,x_r,'--k')
hold off

%% Function that calculates derivatives
function [dq]=vel(q,t)

dq=zeros(length(t),1);

for i = 1:length(t)-1
    dq(i+1) = (q(i+1)-q(i))/(t(i+1)-t(i));
end

end

%% Function that adds parabolic blends to the joint trajectories
function [traj]=PolyTraj(q,dq,t,blends)

traj=zeros(length(q),1);
window=floor(length(q)/blends);

up=1;
down=window;

for i=1:blends
    
    if i==blends
        pad=(length(q)-down);
        window=window+pad;
        down=down+pad;
    end
    
    theta_s=q(up);
    theta_f=q(down);
    
    theta_dot_s=dq(up);
    theta_dot_f=dq(down);
    
    c = MyPoly3(theta_s,theta_f,t(window),theta_dot_s,theta_dot_f);
    dummy= Traj(c,t(1:window));
    
    traj(down-window+1:down)=dummy;
    
    up=down+1;
    down=down+window;
    
end

end

%% Polynomial function coefficients
function alpha = MyPoly3(theta_s,theta_f,time,theta_dot_s,theta_dot_f)

alpha(1)=theta_s;
alpha(2)=theta_dot_s;
alpha(3)=3*(theta_f-theta_s)/time^2-2*(theta_dot_s)/time-1*(theta_dot_f)/time;
alpha(4)=-2*(theta_f-theta_s)/time^3+(theta_dot_f+theta_dot_s)/time^2;

end

%% Polynomial fitting
function traj = Traj(alpha,time)

traj=zeros(1,length(time));

for i=1:length(time)
    
    traj(i)=alpha(1)+alpha(2)*time(i)+alpha(3)*time(i)^2+alpha(4)*time(i)^3;
    
end

end

