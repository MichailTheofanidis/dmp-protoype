%% Script that animates the robot from the DMP demonstrations
%% by Michail Theofanidis

clear all
clc

%% Import DMP from demonstration

% id of the DMPset
id=15;
name='../Demos/demo';
name=strcat(name,num2str(id));
name=strcat(name,'.txt');
Demo=importdata(name);

% Ploting arguments
animation=0;
speed=5;
positions=1;
sigma=0;
gaussians=1;
data=1;

%% Import the manipulator as a robotics.RigidBodyTree Object
sawyer = importrobot('robot_description/sawyer.urdf');
sawyer.DataFormat = 'column';

%% Create the joint and time vector
q=Demo.data(:,2:8);
t=Demo.data(:,1);
l=size(q);

% Normalize time vector
t=linspace(0,t(end),length(t));

%% Filtering parameters
r=50;
blends=10;

%% Compute velocity and acceleration

dq=zeros(l(1),l(2));
ddq=zeros(l(1),l(2));

for i=1:l(2)
    q(:,i)=smooth(q(:,i));
    [dq(:,i)]=vel(q(:,i),t);
    [ddq(:,i)]=vel(dq(:,i),t);
end

%% Perform filtering
fq=zeros(l(1),l(2));
fdq=zeros(l(1),l(2));
fddq=zeros(l(1),l(2));

for i=1:l(2)
    
    fq(:,i)=PolyTraj(q(:,i),dq(:,i),t,blends);
    fdq(:,i)=vel(fq(:,i),t);
    fddq(:,i)=vel(fdq(:,i),t);

end

%% Create the cartesian vector
p=FK(q);

%% Initialize the dmps
for i=1:l(2)
    dmp(i)=Dmp();
end

%% Calculate the phase of the dmp
s=dmp(1).phase(t);

%% Calculate the spread out gaussians
psV=dmp(1).distributions(s);

%% Perform immitation learning on all degrees of freedom
ftarget=zeros(l(1),l(2));
w=zeros(dmp(1).ng,l(2));

for i=1:l(2)
    [ftarget(:,i),w(:,i)]=dmp(i).immitate(fq(:,i),fdq(:,i),fddq(:,i),t,s,psV);
end

%% Perform forward pass of the DMP
x=zeros(l(1),l(2));
dx=zeros(l(1),l(2));
ddx=zeros(l(1),l(2));

for i=1:l(2)
    [x(:,i),dx(:,i),ddx(:,i)]=dmp(i).generate(w(:,i),fq(1,i),fq(end,i),t,s,psV);
end
    
%% Plotting functions
plot_counter=1;

% Sawyer robot animation
if (animation==1)
    
    figure(plot_counter)
    for j = 1:speed:length(q)
        
        jnt=[q(j,1:end)];
        
        [TeR,TrR,TR]=getSawyerFK_R(jnt);
        
        hold on
        cla
        plotSawyer(TR)
        grid on
        drawnow
        
    end
    plot_counter=plot_counter+1;
end

% Cartesian positions
if (positions==1)
    
    figure(plot_counter)
    plot3(p(:,1),p(:,2),p(:,3))
    plot3(x(:,1),x(:,2),x(:,3))
    grid on
    
    plot_counter=plot_counter+1;
end

% Plot sigma with respect to time
if (sigma==1)
    
    figure(plot_counter)
    plot(t,s)
    
    plot_counter=plot_counter+1;
end

% Initial placement of the gaussians
if (gaussians==1)
    
    figure(plot_counter)
    for i=1:l(2)
        cg=jet(dmp(1).ng); %colors of the gaussians
        for j=1:dmp(1).ng
            subplot(l(2),1,i)
            hold on
            plot(s,psV(:,j),'defaultAxesColorOrder',cg(i,:))
            hold off
        end
    end
    
    plot_counter=plot_counter+1;
    
    figure(plot_counter)
    for i=1:l(2)
        for j=1:dmp(1).ng
            subplot(l(2),1,i)
            hold on
            plot(s,psV(:,j)*w(j,i),'defaultAxesColorOrder',cg(i,:))
            hold off
        end
    end
    
    plot_counter=plot_counter+1;
end

% Initial position, velocity and acceleration
if (data==1)
    
    figure(plot_counter)
    for j=1:l(2)
        subplot(l(2),1,j)
        hold on
        plot(t,q(:,j),'b')
        plot(t,fq(:,j),'r--')
        plot(t,x(:,j),'k')
        hold off
    end
    plot_counter=plot_counter+1;
    
    figure(plot_counter)
    for j=1:l(2)
        subplot(l(2),1,j)
        hold on
        plot(t,dq(:,j),'b')
        plot(t,fdq(:,j),'r')
        plot(t,dx(:,j),'k')
        hold off
    end
    plot_counter=plot_counter+1;

    figure(plot_counter)
    for j=1:l(2)
        subplot(l(2),1,j)
        hold on
        plot(t,ddq(:,j),'b')
        plot(t,fddq(:,j),'r')
        plot(t,ddx(:,j),'k')
        hold off
    end
    plot_counter=plot_counter+1;
    
end

%% FUNCTIONS

% function for the forward kinematics
function [p]=FK(q)

% Initialize the reward function
p=zeros(length(q),3);
for i=1:length(q)
    [TeR,~,~]=getSawyerFK_R(q(i,:));
    [x,y,z]=MyTransl(TeR);
    p(i,:)=[x,y,z];
end

end

%% Bandpass filter
function [flt]=MyFFT(q,r)

temp_fft=fft(q');
rectangle=zeros(1,length(temp_fft));
rectangle(1:r+1)=1;
rectangle(end-r+1:end)=1;
flt=ifft(temp_fft.*rectangle);

end

%% Function that calculates derivatives
function [dq]=vel(q,t)

dq=zeros(length(t),1);

for i = 1:length(t)-1
    dq(i+1) = (q(i+1)-q(i))/(t(i+1)-t(i));
end

end

%% Function that calculates the integral
function [q]=int(dq,t)

q=zeros(length(t),1);

q(1)=dq(1);

for i = 2:length(t)
    
    q(i)=dq(i)*(t(i)-t(i-1))+q(i-1);
    
end

end

%% Function that adds parabolic blends to the joint trajectories
%% q = joint position trajectory
%% dq = joint velocity trajectory
%% blends = number of blends
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
