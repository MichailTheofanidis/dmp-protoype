%% Learn the DMP parameters of the Inverse Model Output

clear all
clc

%% Load the data of the inverse Models

id=[3];
convert=1000;

for i=1:length(id)
    name='DMP_';
    name_id=strcat(name,num2str(id(i)));
    full_name=strcat(name_id,'.mat');
    Data(i)=load(full_name);
    name=[];
    name_id=[];
    full_name=[];
end

%% Calculate the DMP for every cartesian dimension
for i=1:length(id)
    temp=[]; 
    for j=1:3
        target=Data(i).DMP.target(j);
        %target=Data(i).DMP.p(end,j);
        [result,par]=DynamicMotionPrimitive(Data(i).DMP.p(:,j),Data(i).DMP.time,1000,target);
        temp(:,j)=result.y_xr;     
    end
    Data(i).DMP.res=temp;
end

%% Visualize the Stored Data
colors=['r','b','k','c'];
for i=1:length(id)
    figure(1)
    hold on
    plot3(Data(i).DMP.p(:,1), Data(i).DMP.p(:,2), Data(i).DMP.p(:,3),colors(i))
    plot3(Data(i).DMP.res(:,1), Data(i).DMP.res(:,2), Data(i).DMP.res(:,3),colors(i))
    plot3(Data(i).DMP.target(1),Data(i).DMP.target(2),Data(i).DMP.target(3),strcat('*',colors(i)))
    grid on
    hold off  
end

%% Function that calculates the DMP
function [result,par]=DynamicMotionPrimitive(x,time,convert,target)

% Initialize the state parameters of the DMP
x=x;
time=time./convert;
time=time-time(1);

dtx=diff(time);
dtx=dtx(1);
dx=[0;diff(x)]./dtx;
ddx=[0;diff(dx)]./dtx;
x0=x(1);
gx=x(end);
target=target;

% Parameters of the DMP
par.ng=100; % Number of Gaussians
h=1;
par.h=ones(1,par.ng)*(h); % width of the Gaussians
par.s=1; % Init of phase
par.as=1; % Decay of s phase var
par.tau=max(time); % Time scaling
par.K=10000; % K gain
par.D=100; % D gain

% Start training the DMP
len=length(time);

% ftarget calculation
stime=[];
sE_x=[];

for i=1:len
    
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
    for j=1:len
        psV_x=[psV_x psiF(par.h,par.c,stime(j)/d(1),i)];
    end
    %Locally Weighted Learning
    w_x(i)=(transpose(sE_x)*diag(psV_x)*transpose(ftarget_x))/(transpose(sE_x)*diag(psV_x)*sE_x);
end

size(sE_x)
size(psV_x)
size(ftarget_x)

%% Store the results

r=par;
r.len=len;
r.time=time;
r.stime=stime;
r.ftarget_x=ftarget_x;
r.w_x=w_x;
r.x0=x0;
r.gx=gx;
r.d1=d(1);
r.dt=dtx;
r.ctime=ctime;

%% Generate the estimated DMP

% Arrays to store the estimated trajectories
f_replay_x=[];
fr_x_zeros=[];

ydd_x_r=0;
yd_x_r=0;
y_x_r=r.x0;
dtx=r.dt;

for j=1:length(r.time)
    psum_x=0;
    pdiv_x=0;
    for i=1:r.ng

        % Normalizing the center of gaussians
        psum_x=psum_x+psiF(r.h, r.c, r.stime(j)/r.d1,i)*r.w_x(i);
        pdiv_x=pdiv_x+psiF(r.h, r.c, r.stime(j)/r.d1,i);
        
    end

    % Generate the new trajectories according to the new control input
    f_replay_x(j)=(psum_x/pdiv_x)*r.stime(j)*(target-r.x0);
    
    if(j>1)
        if sign(f_replay_x(j-1))~=sign(f_replay_x(j))
            fr_x_zeros=[fr_x_zeros j-1];
        end
        
    end
    
    % Apply the formula tau*ydd= K(g-x)-Dv+(g-x0)f
    ydd_x_r=(r.K*(target-y_x_r)-(r.D*yd_x_r)+(target-r.x0)*f_replay_x(j))/r.tau;
    yd_x_r= yd_x_r+ (ydd_x_r*dtx)/r.tau;
    y_x_r= y_x_r+ (yd_x_r*dtx)/r.tau;
    
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