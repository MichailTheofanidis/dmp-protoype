%% Extract DMP parameters

clear all
clc

%% Load the data
load saveddata

%% Establish some of the variables

x=saveddata.x;
dx=saveddata.vx;
ddx=saveddata.ax;
x0=x(1);
gx=x(end);

dtx=diff(saveddata.times);
dtx=dtx(1);

y=saveddata.y;
dy=saveddata.vy;
ddy=saveddata.ay;
y0=y(1);
gy=y(end);

time=saveddata.times;
%% Normalize the time vector from zero to to one

time=time-time(1);

%% Parameters of the DMP
par.ng=50; % Number of Gaussians

h=1;
par.h=ones(1,par.ng)*(h); % width of the Gaussians


par.s=1; % Init of phase
par.as=1; % Decay of s phase var
par.tau=max(time); % Time scaling

par.K=1; % K gain
par.D=1; % D gain

%% Start training the DMP

len=length(time);

% ftarget calculation
stime=[];
sE_x=[];
sE_y=[];

for i=1:len
    
    t=time(i);
    s=exp((-1*par.as*t)/par.tau);
    stime=[stime s];
    
    % fdemonstration=ftarget
    ftarget_x(i)= (-1*par.K*(gx-x(i))+par.D*dx(i)+par.tau*ddx(i))/(gx-x0);
    ftarget_y(i)= (-1*par.K*(gy-y(i))+par.D*dy(i)+par.tau*ddy(i))/(gy-y0);
    
    sE_x=[sE_x; s*(gx-x0)];
    sE_y=[sE_y; s*(gy-y0)];
    
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
    psV_y=[];
    for j=1:len
        psV_x=[psV_x psiF(par.h,par.c,stime(j)/d(1),i)];
        psV_y=[psV_y psiF(par.h,par.c,stime(j)/d(1),i)];
    end
    %Locally Weighted Learning
    w_x(i)=(transpose(sE_x)*diag(psV_x)*transpose(ftarget_x))/(transpose(sE_x)*diag(psV_x)*sE_x);
    w_y(i)=(transpose(sE_y)*diag(psV_y)*transpose(ftarget_y))/(transpose(sE_y)*diag(psV_y)*sE_y);
end
%% Store the results

r=par;
r.len=len;
r.time=time;
r.stime=stime;
r.ftarget_x=ftarget_x;
r.ftarget_y=ftarget_y;
r.w_x=w_x;
r.w_y=w_y;
r.x0=x0;
r.y0=y0;
r.gx=gx;
r.gy=gy;
r.d1=d(1);
r.dt=dtx;
r.ctime=ctime;
r.penWidth1=2;
r.penWidth2=2;

%% Generate the estimated DMP

% Arrays to store the estimated trajectories
f_replay_x=[];
fr_x_zeros=[];

f_replay_y=[];
fr_y_zeros=[];

ydd_x_r=0;
yd_x_r=0;
y_x_r=r.x0;
dtx=r.dt;

ydd_y_r=0;
yd_y_r=0;
y_y_r=r.y0;
dty=dtx;

for j=1:length(r.time)
    psum_x=0;
    pdiv_x=0;
    psum_y=0;
    pdiv_y=0;
    for i=1:r.ng

        % Normalizing the center of gaussians
        psum_x=psum_x+psiF(r.h, r.c, r.stime(j)/r.d1,i)*r.w_x(i);
        pdiv_x=pdiv_x+psiF(r.h, r.c, r.stime(j)/r.d1,i);
        
        psum_y=psum_y+psiF(r.h, r.c, r.stime(j)/r.d1,i)*r.w_y(i);
        pdiv_y=pdiv_y+psiF(r.h, r.c, r.stime(j)/r.d1,i);
    end

    % Generate the new trajectories according to the new control input
    f_replay_x(j)=(psum_x/pdiv_x)*r.stime(j)*(r.gx-r.x0);
    f_replay_y(j)=(psum_y/pdiv_y)*r.stime(j)*(r.gy-r.y0);
    
    if(j>1)
        if sign(f_replay_x(j-1))~=sign(f_replay_x(j))
            fr_x_zeros=[fr_x_zeros j-1];
        end
        
        if sign(f_replay_y(j-1))~=sign(f_replay_y(j))
            fr_y_zeros=[fr_y_zeros j-1];
        end
    end
    
    % Apply the formula tau*ydd= K(g-x)-Dv+(g-x0)f
    ydd_x_r=(r.K*(r.gx-y_x_r)-(r.D*yd_x_r)+(r.gx-r.x0)*f_replay_x(j))/r.tau;
    yd_x_r= yd_x_r+ (ydd_x_r*dtx)/r.tau;
    y_x_r= y_x_r+ (yd_x_r*dtx)/r.tau;
    
    ydd_xr(j)=ydd_x_r;
    yd_xr(j)=yd_x_r;
    y_xr(j)=y_x_r;
    
    ydd_y_r=(r.K*(r.gy-y_y_r)-(r.D*yd_y_r)+(r.gy-r.y0)*f_replay_y(j))/r.tau;
    yd_y_r= yd_y_r + (ydd_y_r*dty)/r.tau;
    y_y_r= y_y_r + (yd_y_r*dty)/r.tau;
    
    ydd_yr(j)=ydd_y_r;
    yd_yr(j)=yd_y_r;
    y_yr(j)=y_y_r;
end

%% Store the estimated DMP
result=r;
result.ydd_yr=ydd_yr;
result.yd_yr=yd_yr;
result.y_yr=y_yr;
result.ydd_xr=ydd_xr;
result.yd_xr=yd_xr;
result.y_xr=y_xr;
result.fr_x_zeros=fr_x_zeros;
result.fr_y_zeros=fr_y_zeros;
result.f_replay_x=f_replay_x;
result.f_replay_y=f_replay_y;
result.options=[14];

%% Display the DMP and the initial trajectory

figure
plot(y,time, 'LineWidth',r.penWidth1);
xlabel('time (sec)');
ylabel('data');
hold on
plot(result.y_yr, time, 'r', 'LineWidth', r.penWidth2);
xlabel('time (sec)');
ylabel('xy replay from DMP');