%% Dynamic Movement Primitives Class
%% by Michail Theofanidis

classdef Dmp
    
    properties
        
        a % Gain a of the transformation system
        b % Gain b of the transformation system
        as % Degredation of the canonical system
        ng=20 % Number of gaussians
        stb=1 % Stabilization term
        
    end
    
    methods
        
        % Constructor method for the transformation system
        function obj = Dmp(varargin)
            
            if nargin==0
                obj.a = 25;
                obj.b = obj.a/4;
                obj.as = obj.a/3;
            end
            
            if nargin==1
                if isnumeric(varargin{:})
                    obj.a = varargin{1};
                    obj.b = obj.a/4;
                    obj.as = obj.a/3;
                else
                    error('Value must be numeric')
                end
            end
            
            if nargin==2
                varargin{:}
                if isnumeric([varargin{:}])
                    obj.a = varargin{1};
                    obj.b = varargin{2};
                    obj.as = obj.a/3;
                else
                    error('Value must be numeric')
                end
            end
            
            if nargin==3
                varargin{:}
                if isnumeric([varargin{:}])
                    obj.a = varargin{1};
                    obj.b = varargin{2};
                    obj.as = obj.a/3;
                    obj.ng = varargin{3};
                else
                    error('Value must be numeric')
                end
            end
            
        end
        
        % Create the phase of the system
        function s = phase(obj,time)
            r=exp(-1*[obj.as]*linspace(0,1,length(time)));
            s=r';
        end
        
        % Gaussian distributions
        function psv=distributions(obj,s)
            
            % centers of gaussian are placed even in s time.
            incr=(s(1)-s(end))/(obj.ng-1);
            c=s(end):incr:s(1);
            lrc=fliplr(c);
            d=diff(c);
            c=c/d(1);
            
            % calculate every gaussian
            h=1;
            psv=zeros(length(s),obj.ng);
            
            for i=1:obj.ng
                for j=1:length(s)
                    psv(j,i)=psiF(h,c(i),s(j)/d(1));
                end
            end
        end
        
        % Immitation learning
        function [ftarget,w]=immitate(obj,x,dx,ddx,time,s,psv)
            
            % Define some variables
            g=x(end);
            x0=x(1);
            tau=time(end);
            
            % Initialize ftarget and variables for regression
            ftarget=zeros(length(time),1);
            sigma=zeros(length(time),1);
            
            % Compute ftarget
            for i=1:length(time)
                
                % Add the sabilization term
                if obj.stb==1
                    mod=obj.b*(g-x0)*s(i);
                else
                    mod=0;
                end
                
                ftarget(i)=(ddx(i)*tau^2-obj.a*(obj.b*(g-x(i))-tau*dx(i)-mod))/(g-x0);
                
                sigma(i)=s(i)*(g-x0);
                
            end
            
            % Regression
            for i=1:obj.ng
                
                %Locally Weighted Learning
                r=(transpose(sigma)*diag(psv(:,i))*ftarget)/(transpose(sigma)*diag(psv(:,i))*sigma);
                w(i)=r';
            end
            
            
        end
        
        % Generate trajectory
        function[x,dx,ddx]=generate(obj,w,x0,g,t,s,psV)
            
            ddx=zeros(length(s),1);
            dx=zeros(length(s),1);
            x=zeros(length(s),1);
            
            tau=t(end);
            
            ddx_r=0;
            dx_r=0;
            x_r=0;
            
            for i=1:length(t)
                
                psum_x=0;
                pdiv_x=0;
                
                if i==1
                    dt=t(i);
                else
                    dt=t(i)-t(i-1); 
                end
                    
                for j=1:obj.ng
                    
                    % Normalizing the center of gaussians
                    psum_x=psum_x+psV(i,j)*w(j);
                    pdiv_x=pdiv_x+psV(i,j);
                    
                end
                
                % Add the sabilization term
                if obj.stb==1
                    mod=obj.b*(g-x0)*s(i);
                else
                    mod=0;
                end
                
                % Generate the new trajectories according to the new control input
                f_rep(i)=(psum_x/pdiv_x)*s(i)*(g-x0);
                
                % Apply the formula tau*ydd= K(g-x)-Dv+(g-x0)f
                ddx_r=(f_rep(i)*(g-x0)+ddx(i)*tau^2+obj.a*(obj.b*(g-x(i))+tau*dx(i)+mod))/tau;
                dx_r= dx_r+(ddx_r*dt)/tau;
                x_r= x_r+(ddx_r*dt)/tau;
                
                ddx(i)=ddx_r;
                dx(i)=dx_r;
                x(i)=x_r;
                
            end
            
        end
        
        
    end
    
end

%% My Gaussian function
function r=psiF(h, c, s)
r=exp(-h*(s-c)^2);
end
