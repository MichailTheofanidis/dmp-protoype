%% Script to test the IK function of the robotics toolbox for the Sawyer Robot

%% Import the manipulator as a robotics.RigidBodyTree Object
sawyer = importrobot('robot_description/sawyer.urdf');
sawyer.DataFormat = 'column';
 
% Define end-effector body name
eeName = 'right_hand';
 
%Define the number of joints in the manipulator
numJoints = 8;

% Visualize the manipulator
show(sawyer);
xlim([-1.00 1.00])
ylim([-1.00 1.00]);
zlim([-1.02 0.98]);
view([128.88 10.45]);

% Define the origin
Origin = [0.4,-0.8,-0.08];

% Desired end effector orientation
eeOrientation = [0, pi, 0];

% Up location 
waypt0 = [Origin + [0 0 .2],eeOrientation];

% Down location
waypt1 = [Origin,eeOrientation];

% Interpolate each element for smooth motion to the origin of the image
for i = 1:6
    
    interp = linspace(waypt0(i),waypt1(i),100);
    wayPoints(:,i) = interp';
    
end

% Initialize size of q0, the robot joint configuration at t=0.
q0 = zeros(numJoints,1);

% Define a sampling rate for the simulation.
Ts = .01;

% Define a [1x6] vector of relative weights on the orientation and 
% position error for the inverse kinematics solver.
weights = ones(1,6);

% Transform the first waypoint to a Homogenous Transform Matrix for initialization
initTargetPose = eul2tform(wayPoints(1,4:6));
initTargetPose(1:3,end) = wayPoints(1,1:3)';

% Solve for q0 such that the manipulator begins at the first waypoint
ik = robotics.InverseKinematics('RigidBodyTree',sawyer);
[q1,solInfo] = ik(eeName,initTargetPose,weights,q0);

% Close currently open figures 
close all

% Create the trajectory
for i = 1:8
    
    interp = linspace(q0(i),q1(i),100);
    jointData(:,i) = interp';
    
end


% Plot the initial robot position
show(sawyer, jointData(1,:)');
xlim([-1.00 1.00])
ylim([-1.00 1.00]);
zlim([-1.02 0.98]);
view([128.88 10.45]);


% Iterate through the outputs at 10-sample intervals to visualize the results
for j = 1:10:length(jointData)
    
    % Display manipulator model
    show(sawyer,jointData(j,:)');
    
    % Update the figure
    drawnow
end