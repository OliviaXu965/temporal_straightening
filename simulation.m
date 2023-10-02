% This script simulates the behavioral responses of the AxB task in the 
% temporal straightening paper (2019)
close all; clear all; clc;

%% first, generate a perceptual trajectory of N frames
numFrames = 11;
v_0       = [1; 1]; % first vector at t0
vectors   = zeros(2, numFrames); 
unit      = @(vec) vec/sqrt(sum(vec.^2));

% define a length range for all vectors and assign a random length to each
% vector
min_length = 1; max_length = 5;
lengths    = (max_length - min_length) * rand(1,numFrames) + min_length;

% all local curvatures = 30°, randomly assign the direction of the
% curvature (clockwise or counterclockwise)
theta        = deg2rad(70);
cw_ccw_order = randi([0, 1], 1, numFrames);

% calculate the rotation matrices for a 30° cw or ccw rotation
cw_rotation_mat  = [cos(theta), -sin(theta); sin(theta), cos(theta)];
ccw_rotation_mat = [cos(theta), sin(theta); -sin(theta), cos(theta)];

% generate the vectors
for i = 1:numFrames
    % calculate the next vector by rotating the previous vector and moving
    % it to the head of the previous vector
    if i == 1
        vectors(:,i) = lengths(i) * unit(v_0);
    else
        if cw_ccw_order(i) == 1
            vectors(:,i) = lengths(i) * unit(cw_rotation_mat * vectors(:,i-1))...
                + vectors(:,i-1);
        else
            vectors(:,i) = lengths(i) * unit(ccw_rotation_mat * vectors(:,i-1))...
                + vectors(:,i-1);
        end
    end
end

% visualize the perceptual trajectory
figure(1);
for i = 2:numFrames  
    plot([vectors(1,i-1),vectors(1,i)],[vectors(2,i-1),vectors(2,i)],'-o','LineWidth', 1.5)
    hold on;
end
axis equal; grid on; hold on;
xlim([round(min(min(vectors)))-3, round(max(max(vectors)))+3]); 
ylim([round(min(min(vectors)))-3, round(max(max(vectors)))+3]);
title('Simulated perceptual trajectory');

%% generate isotropic gaussians centered on each frame in the perceptual space
% define parameters
mus   = vectors;
sigma = 1;           

[x, y] = meshgrid((round(min(min(mus)))-5)*sigma:0.1:(round(max(max(mus)))+5)*sigma,...
    (round(min(min(mus)))-5)*sigma:0.1:(round(max(max(mus)))+5)*sigma);

% calculate gaussian values at each point on the grid
gaussians = cell(numFrames,1);
for i = 1:numFrames
    gaussians{i} = (1/(2*pi*sigma^2)) * exp(-((x-mus(1,i)).^2 + (y-mus(2,i)).^2)/(2 * sigma^2));
end

% combine the gaussians for visualization
combined_gaussians = zeros(size(gaussians{1}));
for i = 1:numFrames
    combined_gaussians = combined_gaussians + gaussians{i};
end

% visualize the gaussians
figure(2);
contourf(x, y, combined_gaussians, 20, 'LineStyle', 'none');
colorbar; axis equal; 

%% simulate the AXB responses
% get all pairwise combinations (A,B), X will be identical to A or B
all_pairs = [repelem(1:numFrames,numFrames)', repmat(1:numFrames,1,numFrames)'];

% simulate responses 
sigma_mat = [sigma^2 0; 0 sigma^2];
numTrials = 100; 
distance  = @(p1,p2) sqrt(sum((p1-p2).^2));
resp_mat  = NaN(numTrials,length(all_pairs)); 

for i = 1:length(all_pairs)
    for j = 1:numTrials
        simA = mvnrnd(mus(:,all_pairs(i,1)),sigma_mat,1);
        simB = mvnrnd(mus(:,all_pairs(i,2)),sigma_mat,1);
        if rem(j,2) == 0
            simX = mvnrnd(mus(:,all_pairs(i,1)),sigma_mat,1); %draw from A
            dist_AX = distance(simA,simX);
            dist_BX = distance(simB,simX);
                if dist_AX < dist_BX 
                    resp_mat(j,i) = 1; %correct response
                else
                    resp_mat(j,i) = 0; %incorrect response
                end
        else
            simX = mvnrnd(mus(:,all_pairs(i,2)),sigma_mat,1); %draw from B
            dist_AX = distance(simA,simX);
            dist_BX = distance(simB,simX);
                if dist_BX < dist_AX  
                    resp_mat(j,i) = 1; %correct response
                else
                    resp_mat(j,i) = 0; %incorrect response
                end
        end 
    end
end
        
% calculate proportion correct
Pc = NaN(1,length(all_pairs));
for i = 1:length(all_pairs)
    Pc(i) = sum(resp_mat(:,i))/numTrials;
end
Pc_reshaped = reshape(Pc,numFrames,numFrames);

%% visualize the simulated data
figure(3);
imagesc(Pc_reshaped);
colormap('gray'); 
c = colorbar; 
title(c,'Proportion correct'); %how to set position?
c.Ticks = [0.5,1];
xticks([1 numFrames]); yticks([1 numFrames]);
xlim([1 numFrames]); ylim([1 numFrames]);
xlabel('Frame number'); ylabel('Frame number');
title('Simulated discriminability, 10,000 trials');
set(gca, 'FontName', 'Arial');
set(gca, 'FontSize', 12);
axis equal; 

