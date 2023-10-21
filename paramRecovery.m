%% Parameter recovery analysis
clear all; close all; clc;

%% Load simulated data files and expt. params
C            = load('perceptual_traj_sim_data.mat');
simResp      = C.perceptual_traj_sim_data{2}.resp_mat; 

% Ground truth parameters
realParams   = C.perceptual_traj_sim_data{1}.vectors'; %cartesian coordinate values of frames
laspe_rate   = 0.01;

% Expt. parameters used to simulate the data: 
numFrames    = C.perceptual_traj_sim_data{1}.numFrames;
avgLocCurv   = C.perceptual_traj_sim_data{1}.avgLocCurv;
numTrials    = C.perceptual_traj_sim_data{1}.numTrials;
numDim       = C.perceptual_traj_sim_data{1}.numDim;
stimCond     = C.perceptual_traj_sim_data{1}.all_pairs;
locCurvs     = C.perceptual_traj_sim_data{1}.thetas; %local curvatures in deg
nCorrect     = sum(simResp,1); %nCorrect for each condition

modelParams  = [realParams(:);laspe_rate];

%% Fit model to data
options     = optimset('Display', 'iter', 'Maxiter', 10^5, 'MaxFuneval', 10^5);
objFun      = @(paramVec) giveNLL2(paramVec,stimCond,nCorrect,numFrames,numTrials,numDim);

%startVec   = [(20-(-20))*rand(numFrames*numDim,1)+(-20);laspe_rate]; % initial values (random values within a range)
startVec    = [reshape(realParams,33,1); laspe_rate]; % sanity check: use real params, see if the fit is better
lb          = -50.*ones(numFrames*numDim+1,1); 
ub          = 50.*ones(numFrames*numDim+1,1); 

paramEst    = fmincon(objFun, startVec, [], [], [], [], lb, ub, [], options);

%% visualize the recovered vs. real trajectories 2d
% paramEst_reshaped = reshape(paramEst(1:end-1),numFrames,numDim);
% figure(1)
% plot(paramEst_reshaped(:, 1), paramEst_reshaped(:, 2), 'c-o','LineWidth', 1.5)
% hold on;
% plot(realParams(:,1),realParams(:,2),'b-o','LineWidth', 1.5)
% axis equal; grid on; 
% xlim([round(min(min(min(paramEst_reshaped),min(realParams))))-3 ...
%     round(max(max(max(paramEst_reshaped),max(realParams))))+3]); 
% ylim([round(min(min(min(paramEst_reshaped),min(realParams))))-3 ...
%     round(max(max(max(paramEst_reshaped),max(realParams))))+3]);
% legend('Recovered', 'Real', 'Location', 'Northeast');
% title('Recovered vs real perceptual trajectories');

%% visualize the recovered vs. real trajectories >=3d
paramEst_reshaped = reshape(paramEst(1:end-1),numFrames,numDim);
set(figure(1), 'OuterPosition', [1250 300 500 500])
plot3(paramEst_reshaped(:,1), paramEst_reshaped(:,2), paramEst_reshaped(:,3),'c-o','LineWidth', 1.5)
hold on;
plot3(realParams(:,1),realParams(:,2),realParams(:,3),'b-o','LineWidth', 1.5)
axis equal; grid on; 
xlim([round(min(min(min(paramEst_reshaped),min(realParams))))-3 ...
    round(max(max(max(paramEst_reshaped),max(realParams))))+3]); 
ylim([round(min(min(min(paramEst_reshaped),min(realParams))))-3 ...
    round(max(max(max(paramEst_reshaped),max(realParams))))+3]);
zlim([round(min(min(min(paramEst_reshaped),min(realParams))))-3 ...
    round(max(max(max(paramEst_reshaped),max(realParams))))+3]);
legend('Recovered', 'Real', 'Location', 'Northeast');
title('Recovered vs real perceptual trajectories');

%% simulate the AXB responses with recovered params
mus       = paramEst_reshaped';
sigma     = 1;
sigma_mat = eye(numDim) * sigma;
distance  = @(p1,p2) sqrt(sum((p1-p2).^2));
resp_mat  = NaN(numTrials,length(stimCond)); 

for i = 1:length(stimCond)
    for j = 1:numTrials
        simA = mvnrnd(mus(:,stimCond(i,1)),sigma_mat,1);
        simB = mvnrnd(mus(:,stimCond(i,2)),sigma_mat,1);
        if rem(j,2) == 0
            simX = mvnrnd(mus(:,stimCond(i,1)),sigma_mat,1); %draw from A
            dist_AX = distance(simA,simX);
            dist_BX = distance(simB,simX);
                if dist_AX < dist_BX 
                    resp_mat(j,i) = 1; %correct response
                else
                    resp_mat(j,i) = 0; %incorrect response
                end
        else
            simX = mvnrnd(mus(:,stimCond(i,2)),sigma_mat,1); %draw from B
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
Pc = NaN(1,length(stimCond));
for i = 1:length(stimCond)
    Pc(i) = sum(resp_mat(:,i))/numTrials;
end
Pc_reshaped = reshape(Pc,numFrames,numFrames);

%% plot simulated data from recovered parameters
set(figure(2), 'OuterPosition', [650 300 500 500])
imagesc(Pc_reshaped);
colormap('gray'); 
c = colorbar; 
c.Ticks = [0.5,1];
xticks([1 numFrames]); yticks([1 numFrames]);
xlim([1 numFrames]); ylim([1 numFrames]);
xlabel('Frame number'); ylabel('Frame number');
titleText = sprintf('Simulated discriminability, %d trials, from recovered parameters', numTrials);
title(titleText);
set(gca, 'FontName', 'Arial');
set(gca, 'FontSize', 12);
axis equal; 

%% plot simulated data from real parameters 
% calculate proportion correct
Pc_real = NaN(1,length(stimCond));
for i = 1:length(stimCond)
    Pc_real(i) = sum(simResp(:,i))/numTrials;
end
Pc_reshaped_real = reshape(Pc_real,numFrames,numFrames);

set(figure(3), 'OuterPosition', [100 300 500 500])
imagesc(Pc_reshaped_real);
colormap('gray'); 
c = colorbar; 
c.Ticks = [0.5,1];
xticks([1 numFrames]); yticks([1 numFrames]);
xlim([1 numFrames]); ylim([1 numFrames]);
xlabel('Frame number'); ylabel('Frame number');
titleText = sprintf('Simulated discriminability, %d trials, from real parameters', numTrials);
title(titleText);
set(gca, 'FontName', 'Arial');
set(gca, 'FontSize', 12);
axis equal; 
