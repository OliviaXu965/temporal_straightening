% calculates binomial likelihood for each frame pair (stimCond) 

function [NLL] = giveNLL2(paramVec,stimCond,nCorrect,numFrames,numTrials,numDim)

% Decode paramVec
lapseRate   = paramVec(end);
testParams  = reshape(paramVec(1:end-1),numFrames,numDim); %coordinate values for each frame

LL = NaN(length(stimCond),1); d_prime   = NaN(length(stimCond),1);
pCorrect  = NaN(length(stimCond),1); pC_lapse  = NaN(length(stimCond),1); 

distance  = @(p1,p2) sqrt(sum((p1-p2).^2));
    
    for i = 1:length(stimCond)
        d_prime(i)   = distance(testParams(stimCond(i,1),:), testParams(stimCond(i,2),:));
        pCorrect(i)  = normcdf(d_prime(i)/sqrt(2)) * normcdf(d_prime(i)/2)...
            + normcdf(-d_prime(i)/sqrt(2))*normcdf(-d_prime(i)/2);
        pC_lapse(i)  = (1-2*lapseRate)*pCorrect(i) + lapseRate;
        LL(i) = sum(nCorrect(i).*log(pC_lapse(i)) + (numTrials-nCorrect(i)).*log(1-pC_lapse(i)));
    end
    NLL = -sum(LL);
end

% an alternative for looping through all conditions is doing it in matrix
% forms. eg. normcdf, instead of putting in one mean and one variance, put
% in a matrix of means a matrix of vars. 
