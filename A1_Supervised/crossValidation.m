function [z,k,means ] = crossValidation( X, D, L ,folds)
% CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels
numBins = folds;                    % Number of bins you want to devide your data into
numSamplesPerLabelPerBin = Inf; % Number of samples per label per bin, set to inf for max number (total number is numLabels*numSamplesPerBin)
selectAtRandom = true;          % true = select samples at random, false = select the first features
k= 1:20;

[XBins, DBins, LBins] = selectTrainingSamples(X, D, L, numSamplesPerLabelPerBin, numBins, selectAtRandom);

CV = zeros(folds ,length(k));
for i = 1:length(XBins)

    diff = setdiff(1:length(XBins),i);
    Xtrain = XBins{diff(1)};
    Ltrain = LBins{diff(1)};
    for j = 2:length(diff)
        Xtrain = cat(1,Xtrain, XBins{diff(j)});
        Ltrain = cat(1,Ltrain, LBins{diff(j)});
    end
    
    for l = 1:length(k)
        LPredTest  = kNN(XBins{i} , k(l), Xtrain, Ltrain);
        cM = calcConfusionMatrix(LPredTest, LBins{i});
        CV(i,l) = calcAccuracy(cM);
    end   
end
means = mean(CV,1)
[M,I] = max(means);
M
z = k(I);
end
