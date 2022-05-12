function [ cM ] = calcConfusionMatrix( LPred, LTrue )
% CALCCONFUSIONMATRIX returns the confusion matrix of the predicted labels

classes  = unique(LTrue);
NClasses = length(classes);

% Add your own code here
cM = zeros(NClasses);
for i = 1: length(classes)
    Lpredtrue = LPred(LTrue==classes(i));
    for k = 1: length(classes)
        cM(i,k) = sum(Lpredtrue==classes(k));
    end
end
end

