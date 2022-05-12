function [ LPred ] = kNN(X, k, XTrain, LTrain)
% KNN Your implementation of the kNN algorithm
%    Inputs:
%              X      - Samples to be classified (matrix)
%              k      - Number of neighbors (scalar)
%              XTrain - Training samples (matrix)
%              LTrain - Correct labels of each sample (vector)
%
%    Output:
%              LPred  - Predicted labels for each sample (vector)

classes = unique(LTrain);
NClasses = length(classes);

% Add your own code here
LPred  = zeros(size(X,1),1);

for i=1:height(X)
   [val,I]=sort(sqrt(sum((XTrain - X(i,:)).^2,2)));
   pred = LTrain(I(1:k));
   
   y = zeros(length(classes),1);
   for j=1:length(classes)
       y(j) = sum(pred == classes(j));
   end
   [M,J] = max(y);
   if sum(y == M) == 1
       LPred(i) = classes(J);
   else
       % sort according to distance
       selclasses = classes(y==M);
       lselected = pred;
       valselected = val(1:k);
       sumdistances = zeros(length(selclasses),1);
       for k = 1:length(selclasses)
           sumdistances(k) = sum(valselected(lselected==selclasses(k)));
       end
       [M,J] = min(sumdistances);
       LPred(i) = classes(J);
   end
   
end

