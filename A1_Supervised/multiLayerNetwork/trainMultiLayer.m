function [Wout,Vout,ErrTrain,ErrTest] = trainMultiLayer(XTrain,DTrain,XTest,DTest,W0,V0,numIterations,learningRate)
% TRAINMULTILAYER Trains the multi-layer network (Learning)
%    Inputs:
%                X* - Training/test samples (matrix)
%                D* - Training/test desired output of net (matrix)
%                V0 - Initial weights of the output neurons (matrix)
%                W0 - Initial weights of the hidden neurons (matrix)
%                numIterations - Number of learning steps (scalar)
%                learningRate  - The learning rate (scalar)
%
%    Output:
%                Wout - Weights after training (matrix)
%                Vout - Weights after training (matrix)
%                ErrTrain - The training error for each iteration (vector)
%                ErrTest  - The test error for each iteration (vector)

% Initialize variables
ErrTrain = nan(numIterations+1, 1);
ErrTest  = nan(numIterations+1, 1);
NTrain   = size(XTrain, 1);
NTest    = size(XTest , 1);
NClasses = size(DTrain, 2);
Wout = W0;
Vout = V0;



% Calculate initial error
[YTrain, ~, HTrain] = runMultiLayer(XTrain, W0, V0);
YTest               = runMultiLayer(XTest , W0, V0);
ErrTrain(1) = sum(sum((tanh(YTrain) - DTrain).^2)) / (NTrain * NClasses);
ErrTest(1)  = sum(sum((tanh(YTest)  - DTest ).^2)) / (NTest  * NClasses);

% gpu array

% XTrain = gpuArray(XTrain);
% DTrain = gpuArray(DTrain);
% Vout = gpuArray(Vout);

for n = 1:numIterations
    % Add your own code here
    

    net1 = XTrain * Wout;
    z = [tanh(net1) ones(length(net1),1)];
    ader = 1-(tanh(net1).^2);
    net2 = z * Vout;
    Ypred = tanh(net2);
%     ader2 = 1-(Ypred.^2);
    error = -2*(DTrain - Ypred);
    
    %grad_v = transpose(z) * (error .* ader2); % Gradient for the output layer
    grad_v = transpose(z) * (error); % Gradient for the output layer
    
    tempV = Vout;
    tempV(size(Vout,1),:) = [];
    
%     grad_w = transpose(XTrain) * ((error .* ader2) * transpose(tempV) .* ader);
    grad_w = transpose(XTrain) * ((error ) * transpose(tempV) .* ader);
 
     
    
    % Take a learning step
    Vout = Vout - learningRate * grad_v/NTrain;
    Wout = Wout - learningRate * grad_w/NTrain;
    
    if rem(n,50) == 0
        disp(n*100/numIterations)
    end
    
    % Evaluate errors
    [YTrain, ~, HTrain] = runMultiLayer(XTrain, Wout, Vout);
    YTest               = runMultiLayer(XTest , Wout, Vout);
    ErrTrain(1+n) = sum(sum((tanh(YTrain) - DTrain).^2)) / (NTrain * NClasses);
    ErrTest(1+n)  = sum(sum((tanh(YTest)  - DTest ).^2)) / (NTest  * NClasses);
end

end
