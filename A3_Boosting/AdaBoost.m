%% Hyper-parameters

% Number of randomized Haar-features
nbrHaarFeatures = 300;
% Number of training images, will be evenly split between faces and
% non-faces. (Should be even.)
nbrTrainImages = 500;
% Number of weak classifiers
nbrWeakClassifiers = 300;

%% Load face and non-face data and plot a few examples
load faces;
load nonfaces;
faces = double(faces(:,:,randperm(size(faces,3))));
nonfaces = double(nonfaces(:,:,randperm(size(nonfaces,3))));

figure(1);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(faces(:,:,10*k));
    axis image;
    axis off;
end

figure(2);
colormap gray;
for k=1:25
    subplot(5,5,k), imagesc(nonfaces(:,:,10*k));
    axis image;
    axis off;
end

%% Generate Haar feature masks
haarFeatureMasks = GenerateHaarFeatureMasks(nbrHaarFeatures);

figure(3);
colormap gray;
for k = 1:25
    subplot(5,5,k),imagesc(haarFeatureMasks(:,:,k),[-1 2]);
    axis image;
    axis off;
end

%% Create image sets (do not modify!)

% Create a training data set with examples from both classes.
% Non-faces = class label y=-1, faces = class label y=1
trainImages = cat(3,faces(:,:,1:nbrTrainImages/2),nonfaces(:,:,1:nbrTrainImages/2));
xTrain = ExtractHaarFeatures(trainImages,haarFeatureMasks);
yTrain = [ones(1,nbrTrainImages/2), -ones(1,nbrTrainImages/2)];

% Create a test data set, using the rest of the faces and non-faces.
testImages  = cat(3,faces(:,:,(nbrTrainImages/2+1):end),...
                    nonfaces(:,:,(nbrTrainImages/2+1):end));
xTest = ExtractHaarFeatures(testImages,haarFeatureMasks);
yTest = [ones(1,size(faces,3)-nbrTrainImages/2), -ones(1,size(nonfaces,3)-nbrTrainImages/2)];

% Variable for the number of test-data.
nbrTestImages = length(yTest);

%% Implement the AdaBoost training here
%  Use your implementation of WeakClassifier and WeakClassifierError
weights = ones(size(xTrain,2),1) ./ size(xTrain,2);
classifier = zeros(nbrWeakClassifiers,4);
for i = 1:nbrWeakClassifiers
    Emin = 2;
    Pb = 1;
    Tb = 0;
    Fb = 0;
    alphab = 0;
for j = 1:size(xTrain,1)
for z = 1:size(xTrain,2)
    P = 1;
    C = WeakClassifier(xTrain(j,z)', P, xTrain(j,:)');
    E = WeakClassifierError(C, weights, yTrain');
    if E > 0.5
        P = -1;
        E = 1-E;
    end
    
    if E < Emin
        Pb = P;
        Tb = xTrain(j,z);
        Fb = j;
        alphab = (1/2) * log((1-E)/E);
        Emin = E;
    end
end
end
% save classifier 
classifier(i,:) = [Pb,Tb,Fb, alphab];
% update weights
res = WeakClassifier(Tb, Pb, xTrain(Fb,:));
Err = WeakClassifierError(res', weights, yTrain');
weights = weights' .* exp(-alphab .* res .* yTrain);
weights = weights' ./ sum(weights);
end


%% Evaluate your strong classifier here
%  Evaluate on both the training data and test data, but only the test
%  accuracy can be used as a performance metric since the training accuracy
%  is biased.


disp("train error:")
disp(Evaluate(classifier,xTrain,yTrain))

disp("test error:")
disp(Evaluate(classifier,xTest,yTest))




 
%% Plot the error of the strong classifier as a function of the number of weak classifiers.
%  Note: you can find this error without re-training with a different
%  number of weak classifiers.
accuracy = zeros( size(classifier,1),1);

for i = 1: size(classifier,1)
    accuracy(i) = Evaluate(classifier(1:i,:),xTest,yTest);
end

plot(accuracy)
hold on 
%%
accuracy = zeros( size(classifier,1),1);

for i = 1: size(classifier,1)
    accuracy(i) = Evaluate(classifier(1:i,:),xTrain,yTrain);
end
plot(accuracy)
hold off
%% Plot some of the misclassified faces and non-faces
%  Use the subplot command to make nice figures with multiple images.
suma = zeros(1,size(xTest,2));
classifier = classifier(1:84,:);
for i = 1:size(classifier)
    suma = suma + classifier(i,4) * WeakClassifier(classifier(i,2), classifier(i,1), xTest(classifier(i,3),:));
end
pred = sign(suma);
res = find((sign(suma)~= yTest) .* (yTest == 1));
res = res(1:25);
figure(4);
colormap gray;
sgtitle("misclassified face")
for k=1:20  
    subplot(4,5,k), imagesc(testImages(:,:,res(k)));
    title(['Tr: ',num2str(yTest(res(k))),', Pr = ', num2str(pred(res(k)))])
    axis image;
    axis off;
end
%%
res = find((sign(suma)~= yTest) .* (yTest == -1));
res = res(1:25);
figure(5);
colormap gray;
sgtitle("misclassified non-face")
for k=1:20
    subplot(4,5,k), imagesc(testImages(:,:,res(k)));
    title(['Tr: ',num2str(yTest(res(k))),', Pr = ', num2str(pred(res(k)))])
    axis image;
    axis off;
end


%% Plot your choosen Haar-features
%  Use the subplot command to make nice figures with multiple images.
res = unique(classifier(:,3));

figure(6);
colormap gray;
sgtitle("first 30 selected HAAR features")
for k = 1:30
    title(k-1)
    subplot(5,6,k),imagesc(haarFeatureMasks(:,:,classifier(k,3)),[-1 2]);
    axis image;
    axis off;
end