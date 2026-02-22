function [cnn,weight]=ConvNet(D,C,test,testlabel)

D=[D,C];
test=[test,testlabel];
tbl = array2table(D);
labelName = tbl.Properties.VariableNames(end);
tbl = convertvars(tbl,labelName,'categorical');

tbl = splitvars(tbl);
classNames = categories(tbl{:,labelName});
numObservations = size(tbl,1);
numObservationsTrain = floor(0.80*numObservations);
numObservationsValidation = floor(0.2*numObservations);
%numObservationsTest = numObservations - numObservationsTrain - numObservationsValidation;
numObservationsTest=size(test,1);
idx = randperm(numObservations);
idxTrain = idx(1:numObservationsTrain);
idxValidation = idx(numObservationsTrain+1:numObservationsTrain+numObservationsValidation);
% idxTest = randperm(numObservationsTest);
tblTrain = tbl(idxTrain,:);
tblValidation = tbl(idxValidation,:);
tblTest = array2table(test);

numFeatures = size(tbl,2) - 1;
numClasses = numel(classNames);
 
layers = [
    featureInputLayer(numFeatures,'Normalization', 'zscore')
    fullyConnectedLayer(100)
    batchNormalizationLayer
    reluLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
miniBatchSize = 50;
weight=1;

options = trainingOptions('adam', ...
    'InitialLearnRate',0.002,...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',20,...
    'Shuffle','every-epoch');%, ...
%     'Plots','training-progress', ...
%     'Verbose',true);
% % 'ValidationData',tblValidation, ...
net = trainNetwork(tblTrain,labelName,layers,options);

cnn = classify(net,tblTest(:,1:end-1),'MiniBatchSize',miniBatchSize);

% U= unique(CNNYPred);
% for i=1:length(U)
% index=find (ismember(CNNYPred,U(i)));
% cnn(index,1)=i-1;
% end