function [labelsNew,Ytest,Testidx]=CGD(filename,Category,a)
% data = readtable(filename,'TextType','string');
data=filename;
% head(data)
% filename2 = "Test.csv";
% Test = readtable(filename2,'TextType','string');
Testidx=randsample(length(Category),round(a*length(Category)));
Test=data(Testidx,:);
data(Testidx,:)=[];
Ytest=Category(Testidx,:);
Category(Testidx,:)=[];
cvp = cvpartition(Category,'Holdout',0.2);
dataTrain = data(training(cvp),:);
dataValidation = data(test(cvp),:);
YTrain = Category(training(cvp),:);
YValidation = Category(test(cvp),:);
%%
textDataTrain = dataTrain;
textDataValidation = dataValidation;
% % YTrain = dataTrain.Category;
% % YValidation = dataValidation.Category;
%%
documentsTrain = preprocessText(textDataTrain);
documentsValidation = preprocessText(textDataValidation);

documentsTrain(1:5);

enc = wordEncoding(documentsTrain);

documentLengths = doclength(documentsTrain);
% figure
% histogram(documentLengths)
% title("Document Lengths")
% xlabel("Length")
% ylabel("Number of Documents")
%%
sequenceLength = 10;
XTrain = doc2sequence(enc,documentsTrain,'Length',sequenceLength);
XTrain(1:5);

XValidation = doc2sequence(enc,documentsValidation,'Length',sequenceLength);

inputSize = 1;
embeddingDimension = 50;
numHiddenUnits = 80;
%%
% U=unique(YTrain)
% for i=1:length(U)
%     ytrain(YTrain==U(i),1)=i;
% end
% ytrain=categorical(YTrain)
%%
numWords = enc.NumWords;
numClasses = numel(categorical(YTrain));
YTrain=categorical(YTrain);
YValidation=categorical(YValidation);
%%
layers = [ ...
    sequenceInputLayer(inputSize)
    wordEmbeddingLayer(embeddingDimension,numWords)
    lstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',10,...
    'MiniBatchSize',100, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{XValidation,YValidation}, ...%'Plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(XTrain,YTrain,layers,options);


reportsNew = Test;
documentsNew = preprocessText(reportsNew);

XNew = doc2sequence(enc,documentsNew,'Length',sequenceLength);


labelsNew = classify(net,XNew);

% acc=length(labelsNew==Ytest)/length(Ytest);

% % YTest=categorical(Test.Category);