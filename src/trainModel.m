function info = trainModel(cfg)
% TRAINMODEL  Train AlexNet-transfer model on data/splits/train.csv,
% holding out cfg.validationSplit for validation. Saves best checkpoint,
% training history JSON, and a loss/accuracy curve figure.
if nargin < 1, cfg = config(); end
rng(cfg.seed);

trainTbl = readtable('data/splits/train.csv','Delimiter',',');
imds = imageDatastore(trainTbl.filepath, 'Labels', categorical(trainTbl.label));
[imdsTrain, imdsVal] = splitEachLabel(imds, 1 - cfg.validationSplit, 'randomized');

inputSize = [227 227 3];   % AlexNet's native input size
augTrain = augmentedImageDatastore(inputSize, imdsTrain);
augVal   = augmentedImageDatastore(inputSize, imdsVal);

net = buildAlexNet(cfg);

options = trainingOptions('sgdm', ...
    'InitialLearnRate', cfg.initialLearnRate, ...
    'MiniBatchSize', cfg.miniBatchSize, ...
    'MaxEpochs', cfg.maxEpochs, ...
    'ValidationData', augVal, ...
    'ValidationFrequency', max(1, floor(numel(imdsTrain.Files)/cfg.miniBatchSize)), ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'OutputNetwork', 'best-validation-loss', ...
    'Plots', 'none');

[trainedNet, trainInfo] = trainNetwork(augTrain, net, options);

if ~exist('outputs/models','dir'), mkdir('outputs/models'); end
save(cfg.modelOutPath, 'trainedNet', 'cfg');

if ~exist('outputs/metrics','dir'), mkdir('outputs/metrics'); end
hist = struct('TrainingLoss', trainInfo.TrainingLoss, ...
               'TrainingAccuracy', trainInfo.TrainingAccuracy, ...
               'ValidationLoss', trainInfo.ValidationLoss, ...
               'ValidationAccuracy', trainInfo.ValidationAccuracy);
fid = fopen('outputs/metrics/training_history.json','w');
fprintf(fid, '%s', jsonencode(hist));
fclose(fid);

f = figure('Visible','off');
subplot(1,2,1); plot(trainInfo.TrainingAccuracy); hold on;
plot(trainInfo.ValidationAccuracy,'o'); title('Accuracy'); legend('train','val');
subplot(1,2,2); plot(trainInfo.TrainingLoss); hold on;
plot(trainInfo.ValidationLoss,'o'); title('Loss'); legend('train','val');
if ~exist('outputs/figures','dir'), mkdir('outputs/figures'); end
saveas(f, 'outputs/figures/training_curves.png'); close(f);

info = trainInfo;
end
