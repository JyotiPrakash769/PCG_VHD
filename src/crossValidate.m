function cvResults = crossValidate(cfg)
% CROSSVALIDATE  Separate protocol from the 150/50 split: cfg.kFolds-fold
% stratified CV over ALL scalograms (not just the fixed train/test split),
% matching the paper's reported "five-fold CV" evaluation. Trains a fresh
% AlexNet-transfer model per fold. Computationally heavy — intended to be
% run as a full experiment, not part of `main all` smoke test.
if nargin < 1, cfg = config(); end
rng(cfg.seed);

allFiles = {}; allLabels = {};
for c = 1:numel(cfg.classes)
    cls = cfg.classes{c};
    d = dir(fullfile('data/scalograms', cls, '*.png'));
    for i = 1:numel(d)
        allFiles{end+1} = fullfile(d(i).folder, d(i).name); %#ok<AGROW>
        allLabels{end+1} = cls; %#ok<AGROW>
    end
end
labels = categorical(allLabels);
cvp = cvpartition(labels, 'KFold', cfg.kFolds);

foldAcc = zeros(cfg.kFolds,1);
for k = 1:cfg.kFolds
    trIdx = training(cvp, k); teIdx = test(cvp, k);
    imdsTr = imageDatastore(allFiles(trIdx), 'Labels', labels(trIdx));
    imdsTe = imageDatastore(allFiles(teIdx), 'Labels', labels(teIdx));
    augTr = augmentedImageDatastore([227 227 3], imdsTr);
    augTe = augmentedImageDatastore([227 227 3], imdsTe);

    net = buildAlexNet(cfg);
    opts = trainingOptions('sgdm', 'InitialLearnRate', cfg.initialLearnRate, ...
        'MiniBatchSize', cfg.miniBatchSize, 'MaxEpochs', cfg.maxEpochs, ...
        'Shuffle','every-epoch', 'Verbose', false, 'Plots','none');
    trainedNet = trainNetwork(augTr, net, opts);

    pred = classify(trainedNet, augTe);
    foldAcc(k) = mean(pred == imdsTe.Labels);
    fprintf('Fold %d/%d accuracy: %.4f\n', k, cfg.kFolds, foldAcc(k));
end

cvResults = struct('foldAcc', foldAcc, 'meanAcc', mean(foldAcc), ...
                    'maxAcc', max(foldAcc), 'stdAcc', std(foldAcc));
if ~exist('outputs/metrics','dir'), mkdir('outputs/metrics'); end
save('outputs/metrics/cv_results.mat','cvResults');
fprintf('Mean CV accuracy: %.4f | Highest fold: %.4f\n', cvResults.meanAcc, cvResults.maxAcc);
fprintf('(Compare against paper-reported: mean overall 98.32%%, highest fold 99.6%% — do not force a match.)\n');
end
