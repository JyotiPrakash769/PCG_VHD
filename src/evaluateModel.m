function results = evaluateModel(cfg)
% EVALUATEMODEL  Load best checkpoint, run on data/splits/test.csv,
% compute confusion matrix + per-class precision/recall/F1 + macro/
% weighted averages. Saves figure + CSV report.
if nargin < 1, cfg = config(); end
S = load(cfg.modelOutPath, 'trainedNet');
net = S.trainedNet;

testTbl = readtable('data/splits/test.csv','Delimiter',',');
imds = imageDatastore(testTbl.filepath, 'Labels', categorical(testTbl.label));
augTest = augmentedImageDatastore([227 227 3], imds);

predLabels = classify(net, augTest);
trueLabels = imds.Labels;

acc = mean(predLabels == trueLabels);
C = confusionmat(trueLabels, predLabels);
classes = categories(trueLabels);

precision = zeros(numel(classes),1); recall = precision; f1 = precision;
for i = 1:numel(classes)
    tp = C(i,i);
    fp = sum(C(:,i)) - tp;
    fn = sum(C(i,:)) - tp;
    precision(i) = tp / max(tp+fp, eps);
    recall(i)    = tp / max(tp+fn, eps);
    f1(i)        = 2*precision(i)*recall(i) / max(precision(i)+recall(i), eps);
end
macroP = mean(precision); macroR = mean(recall); macroF1 = mean(f1);
support = sum(C,2);
weightedP = sum(precision.*support)/sum(support);
weightedR = sum(recall.*support)/sum(support);
weightedF1 = sum(f1.*support)/sum(support);

report = table(classes, precision, recall, f1, support, ...
    'VariableNames', {'class','precision','recall','f1','support'});
if ~exist('outputs/metrics','dir'), mkdir('outputs/metrics'); end
writetable(report, 'outputs/metrics/classification_report.csv');

f = figure('Visible','off');
confusionchart(trueLabels, predLabels);
title(sprintf('Test accuracy: %.2f%%', acc*100));
if ~exist('outputs/figures','dir'), mkdir('outputs/figures'); end
saveas(f, 'outputs/figures/confusion_matrix.png'); close(f);

fprintf('Overall accuracy: %.4f\n', acc);
fprintf('Macro  P/R/F1: %.4f / %.4f / %.4f\n', macroP, macroR, macroF1);
fprintf('Weighted P/R/F1: %.4f / %.4f / %.4f\n', weightedP, weightedR, weightedF1);

results = struct('accuracy',acc,'confusion',C,'report',report, ...
    'macroP',macroP,'macroR',macroR,'macroF1',macroF1, ...
    'weightedP',weightedP,'weightedR',weightedR,'weightedF1',weightedF1);
save('outputs/metrics/eval_results.mat','results');
end
