function [trainTbl, testTbl] = splitData(cfg)
% SPLITDATA  Deterministic stratified split: cfg.trainPerClass /
% cfg.testPerClass per class, from data/scalograms. Writes CSVs. No leakage
% (disjoint file lists per class, checked below).
if nargin < 1, cfg = config(); end
rng(cfg.seed);
trainRows = {}; testRows = {};
for c = 1:numel(cfg.classes)
    cls = cfg.classes{c};
    d = dir(fullfile('data/scalograms', cls, '*.png'));
    n = numel(d);
    need = cfg.trainPerClass + cfg.testPerClass;
    if n < need
        error('Class %s has %d images, need %d (train+test). Run generateScalograms first.', cls, n, need);
    end
    perm = randperm(n);
    trainIdx = perm(1:cfg.trainPerClass);
    testIdx  = perm(cfg.trainPerClass+1 : cfg.trainPerClass+cfg.testPerClass);
    for i = trainIdx
        trainRows(end+1,:) = {fullfile(d(i).folder, d(i).name), cls}; %#ok<AGROW>
    end
    for i = testIdx
        testRows(end+1,:) = {fullfile(d(i).folder, d(i).name), cls}; %#ok<AGROW>
    end
end
trainTbl = cell2table(trainRows, 'VariableNames', {'filepath','label'});
testTbl  = cell2table(testRows,  'VariableNames', {'filepath','label'});

% Leakage check: no filepath in both sets
assert(isempty(intersect(trainTbl.filepath, testTbl.filepath)), 'Train/test leakage detected!');

if ~exist('data/splits','dir'), mkdir('data/splits'); end
writetable(trainTbl, 'data/splits/train.csv');
writetable(testTbl,  'data/splits/test.csv');
fprintf('Train: %d rows, Test: %d rows. No overlap confirmed.\n', height(trainTbl), height(testTbl));
end
