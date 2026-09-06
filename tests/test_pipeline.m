function tests = test_pipeline
tests = functiontests(localfunctions);
end

function testConfigClasses(testCase)
cfg = config();
verifyEqual(testCase, numel(cfg.classes), 5);
end

function testScalogramShape(testCase)
cfg = config();
x = randn(cfg.fs_expected*3, 1);  % 3s synthetic signal
rgb = computeScalogram(x, cfg.fs_expected, cfg);
verifyEqual(testCase, size(rgb), [cfg.imageSize(1) cfg.imageSize(2) 3]);
verifyTrue(testCase, isa(rgb,'uint8'));
end

function testNoNaNInPreprocessing(testCase)
cfg = config();
x = [1; NaN; -Inf; 0.5];
x(~isfinite(x)) = 0;
verifyTrue(testCase, all(isfinite(x)));
end

function testSplitNoLeakage(testCase)
% Requires data/scalograms populated; skip if absent.
if ~exist('data/scalograms','dir')
    assumeFail(testCase, 'No scalograms present — run generateScalograms first.');
end
cfg = config();
[trainTbl, testTbl] = splitData(cfg);
verifyTrue(testCase, isempty(intersect(trainTbl.filepath, testTbl.filepath)));
end

function testProbabilitiesSumToOne(testCase)
% Only meaningful with a trained model present.
cfg = config();
if ~exist(cfg.modelOutPath, 'file')
    assumeFail(testCase, 'No trained model present — run trainModel first.');
end
S = load(cfg.modelOutPath, 'trainedNet');
img = uint8(rand(227,227,3)*255);
scores = predict(S.trainedNet, img);
verifyEqual(testCase, sum(scores), 1, 'AbsTol', 1e-3);
end
