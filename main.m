function main(cmd, varargin)
% MAIN  CLI entry point.
%   main inspect-data
%   main generate-scalograms
%   main split
%   main train
%   main evaluate
%   main cross-validate
%   main explain --input sample.wav
%   main predict --input sample.wav
%   main predict-folder --input data/raw
%   main all
if nargin < 1, cmd = 'all'; end
addpath('src');
cfg = config();
rng(cfg.seed);

p = inputParser; p.KeepUnmatched = true;
addParameter(p, 'input', '');
parse(p, varargin{:});
inputArg = p.Results.input;

switch cmd
    case 'inspect-data'
        inspectData(cfg);
    case 'generate-scalograms'
        generateScalograms(cfg);
    case 'split'
        splitData(cfg);
    case 'train'
        trainModel(cfg);
    case 'evaluate'
        evaluateModel(cfg);
    case 'cross-validate'
        crossValidate(cfg);
    case 'explain'
        occlusionMap(inputArg, cfg);
        deepDreamViz(cfg);
    case 'predict'
        predictSingle(inputArg, cfg);
    case 'predict-folder'
        predictBatch(inputArg, cfg);
    case 'all'
        inspectData(cfg);
        generateScalograms(cfg);
        splitData(cfg);
        trainModel(cfg);
        evaluateModel(cfg);
        fprintf('\nFull pipeline complete. Run "main cross-validate" and "main explain" separately (heavier compute).\n');
    otherwise
        error('Unknown command: %s', cmd);
end
end
