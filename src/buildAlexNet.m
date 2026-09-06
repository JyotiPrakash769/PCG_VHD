function net = buildAlexNet(cfg)
% BUILDALEXNET  Load pretrained AlexNet (torchvision-equivalent, stock
% MATLAB alexnet from Deep Learning Toolbox), replace final FC/classifier
% for cfg.classes (5-way). This is a faithful AlexNet-transfer-learning
% reproduction per the PPT; NOT verified to be the paper's exact custom
% CNN architecture (paper's own layer config is not publicly specified).
if nargin < 1, cfg = config(); end
if ~exist('alexnet','file')
    error('Deep Learning Toolbox + Add-On "AlexNet" required (Add-On Explorer).');
end
baseNet = alexnet;
lg = layerGraph(baseNet.Layers);

lg = replaceLayer(lg, 'fc8', fullyConnectedLayer(numel(cfg.classes), ...
        'Name','fc_out', 'WeightLearnRateFactor',10, 'BiasLearnRateFactor',10));
lg = replaceLayer(lg, 'prob', softmaxLayer('Name','softmax_out'));
lg = replaceLayer(lg, 'output', classificationLayer('Name','classoutput'));

switch cfg.transferMode
    case 'freezeConv'
        lg = freezeLayers(lg, {'conv1','conv2','conv3','conv4','conv5'});
    case 'classifierOnly'
        lg = freezeLayers(lg, {'conv1','conv2','conv3','conv4','conv5','fc6','fc7'});
    case 'fineTuneAll'
        % no freezing
    otherwise
        error('Unknown cfg.transferMode: %s', cfg.transferMode);
end
net = lg;
end

function lg = freezeLayers(lg, names)
for i = 1:numel(names)
    l = lg.Layers(strcmp({lg.Layers.Name}, names{i}));
    if isprop(l,'WeightLearnRateFactor'), l.WeightLearnRateFactor = 0; end
    if isprop(l,'BiasLearnRateFactor'), l.BiasLearnRateFactor = 0; end
    lg = replaceLayer(lg, names{i}, l);
end
end
