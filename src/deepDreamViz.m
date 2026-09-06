function deepDreamViz(cfg)
% DEEPDREAMVIZ  Class-conditioned activation maximization ("Deep Dream")
% for each of the 5 classes, using MATLAB's built-in deepDreamImage on the
% final fully-connected layer. Exact equivalence to the paper's own
% Deep Dream implementation is NOT established — documented choice.
if nargin < 1, cfg = config(); end
S = load(cfg.modelOutPath, 'trainedNet'); net = S.trainedNet;

if ~exist('deepDreamImage','file')
    error('Deep Learning Toolbox function deepDreamImage not found.');
end
outLayerName = 'fc_out';
if ~exist('outputs/figures/deep_dream','dir'), mkdir('outputs/figures/deep_dream'); end

for c = 1:numel(cfg.classes)
    img = deepDreamImage(net, outLayerName, c, ...
        'NumIterations', 20, 'PyramidLevels', 3, 'PyramidScale', 1.4); % SOURCE NOT SPECIFIED
    f = figure('Visible','off'); imshow(img);
    title(sprintf('Deep Dream — class %s', cfg.classLabels{c}));
    saveas(f, fullfile('outputs/figures/deep_dream', sprintf('%s.png', cfg.classes{c})));
    close(f);
end
end
