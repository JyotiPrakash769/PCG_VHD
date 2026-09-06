function occlusionMap(wavPath, cfg)
% OCCLUSIONMAP  Slide an occluding gray patch over the scalogram of a
% given WAV file, record the drop in the predicted class's confidence at
% each position, and save the resulting heatmap overlaid on the image.
% Uses Deep Learning Toolbox's built-in occlusionSensitivity if available;
% falls back to a manual implementation otherwise.
if nargin < 2, cfg = config(); end
S = load(cfg.modelOutPath, 'trainedNet'); net = S.trainedNet;

[x, fs] = loadPCG(wavPath, cfg);
rgb = computeScalogram(x, fs, cfg);
img = imresize(rgb, [227 227]);

[~, ~, scores] = classify(net, img);
[~, predIdx] = max(scores);
predClass = net.Layers(end).Classes(predIdx);

if exist('occlusionSensitivity','file')
    map = occlusionSensitivity(net, img, predClass, ...
        'MaskSize', 28, 'Stride', 14);   % SOURCE NOT SPECIFIED — implementation choice
else
    map = manualOcclusion(net, img, predIdx, 28, 14);
end

f = figure('Visible','off');
imshow(img); hold on;
imagesc(map, 'AlphaData', 0.5); colormap jet; colorbar;
title(sprintf('Occlusion map — predicted %s', string(predClass)));
if ~exist('outputs/figures/occlusion','dir'), mkdir('outputs/figures/occlusion'); end
[~, name] = fileparts(wavPath);
saveas(f, fullfile('outputs/figures/occlusion', [name '_occlusion.png']));
close(f);
end

function map = manualOcclusion(net, img, predIdx, maskSize, stride)
sz = size(img); H = sz(1); W = sz(2);
map = zeros(H, W);
baseScores = predict(net, img);
baseConf = baseScores(predIdx);
for r = 1:stride:H-maskSize+1
    for c = 1:stride:W-maskSize+1
        occImg = img;
        occImg(r:r+maskSize-1, c:c+maskSize-1, :) = 128;
        s = predict(net, occImg);
        drop = baseConf - s(predIdx);
        map(r:r+maskSize-1, c:c+maskSize-1) = map(r:r+maskSize-1, c:c+maskSize-1) + drop;
    end
end
map = map / max(map(:) + eps);
end
