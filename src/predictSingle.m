function [predClass, confidence, probs] = predictSingle(wavPath, cfg)
% PREDICTSINGLE  Run one WAV file through preprocessing -> scalogram ->
% trained model. Prints predicted class, confidence, and full class
% probability vector. Optionally saves the input scalogram.
if nargin < 2, cfg = config(); end
S = load(cfg.modelOutPath, 'trainedNet'); net = S.trainedNet;

[x, fs] = loadPCG(wavPath, cfg);
rgb = computeScalogram(x, fs, cfg);
img = imresize(rgb, [227 227]);

scores = predict(net, img);
[confidence, idx] = max(scores);
predClass = net.Layers(end).Classes(idx);
probs = table(net.Layers(end).Classes, scores', 'VariableNames', {'class','probability'});

fprintf('Predicted class: %s\n', string(predClass));
fprintf('Confidence: %.2f%%\n', confidence*100);
disp(probs);

if ~exist('outputs/predictions','dir'), mkdir('outputs/predictions'); end
[~, name] = fileparts(wavPath);
imwrite(rgb, fullfile('outputs/predictions', [name '_scalogram.png']));
end
