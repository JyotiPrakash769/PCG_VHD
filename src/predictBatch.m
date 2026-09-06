function T = predictBatch(inputDir, cfg)
% PREDICTBATCH  Run every WAV under inputDir through the model. If the
% file lives in a class-named subfolder, records true_label too. Writes
% outputs/predictions/predictions.csv.
if nargin < 2, cfg = config(); end
S = load(cfg.modelOutPath, 'trainedNet'); net = S.trainedNet;

d = dir(fullfile(inputDir, '**', '*.wav'));
n = numel(d);
filenames = cell(n,1); trueLabels = cell(n,1); predLabels = cell(n,1);
probMat = zeros(n, numel(cfg.classes));

for i = 1:n
    fp = fullfile(d(i).folder, d(i).name);
    [x, fs] = loadPCG(fp, cfg);
    rgb = computeScalogram(x, fs, cfg);
    img = imresize(rgb, [227 227]);
    scores = predict(net, img);
    [~, idx] = max(scores);
    filenames{i} = d(i).name;
    predLabels{i} = char(net.Layers(end).Classes(idx));
    probMat(i,:) = scores;

    [~, parentFolder] = fileparts(d(i).folder);
    if any(strcmp(parentFolder, cfg.classes))
        trueLabels{i} = parentFolder;
    else
        trueLabels{i} = '';
    end
end

T = table(filenames, trueLabels, predLabels, 'VariableNames', {'filename','true_label','predicted_label'});
for c = 1:numel(cfg.classes)
    T.(sprintf('%s_probability', cfg.classes{c})) = probMat(:,c);
end
if ~exist('outputs/predictions','dir'), mkdir('outputs/predictions'); end
writetable(T, 'outputs/predictions/predictions.csv');
fprintf('Wrote %d predictions to outputs/predictions/predictions.csv\n', n);
end
