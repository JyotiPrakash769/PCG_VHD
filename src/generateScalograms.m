function generateScalograms(cfg)
% GENERATESCALOGRAMS  Batch: every WAV in cfg.rawDir/<class>/ -> RGB
% scalogram PNG in data/scalograms/<class>/. Also saves one example
% figure per class under outputs/figures/.
if nargin < 1, cfg = config(); end
if ~exist('outputs/figures','dir'), mkdir('outputs/figures'); end

for c = 1:numel(cfg.classes)
    cls = cfg.classes{c};
    outDir = fullfile('data/scalograms', cls);
    if ~exist(outDir,'dir'), mkdir(outDir); end
    d = dir(fullfile(cfg.rawDir, cls, '*.wav'));
    fprintf('[%s] %d files\n', cls, numel(d));
    for i = 1:numel(d)
        fp = fullfile(d(i).folder, d(i).name);
        [x, fs] = loadPCG(fp, cfg);
        rgb = computeScalogram(x, fs, cfg);
        [~, name] = fileparts(d(i).name);
        imwrite(rgb, fullfile(outDir, [name '.png']));
        if i == 1   % one example per class saved as a figure
            f = figure('Visible','off');
            imshow(rgb); title(sprintf('%s scalogram (analytic Morlet CWT)', cls));
            saveas(f, fullfile('outputs/figures', sprintf('example_%s.png', cls)));
            close(f);
        end
    end
end
end
