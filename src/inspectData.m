function summary = inspectData(cfg)
% INSPECTDATA  Recursively scan cfg.rawDir/<class>/*.wav, report counts,
% sample rates, durations, and flag corrupt files. Never silently drops files.
if nargin < 1, cfg = config(); end
summary = struct('class',{},'nFiles',{},'fs',{},'meanDur',{},'corrupt',{});
fprintf('%-8s %8s %10s %12s %10s\n','Class','Files','Fs(Hz)','MeanDur(s)','Corrupt');
for c = 1:numel(cfg.classes)
    cls = cfg.classes{c};
    d = dir(fullfile(cfg.rawDir, cls, '*.wav'));
    fsList = []; durList = []; corrupt = {};
    for i = 1:numel(d)
        fp = fullfile(d(i).folder, d(i).name);
        try
            info = audioinfo(fp);
            fsList(end+1) = info.SampleRate; %#ok<AGROW>
            durList(end+1) = info.Duration;  %#ok<AGROW>
        catch ME
            corrupt{end+1} = d(i).name; %#ok<AGROW>
            warning('Corrupt/unreadable file kept in report: %s (%s)', fp, ME.message);
        end
    end
    summary(c).class = cls;
    summary(c).nFiles = numel(d);
    summary(c).fs = unique(fsList);
    summary(c).meanDur = mean(durList);
    summary(c).corrupt = corrupt;
    fprintf('%-8s %8d %10s %12.3f %10d\n', cls, numel(d), mat2str(unique(fsList)), mean(durList), numel(corrupt));
end
total = sum([summary.nFiles]);
fprintf('\nTotal files: %d (expected 1000, 200/class)\n', total);
if ~exist('outputs/metrics','dir'), mkdir('outputs/metrics'); end
save('outputs/metrics/dataset_summary.mat','summary');
end
