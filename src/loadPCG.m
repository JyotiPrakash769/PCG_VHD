function [x, fs] = loadPCG(filepath, cfg)
% LOADPCG  Load a WAV file, force mono, resample to cfg.fs_expected if
% needed, peak-normalize if cfg.normalize is true. Deterministic.
if nargin < 2, cfg = config(); end
[x, fs] = audioread(filepath);
if size(x,2) > 1
    x = mean(x, 2);
end
if ~isempty(cfg.fs_expected) && fs ~= cfg.fs_expected
    x = resample(x, cfg.fs_expected, fs);
    fs = cfg.fs_expected;
end
x = double(x(:));
x(~isfinite(x)) = 0;
if cfg.normalize
    m = max(abs(x));
    if m > 0
        x = x ./ m;
    end
end
end
