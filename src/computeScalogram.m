function rgb = computeScalogram(x, fs, cfg)
% COMPUTESCALOGRAM  1D PCG -> analytic Morlet ('amor') CWT -> magnitude
% scalogram -> colormapped RGB image, resized to cfg.imageSize.
% Requires Wavelet Toolbox (cwtfilterbank).
if nargin < 3, cfg = config(); end

fb = cwtfilterbank('SignalLength', numel(x), ...
                    'SamplingFrequency', fs, ...
                    'Wavelet', cfg.wavelet, ...
                    'VoicesPerOctave', cfg.voicesPerOctave);
if ~isempty(cfg.freqRange)
    fb.FrequencyLimits = cfg.freqRange;
end
cfs = abs(fb.wt(x));            % time-frequency magnitude, analytic CWT

% Consistent per-image normalization to [0,1] before colormapping.
cfs = cfs - min(cfs(:));
mx = max(cfs(:));
if mx > 0, cfs = cfs / mx; end

img = imresize(cfs, cfg.imageSize, 'bilinear');   % documented: bilinear interp
cmapFn = str2func(cfg.colormap);
cmap = cmapFn(256);
idx = gray2ind(img, 256);
rgb = ind2rgb(idx, cmap);       % H x W x 3, double in [0,1]
rgb = im2uint8(rgb);
end
