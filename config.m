function cfg = config()
% CONFIG  Single source of truth for all pipeline parameters.
% Values marked SOURCE NOT SPECIFIED are implementation choices made
% because the PPT/repo/paper do not give an exact number.

cfg.seed = 42;

% --- Dataset ---
cfg.classes = {'AS','MR','MS','MVP','N'};      % N = Normal (repo naming)
cfg.classLabels = {'AS','MR','MS','MVP','Normal'};
cfg.rawDir = 'data/raw';                        % one subfolder per class
cfg.fs_expected = 8000;                         % PPT slide 7: 8kHz, 16-bit (confirmed, still re-verified at runtime)
cfg.durationRange = [1.1556 3.9929];            % PPT slide 7: signal length range (informational; signals kept variable-length)

% --- Preprocessing ---
cfg.normalize = true;                           % peak-normalize to [-1,1]

% --- CWT (Analytic Morlet / "amor") ---
% MATLAB's own cwtfilterbank names this wavelet 'amor' (analytic Morlet),
% matching the PPT's terminology directly.
cfg.wavelet = 'amor';
cfg.voicesPerOctave = 12;                       % SOURCE NOT SPECIFIED — implementation choice
cfg.freqRange = [];                             % [] = full supported range (auto)
cfg.imageSize = [224 224];                      % SOURCE NOT SPECIFIED — matches AlexNet input
cfg.colormap = 'jet';                            % SOURCE NOT SPECIFIED — common scalogram convention

% --- Split ---
cfg.trainPerClass = 150;                        % PPT-stated
cfg.testPerClass = 50;                          % PPT-stated
cfg.kFolds = 5;                                 % paper reports 5-fold CV separately

% --- Augmentation (off by default; image-domain only, mild) ---
cfg.augment = false;
cfg.augMaxRotation = 5;                         % degrees, SOURCE NOT SPECIFIED

% --- Training ---
cfg.miniBatchSize = 32;
cfg.maxEpochs = 20;                             % SOURCE NOT SPECIFIED — smoke-testable
cfg.initialLearnRate = 1e-4;
cfg.validationSplit = 0.15;                     % taken out of the 750 training images
cfg.transferMode = 'fineTuneAll';               % 'freezeConv' | 'classifierOnly' | 'fineTuneAll'

cfg.modelOutPath = 'outputs/models/best_model.mat';
end
