# PCG Valvular Heart Disease Classification — MATLAB Reproduction

Analytic-Morlet-CWT scalograms + AlexNet transfer learning for 5-class
PCG classification (AS / MR / MS / MVP / Normal), reproducing the pipeline
in the uploaded internship PPT (ADDANKI HARIKA, ITER) and cross-checked
against the public GitHub data source and the IEEE TIM 2023 paper. See
`REPRODUCTION_NOTES.md` for exactly what's verified from the PPT vs.
assumed vs. a stated implementation choice.

## Requirements
- MATLAB R2021a+ 
- **Wavelet Toolbox** (`cwtfilterbank`)
- **Deep Learning Toolbox** + **AlexNet Add-On** (Add-On Explorer)
- Audio Toolbox recommended (`audioinfo`/`audioread` robustness)

## Dataset setup
1. Download the 5 class `.rar` archives from the GitHub repo and extract
   each into `data/raw/<CLASS>/*.wav`, using folder names exactly:
   `AS`, `MR`, `MS`, `MVP`, `N`.
2. `main('inspect-data')` — verifies file counts, sample rates, durations,
   flags any corrupt files without discarding them.

## Pipeline commands
```matlab
main('inspect-data')          % dataset summary
main('generate-scalograms')   % WAV -> analytic-Morlet CWT -> RGB PNG
main('split')                 % deterministic 150/50 per-class split
main('train')                 % AlexNet transfer learning
main('evaluate')              % confusion matrix + per-class P/R/F1
main('cross-validate')        % separate 5-fold CV protocol (heavy)
main('explain','input','data/raw/AS/some_file.wav')   % occlusion + deep dream
main('predict','input','data/raw/AS/some_file.wav')   % single-file inference
main('predict-folder','input','data/raw')             % batch inference -> CSV
main('all')                   % inspect -> scalograms -> split -> train -> evaluate
```

## Outputs
- `data/scalograms/<class>/*.png` — generated scalograms
- `data/splits/{train,test}.csv` — deterministic split (leakage-checked)
- `outputs/models/best_model.mat` — trained network + config
- `outputs/metrics/` — training history JSON, classification report CSV,
  confusion matrix, CV results
- `outputs/figures/` — training curves, example scalograms, confusion
  matrix, occlusion maps, Deep Dream images

## Reported (source) vs. this implementation's results
PPT slide 19 (classification report): **99.6%** overall accuracy.
PPT slide 20 (conclusion text): **99.2%** — these two are the PPT's own
inconsistency, both kept in `REPRODUCTION_REPORT.md`, neither "corrected."
Paper-reported (Bhardwaj, Singh & Joshi, IEEE TIM 2023): 99.6% highest
5-fold CV accuracy, 98.32% overall, 93.07% binary Abnormal/Normal on
PhysioNet. **None of these numbers are assumed to reproduce** —
`main('evaluate')` and `main('cross-validate')` print and save whatever
this implementation actually achieves.

## Limitations
- CWT scale/voice parameters, exact image size, colormap, and occlusion/
  Deep Dream settings aren't specified in the PPT — implementation choices,
  flagged in `REPRODUCTION_NOTES.md`.
- Occlusion maps / Deep Dream are in the referenced paper, not the PPT —
  included as optional (`main explain`), not part of `main all`.
- Not run end-to-end here — no MATLAB runtime or the actual dataset
  available in this environment.

## Citation
Yaseen, G.-Y. Son, S. Kwon, "Classification of Heart Sound Signal Using
Multiple Features," Applied Sciences, 8(12):2344, 2018.
Bhardwaj, Singh, Joshi, "Explainable Deep Convolutional Neural Network
for Valvular Heart Diseases Classification Using PCG Signals," IEEE
Trans. Instrum. Meas., 2023. DOI: 10.1109/TIM.2023.3274174.
