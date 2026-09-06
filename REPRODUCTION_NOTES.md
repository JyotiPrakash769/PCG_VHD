# Reproduction Notes

## Verified directly from the uploaded PPT (22 slides, ADDANKI HARIKA, ITER)
- Title/task matches exactly: 5-class VHD classification (AS, MR, MS, MVP,
  Normal) from PCG signals.
- **Dataset**: 1000 signals, github.com/yaseen21khan/, 200/class, **8 kHz,
  16-bit**, signal length **1.1556–3.9929 s** (variable length — not fixed).
- **Split**: 150 train / 50 test per class — matches config.
- **Block diagram** (slide 6): load by class folder → CWT → TFR images →
  train/test split → AlexNet fine-tuned → evaluate accuracy. Matches this
  implementation's pipeline order exactly.
- **Preprocessing** (slide 8): "Normalization", "CWT and Spectrogram
  transform 1D→2D", "RGB Conversion", "Data Augmentation and Splitting" —
  named as steps but with **no numeric parameters given**.
- **Wavelet** (slides 9–10): explicitly "Amor" = analytic Morlet, explicitly
  CWT not DWT, explicitly used to produce scalograms. No voices-per-octave,
  frequency range, or scale count given.
- **CNN**: stock **AlexNet**, 5 conv + 3 FC layers, ReLU, transfer learning
  ("fine-tuned"). No custom-CNN alternative shown — mode B (custom CNN) is
  correctly out of scope; this repo's AlexNet-only path is the right one.
- **Metrics**: standard confusion matrix + accuracy/precision/recall/F1.
- **Reported numbers — confirmed inconsistency, exactly as flagged**:
  classification-report table (slide 19) says **0.996** overall accuracy;
  conclusion text (slide 20) says **99.2%**. Both are the PPT's own numbers,
  kept separate in `REPRODUCTION_REPORT.md`, never merged or "corrected."
  Per-class table (slide 19): AS P/R/F1 .98/1/.99, MR 1/1/1, MS 1/1/1,
  MVP 1/.98/.99, Normal 1/1/1.
- **No occlusion maps or Deep Dream anywhere in the PPT.** Those came only
  from the referenced IEEE TIM paper (slide 21 references list). Explain-
  ability is a paper-level bonus here, not a PPT requirement — kept in the
  code as optional (`main explain`), not part of `main all`.

## Still unverifiable (paper is paywalled, no numeric detail in PPT)
- Exact CWT voices-per-octave / scale count / frequency range.
- Exact image size fed to AlexNet, colormap, augmentation type/magnitude.
- Whether "Spectrogram" mentioned alongside CWT on slide 8 was an
  alternative representation actually tried, or just descriptive text.
  Treated as descriptive; CWT-scalogram is the sole representation
  implemented, per slides 9-10 which describe only the CWT path in detail.

## Implementation choices (SOURCE NOT SPECIFIED — unchanged from before)
- `cwtfilterbank('Wavelet','amor')`, 12 voices/octave, image 224→227,
  jet colormap, bilinear resize, occlusion mask 28px/stride 14px, Deep
  Dream 20 iters/3 pyramid levels — all still implementation choices.

## Explicit non-claims
This code will not report 0.996 or 99.2% unless an actual run produces
that number. Both PPT-reported figures are recorded in
`REPRODUCTION_REPORT.md` as the *source's own* numbers for comparison only.
