# FibreSight methods

`fibre_sight_ch2_v1.pt` came from 53 hand-labelled dLight imaging sessions and an 80-epoch training run completed on 12 May 2026. The workbench controls, labelling decisions and training record are kept here; fixed artefact details, intended use and licence remain in the [model card](MODEL_CARD.md).

## prediction controls

The network produces one foreground probability for every image pixel. The controls below convert that probability map into connected ROI components; changing `prediction threshold` or `minimum ROI area (pixels)` reuses the probability map already held in memory.

| control | default | effect |
| --- | ---: | --- |
| `prediction threshold` | `0.25` | Confidence threshold applied to each pixel. Lower values retain weaker and dimmer responses; higher values retain fewer pixels. |
| `minimum ROI area (pixels)` | `45` pixels | Removes connected components smaller than this area after thresholding. Dimness sensitivity stays unchanged. |
| four-view TTA | on | Averages the original image and its horizontal, vertical and two-axis flips. The CLI and API can disable it. |

A faint process with visible confidence can sometimes be recovered by lowering the `prediction threshold`. A process absent from the confidence map falls outside the class learned by this checkpoint; `minimum ROI area (pixels)` acts only on components already present.

## MSER controls

The older MSER route remains in the workbench because it was the proposal method used before the U-Net existed, and it is still useful when a new image falls outside the checkpoint's training distribution. MSER has its own image preparation, stability and shape filters; model prediction uses a separate probability map and ignores these values.

| control | default | effect |
| --- | ---: | --- |
| `brightness threshold` | `85.0` | Brightness percentile applied before candidate detection. |
| `candidate area minimum` | `30` pixels | Lower bound for regions passed to MSER. |
| `candidate area maximum` | `15000` pixels | Upper bound for regions passed to MSER. |
| `delta` | `5` | Intensity step used when MSER tests region stability. |
| `maximum variation` | `1.2` | Stability filter; lower values are stricter. |
| `ROI area minimum` | `30` pixels | Lower bound for final ROIs after shape filtering. |
| `eccentricity minimum` | `0.75` | Higher values favour elongated regions. |
| `aspect-ratio minimum` | `1.2` | Minimum long-axis to short-axis ratio. |
| `solidity minimum` | `0.1` | Minimum filled-area ratio accepted for a candidate. |
| `thinness maximum` | `0.8` | Compactness ceiling; lower values favour thinner regions. |
| `tophat kernel` | `11` pixels | Scale of the background-removal filter used before MSER. |
| `CLAHE clip` | `2.0` | Strength of local contrast equalisation. |

Fixed ROIs are written into the label image before a new MSER pass, so a hand decision survives later changes to these controls.

## labels and data split

Each session contributed one two-dimensional `ref_mat_ch2.npy` reference and one curated ROI dictionary. Every ROI dictionary stored the `xpix` and `ypix` coordinates of its accepted axonal regions; overlapping coordinates were combined into one foreground mask for training.

Dim, out-of-plane processes were generally left unlabelled. Their fluorescence spread into the background and could not be assigned cleanly to one axonal ROI for later dLight analysis. The target describes the in-plane processes which a curator would retain; fibre-like traces outside that convention remain background.

The final manifest contained 53 usable sessions from four animals and 1,296 curated ROIs:

| split | sessions | curated ROIs | foreground pixels |
| --- | ---: | ---: | ---: |
| training | 37 | 892 | 359,351 |
| validation | 7 | 177 | 72,749 |
| test | 9 | 227 | 85,698 |

The split was made at session level. All four animals appeared in each split, so the test set is session-held-out within the same animals and acquisition source. Full reference images and ROI dictionaries remain private. The three arrays under `examples/` are unlabelled 256 × 256 crops released for running the public software.

## image preparation and sampling

Each full reference image was clipped and scaled between its 1st and 99.7th intensity percentiles. The training loader then drew 24 patches of 256 × 256 pixels per session during each epoch; 75% of crop centres were sampled from foreground pixels, whilst the remainder were placed randomly so that background-only patches remained part of the task. Validation used eight patches per session without augmentation.

Training augmentation used independent horizontal and vertical flips, lossless rotations in 90-degree steps, intensity scaling within ±15%, an intensity offset within ±0.0375, and Gaussian noise with a standard deviation of `0.02` after normalisation. The recipe YAML exposes the rotation and noise settings under `data.augmentation` (`rotation_90`, `noise_sd`); each flip has a fixed probability of `0.5`, and the intensity jitter is also fixed loader behaviour. The released checkpoint used both rotations and noise at the values above. The random crop and augmentation sequence changed with the epoch and was seeded with `7`; accelerator kernels may still introduce numerical variation between runs.

## network and optimisation

The selected model is the package's `SmallUNet`: one input channel, one foreground-logit output channel, four resolution levels and base width `24`, giving channel widths of 24, 48, 96 and 192. Each level uses two 3 × 3 convolution, batch-normalisation and ReLU blocks; downsampling uses max pooling, whilst the decoder uses transposed convolution and skip connections.

The loss was the equally weighted sum of binary cross-entropy with logits and soft Dice loss. AdamW used a learning rate of `3e-4`, weight decay of `1e-5` and batch size `8`; training ran for 80 epochs with automatic mixed precision when CUDA was available. The architecture, optimisation and augmentation choices remain in [`ch2_unet.yaml`](src/fibre_sight/configs/ch2_unet.yaml); the supplied recipe now uses the project seed of `42`, whilst the released checkpoint used `7` as recorded above.

## checkpoint selection and training history

The training loop saved the weights with the highest mean validation patch Dice, calculated at a probability threshold of `0.5` before connected-component filtering. Epoch 37 reached `0.7821986335` and became the released checkpoint; validation loss reached its minimum later, at epoch 59, but validation Dice was the selection rule fixed in the training code.

![Training and validation loss with validation Dice across 80 epochs](docs/images/training-history.png)

The plotted values are available as [`training-history.csv`](docs/data/training-history.csv). The public checkpoint contains model weights and inference metadata only.

## choosing the released operating point

Threshold and minimum component size were tuned after the network weights had been fixed. A validation sweep reused the saved probability maps and compared Dice, F2, precision, recall and component count for each setting; four-view flip averaging was included in the sweep used for the released workbench.

| validation setting | threshold | minimum size | Dice | F2 | precision | recall | predicted components per session |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F2 maximum | 0.20 | 30 | 0.7686 | 0.8435 | 0.6778 | 0.9091 | 40.1 |
| released | 0.25 | 45 | 0.7753 | 0.8368 | 0.6983 | 0.8888 | 35.1 |
| Dice maximum | 0.60 | 60 | 0.7966 | 0.8016 | 0.7974 | 0.8078 | 29.0 |

The released `0.25 / 45` setting sits between the most recall-heavy and Dice-heavy validation optima. ROI overlays and component counts were inspected alongside these pixel scores because a setting with high foreground overlap can still split one axon into several proposals, or merge neighbouring axons into one.

Gradient training and epoch selection used the training and validation splits. The nine test sessions entered the decision record later, when their results were consulted whilst settling the public postprocessing setting; the reported test scores are therefore a descriptive evaluation of the released configuration. A future animal-held-out evaluation with a fixed operating point would provide a cleaner external estimate.

## evaluation metrics

Evaluation thresholded the full-image probability maps, removed components smaller than 45 pixels, and calculated one set of foreground-pixel scores for each session. The reported values are means across the nine test sessions:

| metric | definition | mean |
| --- | --- | ---: |
| Dice | `2TP / (2TP + FP + FN)` | 0.7226541 |
| intersection over union | `TP / (TP + FP + FN)` | 0.5687023 |
| precision | `TP / (TP + FP)` | 0.5957682 |
| recall | `TP / (TP + FN)` | 0.9260965 |
| F2 | `5TP / (5TP + 4FN + FP)` | 0.8312405 |

The released setting predicted a mean of 38.2 connected components per test session; the curated targets contained 25.2. Pixel metrics describe foreground overlap and give only an indirect account of ROI identity, so split and merge errors still require inspection in the editor.

## retained experiments

Three alternative recipes remain beside the selected baseline. Each changed one dimension of the training setup whilst holding the rest fixed; the training pipeline seeds numpy and torch from the recipe's `train.seed` value before model initialisation, so single-variable comparisons are reproducible.

**`ch2_unet_aux_recall.yaml`** added a dilated support target (binary dilation with radius 3 around each foreground mask) and replaced soft Dice with Tversky loss (`alpha=0.15, beta=0.85`), penalising false negatives far more than false positives. The model widened to base channels 32, output two channels (foreground + support), and trained for 100 epochs at a higher foreground crop fraction (0.85). This was the recall-heavy extreme; the released postprocessing already favours recall through a low confidence threshold, so the additional recall from a modified loss did not justify the added complexity.

**`ch2_unet_512.yaml`** doubled the crop size to 512 pixels and widened the base channels to 32, giving the network more spatial context per patch. This was a sanity check that the 256-pixel baseline was not leaving performance on the table; the smaller model remained the released checkpoint.

**`ch2_unet_attention.yaml`** added attention gates to the skip connections of the standard `SmallUNet` whilst keeping the 256-pixel crop, base width 24, BCE + Dice loss, and the baseline seed unchanged. The gate projects skip features and the upsampled decoder features into a shared low-dimensional space (half the minimum of skip and gate channels), applies ReLU then a 1×1 convolution to a single-channel sigmoid score, and multiplies that score back onto the skip path before concatenation. The decoder path itself is unmodified; only the skip contribution is gated. In our tests the attention-gated variant performed comparably to the standard U-Net on the same data. The recipe is included for cases where the training data contains more heterogeneous backgrounds and the skip connections might benefit from learned suppression of irrelevant features.

The 256-pixel ungated baseline remained the released model. The comparison stopped with these working trials whilst the labelling and curation loop was still changing. The full [training guide](TRAINING.md) describes each recipe and how to write a new one.

## scope

The checkpoint was trained and tested within one dLight acquisition source. Percentile normalisation handles simple changes in signal scale. A new indicator, microscope point-spread function, field size, background texture, or labelling convention changes the spatial problem and remains outside that correction. Images from another domain need hand-labelled checks; repeated misses in the confidence map are evidence for retraining, whilst below-threshold responses can be explored with the prediction controls at the top of this file.

## fluorescence extraction and dF/F

Fluorescence extraction and dF/F calculation are separate immutable NWB stages. Extraction measures the mean, median, interquartile range and valid-pixel fraction for each ROI and its surrounding pixels in both channels. 'Surround' is used as the general name because 'annulus' describes one available geometry, whilst 'neuropil' names biological tissue which cannot be inferred from these pixels.

Adaptive surrounds use four-connected pixel growth. The first five pixels around an ROI are excluded; every curated ROI pixel is then removed, and the outer boundary grows in five-pixel steps until more than 350 surrounding pixels are available or the image boundary has been reached. This reproduces the geometry used by Suite2p, with the five-pixel inner exclusion inherited from `lc-ca1-project`. Fixed mode instead selects pixels whose Euclidean distance from the ROI lies between the chosen inner and outer radii. Overlapping ROIs remain separate, but neither surround can include pixels assigned to any curated ROI.

The derived stage selects either the raw mean (default) or median. For each channel it calculates independent ROI, surround and surround-corrected traces. Surround subtraction happens before baseline estimation:

```text
F_corrected = F_roi - coefficient * F_surround
dFF = (F - F0) / F0
```

The default coefficient is `0.7`; it is fixed for the run and is never fitted separately for each ROI. `F0` is a centred, reflectively padded rolling 20th percentile over 300 seconds. NaNs are omitted from each percentile. Frames rejected by `processing/quality_control/registration_qc/analysis_valid` are stored as NaN, excluded from every baseline, and never interpolated in the canonical traces. No temporal median filter is applied.

Signal and control channels receive the same independent calculation. Control correction defaults to `none`. The optional deterministic mode subtracts surround-corrected control dF/F from surround-corrected signal dF/F; it does not subtract raw green and red counts, fit a coefficient, or treat tdTomato as an isosbestic channel. The dLight, GRABNE and nLight papers establish different sensor kinetics and acquisition contexts; Hamid *et al.* analysed tdTomato separately, whilst Keevers and Jean-Richard-dit-Bressel show how regression choices can change fibre-photometry results and Zhang *et al.* show the remaining haemodynamic limits. FibreSight therefore stores the uncorrected channel traces alongside any optional control-corrected result.

| parameter or decision | default | basis |
| --- | ---: | --- |
| surround geometry | adaptive | Suite2p-style four-connected growth [1] |
| inner surround exclusion | `5 px` | `lc-ca1-project` Suite2p configuration; Suite2p's general default is smaller [1] |
| adaptive target | more than `350` pixels | Suite2p minimum surrounding-pixel rule [1] |
| adaptive growth step | `5 px` | Suite2p mask construction [1] |
| raw ROI statistic | mean | selectable analysis choice; median is retained as an alternative |
| surround coefficient | `0.7` | Suite2p and the existing `lc-ca1-project` axonal pipeline [1] |
| baseline | centred rolling 20th percentile | existing `lc-ca1-project` axonal pipeline; this exact percentile is not presented as a cross-sensor standard |
| baseline window | `300 s` | `9,000` frames at 30 Hz in `lc-ca1-project` |
| temporal median filter | none | no canonical filter in the source axonal pipeline |
| control correction | none | sensor and control-channel methods differ across preparations [2-8] |
| optional control correction | signal dF/F minus control dF/F | fixed, reproducible alternative; no per-ROI regression [6, 7] |

## fluorescence references

1. Pachitariu M, Stringer C, Dipoppa M, *et al.* (2017). Suite2p: beyond 10,000 neurons with standard two-photon microscopy. *bioRxiv*. [doi:10.1101/061507](https://doi.org/10.1101/061507).
2. Patriarchi T, Cho JR, Merten K, *et al.* (2018). Ultrafast neuronal imaging of dopamine dynamics with designed genetically encoded sensors. *Science* 360(6396):eaat4422. [doi:10.1126/science.aat4422](https://doi.org/10.1126/science.aat4422).
3. Feng J, Zhang C, Lischinsky JE, *et al.* (2019). A genetically encoded fluorescent sensor for rapid and specific *in vivo* detection of norepinephrine. *Neuron* 102(4):745-761.e8. [doi:10.1016/j.neuron.2019.02.037](https://doi.org/10.1016/j.neuron.2019.02.037).
4. Feng J, Dong H, Lischinsky JE, *et al.* (2024). Monitoring norepinephrine release *in vivo* using next-generation GRABNE sensors. *Neuron* 112(12):1930-1942.e6. [doi:10.1016/j.neuron.2024.03.001](https://doi.org/10.1016/j.neuron.2024.03.001).
5. Kagiampaki Z, Weng Y, Zhang C, *et al.* (2023). Sensitive multicolor indicators for monitoring norepinephrine *in vivo*. *Nature Methods* 20(9):1426-1436. [doi:10.1038/s41592-023-01959-z](https://doi.org/10.1038/s41592-023-01959-z).
6. Hamid AA, Frank MJ and Moore CI (2021). Wave-like dopamine dynamics as a mechanism for spatiotemporal credit assignment. *Cell* 184(10):2733-2749.e16. [doi:10.1016/j.cell.2021.03.046](https://doi.org/10.1016/j.cell.2021.03.046).
7. Keevers N and Jean-Richard-dit-Bressel P (2025). Obtaining artifact-corrected signals in fiber photometry *via* isosbestic signals, robust regression, and dF/F calculations. *Neurophotonics* 12(2):025003. [doi:10.1117/1.NPh.12.2.025003](https://doi.org/10.1117/1.NPh.12.2.025003).
8. Zhang Y, Yee P, Zacharias NM, *et al.* (2022). Spectral fiber photometry derives hemoglobin concentration changes for accurate measurement of fluorescent sensor activity. *Cell Reports Methods* 2(7):100243. [doi:10.1016/j.crmeth.2022.100243](https://doi.org/10.1016/j.crmeth.2022.100243).
