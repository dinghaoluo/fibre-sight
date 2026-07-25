# Fibre Sight ch2 v1 model card

## model

`fibre_sight_ch2_v1.pt` is a single-channel checkpoint for the package's small U-Net implementation. The network has one input channel, one output channel, 24 base channels, and a depth of four. Fibre Sight normalises each two-dimensional channel-2 reference image between its 1st and 99.7th percentiles, predicts a foreground probability map, thresholds that map, removes connected components smaller than 45 pixels, and converts the remaining components into editable `xpix` and `ypix` ROI dictionaries. The [methods record](METHODS.md) describes the labels, split, training run and operating-point decision.

| field | value |
| --- | --- |
| Registry key | `ch2_v1` |
| Checkpoint | `src/fibre_sight/models/fibre_sight_ch2_v1.pt` |
| Input | one two-dimensional channel-2 reference array |
| Output | one foreground probability map and an ROI dictionary |
| Confidence threshold | `0.25` |
| Minimum component size | `45` pixels |
| Test-time augmentation | four flip views, averaged |
| Normalisation percentiles | `[1, 99.7]` |
| Selected epoch | `37` |
| Checkpoint size | `4,377,285` bytes |
| SHA-256 | `a715d1d3ca7e689a374446437f2abbca7113ac75f648d17d19f42759cff1ba89` |

The checkpoint contains 92 model tensors and the metadata required for inference. Training images, ROI dictionaries, optimiser state and local filesystem paths remain outside the file.

## training and evaluation data

The model was trained on the authors' channel-2 reference images and corresponding hand-curated axon ROI dictionaries. Full images and ROI dictionaries are not distributed with Fibre Sight. The final manifest contained 53 sessions from four animals and 1,296 curated ROIs: 37 training sessions, 7 validation sessions and 9 test sessions. All four animals appeared in each split, so the test set contains unseen sessions from the same animals and acquisition source. The repository includes three unlabelled 256 × 256 crops, and no animal or session identifiers are stored in the public training record.

The selected checkpoint reached a best validation pixel Dice score of `0.7821986335` at epoch 37. Test-set evaluation used threshold `0.25`, minimum size `45`, and test-time augmentation, with each metric averaged across the nine test sessions:

| metric | mean |
| --- | ---: |
| Dice | 0.7226541 |
| Intersection over union | 0.5687023 |
| Precision | 0.5957682 |
| Recall | 0.9260965 |
| F2 | 0.8312405 |

These scores measure foreground-pixel overlap after connected-component filtering; an individual ROI split or merge error can leave the aggregate nearly unchanged.

The checkpoint epoch was selected on validation data. The final postprocessing setting was settled after validation sweeps, visual review and later inspection of test behaviour, so these numbers describe the released configuration; the [methods record](METHODS.md#choosing-the-released-operating-point) retains that decision history.

## intended use

The checkpoint supplies initial ROI proposals for images acquired and processed similarly to the training data. The workbench exposes the confidence threshold, minimum component size, confidence map, and ROI editor, so each prediction can be inspected and corrected before its dictionary is saved or returned to a later training round.

The model is a research segmentation aid and has not been evaluated for clinical, diagnostic, or safety-critical use.

## limits

Recall was `0.9260965`, while precision was `0.5957682`; false-positive foreground is therefore expected at this operating point. Changes in indicator, microscope, field size, preprocessing, signal range, or axon morphology may reduce performance. Percentile normalisation cannot correct acquisition changes that alter spatial structure, background texture, or the appearance of labelled fibres.

The nine test sessions came from the same data source as the training set; performance on other laboratories' images is unknown.

## licence

The bundled checkpoint is released under the repository's [MIT License](LICENSE).
