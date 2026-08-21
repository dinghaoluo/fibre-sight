# public examples

## segmentation inputs

Three unlabelled 256 × 256 arrays have same-stem PNG previews:

- `lab-fibresight-demo-test.npy` comes from one session in the test split and is used by the README command;
- `lab-fibresight-demo-train-01.npy` and `lab-fibresight-demo-train-02.npy` come from two sessions in the training split.

The arrays let `fibre-sight-predict` and the GUI run from a fresh clone; the PNGs let the same inputs be inspected without NumPy. Their full source images and hand-labelled ROI dictionaries remain private. Performance is reported across the complete test sessions in the [model card](../MODEL_CARD.md), not on these selected crops.

Predictions and GUI edits are saved under `workspace/output/`, leaving the shipped arrays unchanged. The six files are released under the repository's [MIT License](../LICENSE).

## complete session

`full_session_demo_workflow.png` follows internal session `A126i-20250606-02` from six interleaved TIFF files through rigid registration, a 9,994-frame `p1-p97` tdTomato segmentation reference, 45 proposals, fluorescence extraction, and stimulation-off-aligned dLight and tdTomato traces. The source TIFFs and 4.9 GB NWB file are not distributed. [`benchmarking/full_session_demo.py`](../benchmarking/full_session_demo.py) contains the session-specific pulse detection and plotting code.

## registration records

`registration_benchmark_comparison.png` is the public summary: rigid and piecewise held-out errors sit side by side. `rigid_registration_benchmark.mp4` shows raw, FibreSight, Suite2p and CaImAn in that order. `registration_benchmark_suite.mp4` retains the earlier multi-case review.

The remaining `registration_benchmark_*.png` files record accuracy, deformation, reference construction, resources, intensity sweeps, source groups and focal candidates. `benchmark_source_gallery.png` shows the ten public inputs. [BENCHMARK.md](../BENCHMARK.md) gives the method versions, reproduction commands, failures and claim boundaries; the figures are records of that benchmark, not six independent demonstrations.
