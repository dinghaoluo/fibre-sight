# demonstration images

The three unlabelled 256 × 256 demonstration arrays live in
`../benchmarking/sources/`:

- `lab-fibresight-demo-test.npy` comes from one session in the test split and is used by the main quick-demo command.
- `lab-fibresight-demo-train-01.npy` and `lab-fibresight-demo-train-02.npy` come from two sessions in the training split.

Their full source images and hand-labelled ROI dictionaries remain private. The crops are included so that `fibre-sight-predict` can be run from a fresh clone; performance across the complete test sessions is reported in the [model card](../MODEL_CARD.md), with the selection record kept in [METHODS.md](../METHODS.md).

Predictions and GUI edits made from these examples are saved under `workspace/output/`, leaving the shipped arrays unchanged.

The arrays are released under the repository's [MIT License](../LICENSE).
