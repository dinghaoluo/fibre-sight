# demonstration images

The three `ref_mat_ch2.npy` arrays in this folder are unlabelled 256 × 256 crops from separate channel-2 reference images:

- `demo_test_ref_mat_ch2.npy` comes from one session in the test split and is used by the main quick-demo command.
- `demo_train_01_ref_mat_ch2.npy` and `demo_train_02_ref_mat_ch2.npy` come from two sessions in the training split.

Their full source images and hand-labelled ROI dictionaries remain private. The crops are included so that `fibre-sight-predict` can be run from a fresh clone; performance across the complete test sessions is reported in the [model card](../MODEL_CARD.md), with the selection record kept in [METHODS.md](../METHODS.md).

Predictions and GUI edits made from these examples are saved under `workspace/output/`, leaving the shipped arrays unchanged.

The arrays are released under the repository's [MIT License](../LICENSE).
