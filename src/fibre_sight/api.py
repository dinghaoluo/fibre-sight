'''
Created on 29 April 2026

Modified on 3 June 2026
Modified on 24 July 2026 to load the bundled channel-2 checkpoint
Modified on 14 August 2026
Modified on 19 August 2026

model prediction, named NWB ROI runs, and fluorescence extraction

@author: Dinghao Luo
'''

#%% imports
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from pynwb import NWBHDF5IO

from ._device import get_device
from ._repo import PACKAGE_ROOT
from .fluorescence import extract_fluorescence, load_fluorescence_run
from .nwb_segmentation import (
    CONTROL_REFERENCE_PATH,
    _append_roi_run_transactionally,
    _check_new_run,
    _checkpoint_sha256,
    _read_control_reference,
    _segmentation_partial_path,
    list_roi_runs,
    load_roi_run,
    save_curated_rois,
    )
from .postprocess import probability_to_roi_dict
from .predict_rois import load_model, predict_probability


#%% defaults
BUNDLED_CHECKPOINT = PACKAGE_ROOT / 'models' / 'fibre_sight_ch2_v1.pt'
BUNDLED_THRESHOLD = 0.25
BUNDLED_MIN_SIZE = 45


#%% prediction
class ROIPredictor:
    def __init__(
            self,
            checkpoint_path=None,
            device='auto',
            threshold=None,
            min_size=None,
            tta=None,
            ):
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else BUNDLED_CHECKPOINT
            )
        self.device = get_device(device)
        self.threshold = threshold
        self.min_size = min_size
        self.tta = tta
        self.max_size = None
        self.model = None
        self.checkpoint = None

    def load(self):
        self.model, self.checkpoint = load_model(self.checkpoint_path, self.device)
        postprocess = self.checkpoint['postprocess_config']
        if self.threshold is None:
            self.threshold = postprocess['threshold']
        if self.min_size is None:
            self.min_size = postprocess['min_size']
        if self.tta is None:
            self.tta = postprocess['tta']
        self.max_size = postprocess['max_size']

    def predict_image(self, image):
        if self.model is None:
            self.load()

        probability = predict_probability(
            image,
            self.model,
            self.device,
            normalise_percentiles=self.checkpoint['data_config']['normalise_percentiles'],
            tta=self.tta,
            )
        roi_dict, labelled = probability_to_roi_dict(
            probability,
            threshold=self.threshold,
            min_size=self.min_size,
            max_size=self.max_size,
            )
        return roi_dict, labelled, probability


#%% NWB segmentation
def segment_recording(
        nwb_path,
        run_name,
        *,
        checkpoint_path=None,
        threshold=None,
        min_size=None,
        tta=None,
        device='auto',
        ):
    nwb_path = Path(nwb_path)
    partial_path = _segmentation_partial_path(nwb_path)
    if partial_path.exists():
        raise FileExistsError(f'partial output already exists: {partial_path}')

    with NWBHDF5IO(nwb_path, 'r') as io:
        nwbfile = io.read()
        _check_new_run(nwbfile, run_name)
        reference = _read_control_reference(nwbfile)

    inference_start = perf_counter()
    predictor = ROIPredictor(
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        min_size=min_size,
        tta=tta,
        device=device,
        )
    roi_dict, _, probability = predictor.predict_image(reference)
    inference_time_s = perf_counter() - inference_start
    checkpoint_path = predictor.checkpoint_path.resolve()
    run_metadata = {
        'run_name': run_name,
        'run_type': 'proposed',
        'source_run': '',
        'reference_path': CONTROL_REFERENCE_PATH,
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_sha256': _checkpoint_sha256(checkpoint_path),
        'threshold': float(predictor.threshold),
        'min_size': int(predictor.min_size),
        'max_size': -1 if predictor.max_size is None else int(predictor.max_size),
        'tta': bool(predictor.tta),
        'device': str(predictor.device),
        'created_at': datetime.now(timezone.utc).isoformat(),
        }
    result = _append_roi_run_transactionally(
        nwb_path, run_name, roi_dict, run_metadata, probability)
    result['inference_time_s'] = inference_time_s
    return result
