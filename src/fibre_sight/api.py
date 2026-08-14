'''
Created on 29 April 2026

Modified on 3 June 2026
Modified on 24 July 2026 to load the bundled channel-2 checkpoint
Modified on 14 August 2026

model loading and prediction used by the GUI

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path

from ._device import get_device
from ._repo import PACKAGE_ROOT
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
