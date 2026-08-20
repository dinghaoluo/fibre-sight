'''
Created on 19 August 2026

check immutable segmentation and curation runs in NWB

@author: Dinghao Luo
'''

#%% imports
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTable
import numpy as np
from pynwb import NWBFile, NWBHDF5IO
from pynwb.base import Images
from pynwb.image import GrayscaleImage, ImageSeries

from support import add_source_to_path

add_source_to_path()

from fibre_sight.api import load_roi_run, save_curated_rois, segment_recording


#%% fixtures
def _preprocessed_nwb(path, reference):
    height, width = reference.shape
    nwbfile = NWBFile(
        session_description='segmentation adapter test',
        identifier=path.stem,
        session_start_time=datetime(2026, 8, 19, tzinfo=timezone.utc),
        )
    module = nwbfile.create_processing_module(
        name='preprocessing',
        description='minimal registered recording',
        )
    for channel_i, channel in enumerate(('signal', 'control')):
        movie = np.arange(3 * width * height, dtype=np.int16).reshape(
            3, width, height) + channel_i
        module.add(ImageSeries(
            name=f'registered_{channel}',
            data=H5DataIO(
                movie,
                chunks=(1, width, height),
                compression='gzip',
                compression_opts=1,
                shuffle=True,
                fletcher32=True,
                ),
            unit='counts',
            format='raw',
            dimension=(width, height),
            num_samples=np.uint64(3),
            rate=30.0,
            ))
    module.add(Images(
        name='registration_references',
        images=[GrayscaleImage(
            name='control_reference',
            data=np.asarray(reference, dtype=np.float32).T,
            description='test reference',
            )],
        ))
    metadata = DynamicTable(
        name='recording_metadata', description='test metadata')
    metadata.add_column(name='control_label', description='control label')
    metadata.add_column(name='pixel_size_um', description='pixel size')
    metadata.add_row(control_label='tdTomato', pixel_size_um=1.2)
    module.add(metadata)
    with NWBHDF5IO(path, 'w') as io:
        io.write(nwbfile)


class _Predictor:
    seen_reference = None

    def __init__(self, **kwargs):
        self.checkpoint_path = Path(__file__)
        self.threshold = 0.25
        self.min_size = 2
        self.max_size = 100
        self.tta = True
        self.device = 'cpu'

    def predict_image(self, image):
        type(self).seen_reference = image.copy()
        probability = np.asarray(image, dtype=np.float32) / np.max(image)
        roi_dict = {
            4: {
                'xpix': np.asarray([1, 5, 5]),
                'ypix': np.asarray([0, 2, 3]),
                },
            }
        return roi_dict, np.zeros(image.shape, dtype=np.int32), probability


class _EmptyPredictor(_Predictor):
    def predict_image(self, image):
        probability = np.zeros(image.shape, dtype=np.float32)
        return {}, np.zeros(image.shape, dtype=np.int32), probability


#%% tests
class NWBSegmentationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix='fibre sight segmentation ')
        self.root = Path(self.temp_dir.name)
        self.path = self.root / 'recording.nwb'
        self.reference = np.arange(24, dtype=np.float32).reshape(4, 6)
        _preprocessed_nwb(self.path, self.reference)

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch('fibre_sight.api.ROIPredictor', _Predictor)
    def test_non_square_reference_and_roi_coordinates_round_trip(self):
        result = segment_recording(self.path, 'proposal')
        loaded = load_roi_run(self.path, 'proposal')

        np.testing.assert_array_equal(_Predictor.seen_reference, self.reference)
        np.testing.assert_array_equal(loaded['reference'], self.reference)
        np.testing.assert_array_equal(loaded['probability'], self.reference / 23)
        np.testing.assert_array_equal(loaded['roi_dict'][4]['xpix'], [1, 5, 5])
        np.testing.assert_array_equal(loaded['roi_dict'][4]['ypix'], [0, 2, 3])
        self.assertEqual(result['roi_count'], 1)
        self.assertEqual(result['validation_errors'], [])

    @patch('fibre_sight.api.ROIPredictor', _Predictor)
    def test_proposed_and_curated_runs_are_immutable(self):
        segment_recording(self.path, 'proposal')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            segment_recording(self.path, 'proposal')

        curated = {
            10: {'xpix': np.asarray([1, 2]), 'ypix': np.asarray([1, 1])},
            11: {'xpix': np.asarray([2, 3]), 'ypix': np.asarray([1, 1])},
            }
        save_curated_rois(self.path, 'curated', curated, 'proposal')
        with self.assertRaisesRegex(ValueError, 'already exists'):
            save_curated_rois(self.path, 'curated', curated, 'proposal')

        proposal = load_roi_run(self.path, 'proposal')
        loaded = load_roi_run(self.path, 'curated')
        self.assertEqual(proposal['provenance']['run_type'], 'proposed')
        self.assertEqual(loaded['provenance']['run_type'], 'curated')
        self.assertEqual(loaded['provenance']['source_run'], 'proposal')
        self.assertIsNone(loaded['probability'])
        self.assertIn(2, loaded['roi_dict'][10]['xpix'])
        self.assertIn(2, loaded['roi_dict'][11]['xpix'])

    @patch('fibre_sight.api.ROIPredictor', _EmptyPredictor)
    def test_empty_proposal_is_stored_as_a_valid_run(self):
        result = segment_recording(self.path, 'empty_proposal')
        loaded = load_roi_run(self.path, 'empty_proposal')

        self.assertEqual(result['roi_count'], 0)
        self.assertEqual(result['validation_errors'], [])
        self.assertEqual(loaded['roi_dict'], {})
        np.testing.assert_array_equal(
            loaded['probability'],
            np.zeros(self.reference.shape, dtype=np.float32),
            )

    def test_existing_partial_is_rejected_before_inference(self):
        partial_path = self.root / 'recording.segmenting.partial.nwb'
        partial_path.touch()

        with patch('fibre_sight.api.ROIPredictor') as predictor:
            with self.assertRaisesRegex(FileExistsError, 'partial output already exists'):
                segment_recording(self.path, 'proposal')

        predictor.assert_not_called()

    @patch('fibre_sight.api.ROIPredictor', _Predictor)
    @patch(
        'fibre_sight.nwb_segmentation.validate',
        return_value=['forced validation failure'],
        )
    def test_validation_failure_leaves_original_bytes_unchanged(self, _validate):
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()

        with self.assertRaisesRegex(AssertionError, 'forced validation failure'):
            segment_recording(self.path, 'proposal')

        after = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(after, before)
        self.assertTrue((self.root / 'recording.segmenting.partial.nwb').exists())
        with self.assertRaises(KeyError):
            load_roi_run(self.path, 'proposal')


if __name__ == '__main__':
    unittest.main()
