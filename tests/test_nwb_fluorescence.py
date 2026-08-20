'''
Created on 19 August 2026

check ROI and annulus fluorescence extraction from NWB movies

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
import h5py
import numpy as np
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.base import Images
from pynwb.image import GrayscaleImage, ImageSeries

from support import add_source_to_path

add_source_to_path()

from fibre_sight.api import extract_fluorescence, load_fluorescence_run
from fibre_sight.nwb_segmentation import CONTROL_REFERENCE_PATH, _add_roi_run


#%% fixture
ROI_RUN = 'roi_source'
EMPTY_ROI_RUN = 'empty_roi_source'
ROI_DICT = {
    17: {'xpix': np.asarray([3]), 'ypix': np.asarray([3])},
    4: {'xpix': np.asarray([4]), 'ypix': np.asarray([3])},
    }


def _preprocessed_nwb(path, registration_models=None):
    height, width = 7, 8
    y, x = np.indices((height, width))
    base = 10 * y + x
    signal = np.stack([base, base + 100, base + 200]).astype(np.int16)
    control = signal + 1000

    nwbfile = NWBFile(
        session_description='fluorescence extraction test',
        identifier=path.stem,
        session_start_time=datetime(2026, 8, 19, tzinfo=timezone.utc),
        )
    module = nwbfile.create_processing_module(
        name='preprocessing',
        description='minimal registered recording',
        )
    paired_frames = TimeSeries(
        name='paired_frames',
        data=np.arange(3, dtype=np.int64),
        unit='frame',
        timestamps=np.asarray([0.0, 0.1, 0.2]),
        )
    module.add(paired_frames)
    for channel, movie in (('signal', signal), ('control', control)):
        module.add(ImageSeries(
            name=f'registered_{channel}',
            data=H5DataIO(
                movie.swapaxes(1, 2),
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
            timestamps=paired_frames,
            ))
    module.add(Images(
        name='registration_references',
        images=[GrayscaleImage(
            name='control_reference',
            data=control[0].astype(np.float32).T,
            description='test reference',
            )],
        ))

    metadata = DynamicTable(name='recording_metadata', description='test metadata')
    metadata.add_column(name='control_label', description='control label')
    metadata.add_column(name='pixel_size_um', description='pixel size')
    metadata.add_row(control_label='tdTomato', pixel_size_um=np.nan)
    module.add(metadata)

    signal_bounds = np.asarray([
        [0, height, 0, width],
        [3, 5, 3, 4],
        [0, height, 0, width],
        ])
    control_bounds = np.tile([0, height, 0, width], (3, 1))
    bounds = DynamicTable(
        name='registered_valid_bounds',
        description='half-open valid bounds in row and column order',
        id=np.arange(3),
        )
    for channel, channel_bounds in (
            ('signal', signal_bounds), ('control', control_bounds)):
        for coordinate_i, coordinate in enumerate(('y0', 'y1', 'x0', 'x1')):
            bounds.add_column(
                name=f'{channel}_{coordinate}',
                description=f'{channel} valid {coordinate}',
                data=channel_bounds[:, coordinate_i],
                )
    module.add(bounds)

    if registration_models is not None:
        piecewise_registration = DynamicTable(
            name='piecewise_registration',
            description='per-frame field acceptance and rigid fallback',
            id=np.arange(3),
            )
        piecewise_registration.add_column(
            name='model_used',
            description='piecewise_rigid or rigid',
            data=registration_models,
            )
        module.add(piecewise_registration)

    _add_roi_run(
        nwbfile,
        ROI_RUN,
        ROI_DICT,
        {
            'run_name': ROI_RUN,
            'run_type': 'proposed',
            'source_run': '',
            'reference_path': CONTROL_REFERENCE_PATH,
            'checkpoint_path': 'test_checkpoint.pt',
            'checkpoint_sha256': '0' * 64,
            'threshold': 0.25,
            'min_size': 1,
            'max_size': 10,
            'tta': False,
            'device': 'cpu',
            'created_at': datetime.now(timezone.utc).isoformat(),
            },
        probability=np.zeros((height, width), dtype=np.float32),
        )
    _add_roi_run(
        nwbfile,
        EMPTY_ROI_RUN,
        {},
        {
            'run_name': EMPTY_ROI_RUN,
            'run_type': 'proposed',
            'source_run': '',
            'reference_path': CONTROL_REFERENCE_PATH,
            'checkpoint_path': 'test_checkpoint.pt',
            'checkpoint_sha256': '0' * 64,
            'threshold': 0.25,
            'min_size': 1,
            'max_size': 10,
            'tta': False,
            'device': 'cpu',
            'created_at': datetime.now(timezone.utc).isoformat(),
            },
        probability=np.zeros((height, width), dtype=np.float32),
        )
    with NWBHDF5IO(path, 'w') as io:
        io.write(nwbfile)


#%% tests
class NWBFluorescenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix='fibre sight fluorescence ')
        self.root = Path(self.temp_dir.name)
        self.path = self.root / 'recording.nwb'
        _preprocessed_nwb(self.path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_exact_statistics_orientation_bounds_and_provenance(self):
        result = extract_fluorescence(
            self.path,
            'extraction',
            ROI_RUN,
            annulus_inner_px=0,
            annulus_outer_px=1,
            )
        loaded = load_fluorescence_run(self.path, 'extraction')

        expected = {
            'signal_roi_mean': [[33, 34], [133, np.nan], [233, 234]],
            'signal_roi_median': [[33, 34], [133, np.nan], [233, 234]],
            'signal_roi_iqr': [[0, 0], [0, np.nan], [0, 0]],
            'signal_roi_valid_fraction': [[1, 1], [1, 0], [1, 1]],
            'signal_annulus_mean': [
                [32 + 2 / 3, 34 + 1 / 3],
                [143, np.nan],
                [232 + 2 / 3, 234 + 1 / 3],
                ],
            'signal_annulus_median': [[32, 35], [143, np.nan], [232, 235]],
            'signal_annulus_iqr': [[10, 10], [0, np.nan], [10, 10]],
            'signal_annulus_valid_fraction': [[1, 1], [1 / 3, 0], [1, 1]],
            'control_roi_mean': [[1033, 1034], [1133, 1134], [1233, 1234]],
            'control_roi_median': [[1033, 1034], [1133, 1134], [1233, 1234]],
            'control_roi_iqr': [[0, 0], [0, 0], [0, 0]],
            'control_roi_valid_fraction': [[1, 1], [1, 1], [1, 1]],
            'control_annulus_mean': [
                [1032 + 2 / 3, 1034 + 1 / 3],
                [1132 + 2 / 3, 1134 + 1 / 3],
                [1232 + 2 / 3, 1234 + 1 / 3],
                ],
            'control_annulus_median': [
                [1032, 1035], [1132, 1135], [1232, 1235]],
            'control_annulus_iqr': [[10, 10], [10, 10], [10, 10]],
            'control_annulus_valid_fraction': [[1, 1], [1, 1], [1, 1]],
            }
        self.assertEqual(set(loaded['traces']), set(expected))
        for series_name, values in expected.items():
            with self.subTest(series_name=series_name):
                np.testing.assert_allclose(
                    loaded['traces'][series_name],
                    values,
                    rtol=1e-6,
                    equal_nan=True,
                    )

        np.testing.assert_array_equal(loaded['roi_ids'], [17, 4])
        self.assertEqual(loaded['roi_run'], ROI_RUN)
        self.assertEqual(loaded['provenance']['annulus_inner_px'], 0)
        self.assertEqual(loaded['provenance']['annulus_outer_px'], 1)
        self.assertEqual(result['frame_count'], 3)
        self.assertEqual(result['roi_count'], 2)
        self.assertEqual(result['validation_errors'], [])

        with NWBHDF5IO(self.path, 'r') as io:
            fluorescence = io.read().processing['fluorescence']['extraction']
            for series in fluorescence.roi_response_series.values():
                np.testing.assert_array_equal(series.timestamps, [0.0, 0.1, 0.2])
        with h5py.File(self.path, 'r') as file:
            data = file[
                'processing/fluorescence/extraction/signal_roi_mean/data']
            self.assertEqual(data.shape, (3, 2))
            self.assertEqual(data.dtype, np.dtype('float32'))
            self.assertEqual(data.chunks, (3, 2))
            self.assertEqual(data.compression, 'gzip')
            self.assertTrue(data.shuffle)
            self.assertTrue(data.fletcher32)

    def test_duplicate_extraction_name_is_rejected(self):
        extract_fluorescence(self.path, 'extraction', ROI_RUN)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            extract_fluorescence(self.path, 'extraction', ROI_RUN)

    def test_reserved_run_name_is_rejected_before_extraction(self):
        with patch('fibre_sight.fluorescence._extract_traces') as extract_traces:
            with self.assertRaisesRegex(ValueError, 'reserved'):
                extract_fluorescence(self.path, 'fluorescence_runs', ROI_RUN)

        extract_traces.assert_not_called()
        self.assertFalse(
            (self.root / 'recording.extracting.partial.nwb').exists())

    def test_piecewise_registration_is_rejected_before_extraction(self):
        piecewise_path = self.root / 'piecewise.nwb'
        _preprocessed_nwb(
            piecewise_path,
            registration_models=['rigid', 'piecewise_rigid', 'rigid'],
            )

        with patch('fibre_sight.fluorescence._extract_traces') as extract_traces:
            with self.assertRaisesRegex(ValueError, 'exact valid-pixel masks'):
                extract_fluorescence(piecewise_path, 'extraction', ROI_RUN)

        extract_traces.assert_not_called()
        self.assertFalse(
            (self.root / 'piecewise.extracting.partial.nwb').exists())

    def test_existing_partial_is_rejected_before_extraction(self):
        partial_path = self.root / 'recording.extracting.partial.nwb'
        partial_path.touch()

        with patch('fibre_sight.fluorescence._extract_traces') as extract_traces:
            with self.assertRaisesRegex(FileExistsError, 'partial output already exists'):
                extract_fluorescence(self.path, 'extraction', ROI_RUN)

        extract_traces.assert_not_called()

    def test_empty_roi_run_is_rejected_before_extraction(self):
        with patch('fibre_sight.fluorescence._extract_traces') as extract_traces:
            with self.assertRaisesRegex(ValueError, 'contains no ROIs'):
                extract_fluorescence(self.path, 'extraction', EMPTY_ROI_RUN)

        extract_traces.assert_not_called()
        self.assertFalse(
            (self.root / 'recording.extracting.partial.nwb').exists())

    @patch(
        'fibre_sight.fluorescence.validate',
        return_value=['forced validation failure'],
        )
    def test_validation_failure_leaves_original_bytes_unchanged(self, _validate):
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()

        with self.assertRaisesRegex(AssertionError, 'forced validation failure'):
            extract_fluorescence(self.path, 'extraction', ROI_RUN)

        after = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(after, before)
        self.assertTrue((self.root / 'recording.extracting.partial.nwb').exists())
        with self.assertRaises(KeyError):
            load_fluorescence_run(self.path, 'extraction')


if __name__ == '__main__':
    unittest.main()
