'''
Created on 19 August 2026

check fluorescence extraction and dF/F calculation in NWB

@author: Dinghao Luo
'''

#%% imports
from datetime import datetime, timezone
import hashlib
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from hdmf.backends.hdf5.h5_utils import H5DataIO
from hdmf.common import DynamicTable
import numpy as np
from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.base import Images
from pynwb.image import GrayscaleImage, ImageSeries

from support import add_source_to_path

add_source_to_path()

from fibre_sight.api import (
    calculate_dff,
    extract_fluorescence,
    list_dff_runs,
    list_fluorescence_runs,
    load_dff_run,
    load_fluorescence_run,
    )
from fibre_sight.fluorescence import _roi_and_surround_coordinates
from fibre_sight.list_runs import (
    TABLE_COLUMNS,
    list_analysis_runs,
    main as list_runs_main,
    )
from fibre_sight.nwb_segmentation import CONTROL_REFERENCE_PATH, _add_roi_run


#%% fixture
ROI_RUN = 'roi_source'
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
    metadata.add_column(
        name='sampling_frequency_hz', description='paired observations per second')
    metadata.add_row(
        control_label='tdTomato',
        pixel_size_um=np.nan,
        sampling_frequency_hz=10.0,
        )
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

    quality_control = nwbfile.create_processing_module(
        name='quality_control',
        description='minimal registration quality control',
        )
    registration_qc = DynamicTable(
        name='registration_qc',
        description='one row per paired observation',
        id=np.arange(3),
        )
    registration_qc.add_column(
        name='analysis_valid',
        description='true only for accepted observations',
        data=[True, False, True],
        )
    quality_control.add(registration_qc)

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
            surround_method='fixed',
            surround_inner_px=0,
            surround_outer_px=1,
            )
        loaded = load_fluorescence_run(self.path, 'extraction')

        expected = {
            'signal_roi_mean': [[33, 34], [133, np.nan], [233, 234]],
            'signal_roi_median': [[33, 34], [133, np.nan], [233, 234]],
            'signal_roi_iqr': [[0, 0], [0, np.nan], [0, 0]],
            'signal_roi_valid_fraction': [[1, 1], [1, 0], [1, 1]],
            'signal_surround_mean': [
                [32 + 2 / 3, 34 + 1 / 3],
                [143, np.nan],
                [232 + 2 / 3, 234 + 1 / 3],
                ],
            'signal_surround_median': [[32, 35], [143, np.nan], [232, 235]],
            'signal_surround_iqr': [[10, 10], [0, np.nan], [10, 10]],
            'signal_surround_valid_fraction': [[1, 1], [1 / 3, 0], [1, 1]],
            'control_roi_mean': [[1033, 1034], [1133, 1134], [1233, 1234]],
            'control_roi_median': [[1033, 1034], [1133, 1134], [1233, 1234]],
            'control_roi_iqr': [[0, 0], [0, 0], [0, 0]],
            'control_roi_valid_fraction': [[1, 1], [1, 1], [1, 1]],
            'control_surround_mean': [
                [1032 + 2 / 3, 1034 + 1 / 3],
                [1132 + 2 / 3, 1134 + 1 / 3],
                [1232 + 2 / 3, 1234 + 1 / 3],
                ],
            'control_surround_median': [
                [1032, 1035], [1132, 1135], [1232, 1235]],
            'control_surround_iqr': [[10, 10], [10, 10], [10, 10]],
            'control_surround_valid_fraction': [[1, 1], [1, 1], [1, 1]],
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
        self.assertEqual(loaded['provenance']['surround_method'], 'fixed')
        self.assertEqual(loaded['provenance']['surround_inner_px'], 0)
        self.assertEqual(loaded['provenance']['surround_outer_px'], 1)
        self.assertEqual(result['frame_count'], 3)
        self.assertEqual(result['roi_count'], 2)
        self.assertEqual(result['validation_errors'], [])

        with NWBHDF5IO(self.path, 'r') as io:
            fluorescence = io.read().processing['fluorescence']['extraction']
            for series in fluorescence.roi_response_series.values():
                np.testing.assert_array_equal(series.timestamps, [0.0, 0.1, 0.2])

    def test_adaptive_and_fixed_surround_coordinates(self):
        roi_dict = {
            1: {'xpix': np.asarray([3]), 'ypix': np.asarray([3])},
            2: {'xpix': np.asarray([4]), 'ypix': np.asarray([3])},
            }
        fixed = _roi_and_surround_coordinates(
            roi_dict, (7, 8), 'fixed', 0, 1, 350)
        adaptive = _roi_and_surround_coordinates(
            roi_dict, (7, 8), 'adaptive', 1, 8, 8)

        fixed_pixels = set(zip(*fixed['surround'][0]))
        self.assertEqual(fixed_pixels, {(2, 3), (3, 2), (4, 3)})
        adaptive_pixels = set(zip(*adaptive['surround'][0]))
        expected_adaptive_pixels = {
            (y, x)
            for y in range(7)
            for x in range(8)
            if 1 < abs(y - 3) + abs(x - 3) <= 6
            }
        self.assertEqual(adaptive_pixels, expected_adaptive_pixels)

    def test_duplicate_extraction_name_is_rejected(self):
        extract_fluorescence(self.path, 'extraction', ROI_RUN)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            extract_fluorescence(self.path, 'extraction', ROI_RUN)

    def test_piecewise_registration_is_rejected(self):
        piecewise_path = self.root / 'piecewise.nwb'
        _preprocessed_nwb(
            piecewise_path,
            registration_models=['rigid', 'piecewise_rigid', 'rigid'],
            )

        with self.assertRaisesRegex(ValueError, 'exact valid-pixel masks'):
            extract_fluorescence(piecewise_path, 'extraction', ROI_RUN)

    def test_dff_uses_raw_surround_subtraction_and_excludes_rejected_frames(self):
        extract_fluorescence(
            self.path,
            'extraction',
            ROI_RUN,
            surround_method='fixed',
            surround_inner_px=0,
            surround_outer_px=1,
            )
        result = calculate_dff(
            self.path,
            'derived',
            'extraction',
            baseline_percentile=50,
            baseline_window_s=0.5,
            surround_coefficient=0.7,
            control_correction='subtract_dff',
            )
        extracted = load_fluorescence_run(self.path, 'extraction')['traces']
        loaded = load_dff_run(self.path, 'derived')

        def expected_dff(raw_fluorescence):
            raw_fluorescence = np.asarray(raw_fluorescence, dtype=float).copy()
            raw_fluorescence[1] = np.nan
            padded = np.pad(raw_fluorescence, ((2, 2), (0, 0)), mode='reflect')
            baseline = np.asarray([
                np.nanpercentile(padded[i:i + 5], 50, axis=0)
                for i in range(3)
                ])
            return (raw_fluorescence - baseline) / baseline

        expected = {}
        for channel in ('signal', 'control'):
            roi = extracted[f'{channel}_roi_mean']
            surround = extracted[f'{channel}_surround_mean']
            expected[f'{channel}_roi_dff'] = expected_dff(roi)
            expected[f'{channel}_surround_dff'] = expected_dff(surround)
            expected[f'{channel}_surround_corrected_dff'] = expected_dff(
                roi - 0.7 * surround)
        expected['signal_control_corrected_dff'] = (
            expected['signal_surround_corrected_dff']
            - expected['control_surround_corrected_dff']
            )

        self.assertEqual(set(loaded['traces']), set(expected))
        for series_name, values in expected.items():
            np.testing.assert_allclose(
                loaded['traces'][series_name],
                values,
                rtol=1e-6,
                equal_nan=True,
                )
            self.assertTrue(np.isnan(loaded['traces'][series_name][1]).all())
        dff_space_subtraction = (
            expected['signal_roi_dff']
            - 0.7 * expected['signal_surround_dff']
            )
        self.assertFalse(np.allclose(
            loaded['traces']['signal_surround_corrected_dff'],
            dff_space_subtraction,
            equal_nan=True,
            ))
        self.assertEqual(loaded['provenance']['baseline_window_frames'], 5)
        self.assertEqual(loaded['provenance']['control_correction'], 'subtract_dff')
        self.assertEqual(result['validation_errors'], [])
        with self.assertRaisesRegex(ValueError, 'already exists'):
            calculate_dff(self.path, 'derived', 'extraction')

    def test_analysis_runs_are_composed_and_listed(self):
        extract_fluorescence(self.path, 'extraction', ROI_RUN)
        calculate_dff(
            self.path,
            'derived',
            'extraction',
            baseline_window_s=0.5,
            )

        self.assertEqual(
            [run['run_name'] for run in list_fluorescence_runs(self.path)],
            ['extraction'],
            )
        self.assertEqual(
            [run['run_name'] for run in list_dff_runs(self.path)],
            ['derived'],
            )
        runs = list_analysis_runs(self.path)
        self.assertEqual(
            [(run['kind'], run['run_name']) for run in runs],
            [('roi', ROI_RUN), ('fluorescence', 'extraction'), ('dff', 'derived')],
            )
        self.assertEqual(
            [run['run_name'] for run in list_analysis_runs(
                self.path, kind='roi', run_type='proposed')],
            [ROI_RUN],
            )

        output = StringIO()
        with patch(
                'sys.argv',
                ['fibre-sight-list-runs', str(self.path), '--kind', 'dff'],
                ), redirect_stdout(output):
            list_runs_main()
        lines = output.getvalue().splitlines()
        self.assertEqual(lines[0], '\t'.join(TABLE_COLUMNS))
        self.assertIn('dff\tderived', lines[1])

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

    @patch('fibre_sight.dff.validate', return_value=['forced validation failure'])
    def test_dff_validation_failure_leaves_original_bytes_unchanged(
            self, _validate):
        extract_fluorescence(self.path, 'extraction', ROI_RUN)
        before = hashlib.sha256(self.path.read_bytes()).hexdigest()

        with self.assertRaisesRegex(AssertionError, 'forced validation failure'):
            calculate_dff(
                self.path,
                'derived',
                'extraction',
                baseline_window_s=0.5,
                )

        after = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.assertEqual(after, before)
        self.assertTrue(
            (self.root / 'recording.calculating_dff.partial.nwb').exists())
        with self.assertRaises(KeyError):
            load_dff_run(self.path, 'derived')


if __name__ == '__main__':
    unittest.main()
