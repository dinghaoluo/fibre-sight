'''
Created on 14 August 2026
Modified on 18 August 2026
Modified on 19 August 2026

check TIFF reading, registration, and saved QC

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import ndimage as ndi
import tifffile

from support import add_source_to_path

add_source_to_path()

from fibre_sight.api import add_segmentation_references, preprocess_recording
from fibre_sight.preprocessing import (
    add_quality_control_to_nwb,
    assess_tile_field,
    compare_registration_models,
    estimate_channel_offset,
    estimate_shift,
    estimate_tile_shifts,
    estimate_tile_shifts_coarse_to_fine,
    evaluate_tile_field,
    focal_loss_episodes,
    fit_tile_field,
    index_tiffs,
    make_local_references,
    make_reference,
    measure_quality,
    read_tiffs,
    refine_tile_field,
    register_pair,
    register_pair_piecewise,
    rolling_axial_similarity,
    select_registration_model,
    warp_frame,
    )


#%% helpers
def _write_pages(path, pages):
    with tifffile.TiffWriter(path) as writer:
        for page in pages:
            writer.write(
                page,
                metadata=None,
                contiguous=False,
                )


def _image(shape=(96, 112)):
    rng = np.random.default_rng(42)
    return ndi.gaussian_filter(rng.normal(size=shape), 2).astype(np.float32)


#%% tests
class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix='fibre sight TIFF ')
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_multiplexed_index_follows_chunk_and_channel_order(self):
        early_path = self.root / 'recording_2.tif'
        late_path = self.root / 'recording_10.tif'
        early_signal = np.full((3, 5), 10, dtype=np.int16)
        early_control = np.full((3, 5), 20, dtype=np.int16)
        late_signal = np.full((3, 5), 11, dtype=np.int16)
        late_control = np.full((3, 5), 21, dtype=np.int16)
        _write_pages(
            early_path,
            [early_control, early_signal],
            )
        _write_pages(
            late_path,
            [late_control, late_signal],
            )

        recording = index_tiffs(
            [late_path, early_path],
            signal_channel=2,
            control_channel=1,
            sampling_frequency_hz=30,
            signal_label='dLight',
            control_label='tdTomato',
            )

        self.assertEqual(recording['sampling_frequency_hz'], 30)
        self.assertEqual(recording['signal_label'], 'dLight')
        self.assertEqual(recording['control_label'], 'tdTomato')
        self.assertEqual(recording['shape'], (3, 5))
        self.assertEqual(recording['dtype'], np.dtype('int16'))
        self.assertEqual(recording['n_frames'], 2)
        self.assertEqual(recording['frames'][0]['signal_page'], 1)
        self.assertEqual(recording['frames'][0]['control_page'], 0)
        self.assertEqual(recording['frames'][1]['signal_tiff'], late_path)

        frames = list(read_tiffs(recording))
        np.testing.assert_array_equal(frames[0]['signal'], early_signal)
        np.testing.assert_array_equal(frames[0]['control'], early_control)
        np.testing.assert_array_equal(frames[1]['signal'], late_signal)
        np.testing.assert_array_equal(frames[1]['control'], late_control)

    def test_separate_tiffs_share_one_frame_index(self):
        signal_path = self.root / 'signal_2.tif'
        control_path = self.root / 'control_2.tif'
        signal = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        control = signal + 100
        _write_pages(signal_path, signal)
        _write_pages(control_path, control)

        recording = index_tiffs(
            [signal_path],
            control_tiffs=[control_path],
            multiplexed=False,
            signal_channel=1,
            control_channel=2,
            sampling_frequency_hz=20,
            )

        self.assertEqual(
            [frame['frame'] for frame in recording['frames']], [0, 1])
        self.assertEqual(recording['frames'][1]['signal_page'], 1)
        self.assertEqual(recording['frames'][1]['control_page'], 1)
        frames = list(read_tiffs(recording))
        np.testing.assert_array_equal(frames[1]['signal'], signal[1])
        np.testing.assert_array_equal(frames[1]['control'], control[1])

    def test_incomplete_or_inconsistent_pairs_raise(self):
        path = self.root / 'recording.tif'
        cases = [
            ('incomplete', [
                np.zeros((2, 3), dtype=np.int16),
                np.zeros((2, 3), dtype=np.int16),
                np.zeros((2, 3), dtype=np.int16),
                ]),
            ('shape', [
                np.zeros((2, 3), dtype=np.int16),
                np.zeros((2, 3), dtype=np.int16),
                np.zeros((3, 2), dtype=np.int16),
                np.zeros((3, 2), dtype=np.int16),
                ]),
            ('dtype', [
                np.zeros((2, 3), dtype=np.int16),
                np.zeros((2, 3), dtype=np.int16),
                np.zeros((2, 3), dtype=np.uint16),
                np.zeros((2, 3), dtype=np.uint16),
                ]),
            ]

        for name, pages in cases:
            with self.subTest(name=name):
                _write_pages(path, pages)
                with self.assertRaises(ValueError):
                    index_tiffs(
                        [path], signal_channel=1, control_channel=2,
                        sampling_frequency_hz=30,
                        )

    def test_preprocessing_rejects_channel_and_output_conflicts(self):
        signal_path = self.root / 'signal.tif'
        control_path = self.root / 'control.tif'
        output_path = self.root / 'recording.nwb'
        _write_pages(signal_path, np.zeros((2, 8, 8), dtype=np.int16))
        _write_pages(control_path, np.zeros((1, 8, 8), dtype=np.int16))

        with self.assertRaisesRegex(ValueError, 'different frame counts'):
            preprocess_recording(
                [signal_path], output_path, 1, 2, False, 30,
                control_tiff_paths=[control_path], registration_model='rigid')
        with self.assertRaisesRegex(ValueError, 'different one-based numbers'):
            preprocess_recording(
                [signal_path], output_path, 1, 1, True, 30,
                registration_model='rigid')

        output_path.touch()
        with self.assertRaises(FileExistsError):
            preprocess_recording(
                [signal_path], output_path, 1, 2, True, 30,
                registration_model='rigid')

    def test_preprocessed_nwb_keeps_non_square_image_orientation(self):
        from datetime import datetime, timezone
        from pynwb import NWBHDF5IO

        image = np.rint((_image((64, 96)) + 2) * 1000).astype(np.int16)
        image[16, 20], image[16, 75] = 101, 202
        image[48, 20], image[48, 75] = 303, 404
        pages = np.empty((100, *image.shape), dtype=np.int16)
        pages[0::2] = image
        pages[1::2] = image
        pages[50] = image * 3
        tiff_path = self.root / 'non_square.tif'
        output_path = self.root / 'non_square.nwb'
        _write_pages(tiff_path, pages)

        preprocess_recording(
            [tiff_path], output_path, 1, 2, True, 30,
            registration_model='piecewise',
            session_start_time=datetime(2026, 8, 19, tzinfo=timezone.utc),
            )

        with NWBHDF5IO(output_path, 'r') as io:
            nwbfile = io.read()
            module = nwbfile.processing['preprocessing']
            stored = np.asarray(module['registered_control'].data[0])
            reference = np.asarray(
                module['registration_references']['control_reference'].data)
            mean_reference = np.asarray(
                module['segmentation_references']['mean_control_reference'].data)
            segmentation_reference = np.asarray(
                module['segmentation_references']['segmentation_reference'].data)
            segmentation_frames = np.asarray(
                module['segmentation_reference_frames']['frame_index'], dtype=int)
            reference_metadata = {
                name: module['segmentation_reference_metadata'][name][0]
                for name in module['segmentation_reference_metadata'].colnames
                }
            analysis_valid = np.asarray(
                nwbfile.processing['quality_control'][
                    'registration_qc']['analysis_valid'], dtype=bool)
            reason_code = np.asarray([
                value.decode() if isinstance(value, bytes) else str(value)
                for value in nwbfile.processing['quality_control'][
                    'registration_qc']['reason_code'][:]
                ])
            movie = np.asarray(module['registered_control'].data[:]).transpose(0, 2, 1)
            bounds = module['registered_valid_bounds']
            expected_total = np.zeros(image.shape, dtype=np.float64)
            expected_count = np.zeros(image.shape, dtype=np.uint32)
            for frame_i in segmentation_frames:
                y0 = int(bounds['control_y0'][frame_i])
                y1 = int(bounds['control_y1'][frame_i])
                x0 = int(bounds['control_x0'][frame_i])
                x1 = int(bounds['control_x1'][frame_i])
                expected_total[y0:y1, x0:x1] += movie[
                    frame_i, y0:y1, x0:x1]
                expected_count[y0:y1, x0:x1] += 1
            shift_y, shift_x = module['rigid_translation'].data[0]
            source_hash = module['source_tiffs']['sha256'][0]
            n_grid_points = len(module['piecewise_spline_grid'])
            coefficient_shape = module['piecewise_spline_coefficients'].data.shape

        self.assertEqual(stored.shape, (96, 64))
        self.assertEqual(reference.shape, (96, 64))
        self.assertEqual(mean_reference.shape, (96, 64))
        self.assertEqual(segmentation_reference.shape, (96, 64))
        np.testing.assert_array_equal(
            segmentation_frames, np.flatnonzero(analysis_valid))
        self.assertEqual(reason_code[25], 'photometric_artifact')
        self.assertEqual(reason_code[26], 'photometric_artifact')
        expected_mean = np.divide(
            expected_total,
            expected_count,
            out=np.zeros_like(expected_total),
            where=expected_count > 0,
            ).astype(np.float32)
        low, high = np.percentile(expected_mean, [1, 97])
        expected_segmentation = (
            np.clip((expected_mean - low) / (high - low), 0, 1) * 255
            ).astype(np.uint8)
        np.testing.assert_array_equal(mean_reference.T, expected_mean)
        np.testing.assert_array_equal(
            segmentation_reference.T, expected_segmentation)
        self.assertEqual(segmentation_reference.dtype, np.uint8)
        self.assertEqual(reference_metadata['frame_count'], len(segmentation_frames))
        self.assertEqual(reference_metadata['low_percentile'], 1)
        self.assertEqual(reference_metadata['high_percentile'], 97)
        self.assertEqual(reference_metadata['low_value'], low)
        self.assertEqual(reference_metadata['high_value'], high)
        self.assertEqual(len(source_hash), 64)
        self.assertEqual(n_grid_points, coefficient_shape[2] * coefficient_shape[3])
        expected, _ = warp_frame(image, shift_y, shift_x)
        expected = np.nan_to_num(np.rint(expected), nan=0).astype(np.int16)
        np.testing.assert_array_equal(stored.T, expected)
        with self.assertRaisesRegex(ValueError, 'already exists'):
            add_segmentation_references(output_path)

    def test_rigid_shift_has_subpixel_accuracy_without_wrapped_edges(self):
        reference = _image()
        observed_shift = np.asarray([2.4, -3.7])
        frame = ndi.shift(reference, observed_shift, order=1, mode='constant', cval=0)

        estimate = estimate_shift(reference, frame, check_tiles=False)
        correction = np.asarray([estimate['shift_y'], estimate['shift_x']])
        np.testing.assert_allclose(correction, -observed_shift, atol=0.15)

        registered, bounds = warp_frame(frame, *correction)
        expected_valid = np.zeros(reference.shape, dtype=bool)
        y0, y1, x0, x1 = bounds
        expected_valid[y0:y1, x0:x1] = True
        np.testing.assert_array_equal(np.isfinite(registered), expected_valid)
        self.assertGreater(np.corrcoef(reference[expected_valid], registered[expected_valid])[0, 1], 0.99)

    def test_tile_evidence_keeps_rigid_residual_near_zero(self):
        reference = _image()
        observed_shift = np.asarray([2.4, -3.7])
        frame = ndi.shift(reference, observed_shift, order=1, mode='constant', cval=0)
        rigid = estimate_shift(reference, frame, check_tiles=False)

        result = estimate_tile_shifts(
            reference,
            frame,
            (rigid['shift_y'], rigid['shift_x']),
            tile_size=32,
            stride=32,
            )
        residual = np.hypot(result['residual_y'], result['residual_x'])

        self.assertLess(np.median(residual), 0.25)
        self.assertLess(np.percentile(residual, 95), 0.5)

    def test_coarse_to_fine_tiles_recover_smooth_local_movement(self):
        reference = _image((256, 256))
        y, x = np.indices(reference.shape, dtype=np.float32)
        observed_y = 2 + 1.3 * np.sin(2 * np.pi * x / reference.shape[1])
        observed_x = -2.5 + np.sin(2 * np.pi * y / reference.shape[0])
        frame = ndi.map_coordinates(
            reference,
            [y - observed_y, x - observed_x],
            order=1,
            mode='constant',
            ).astype(np.float32)
        rigid = estimate_shift(reference, frame, check_tiles=False)

        result = estimate_tile_shifts_coarse_to_fine(
            reference,
            frame,
            (rigid['shift_y'], rigid['shift_x']),
            )
        expected_y = (
            -2
            - 1.3 * np.sin(2 * np.pi * result['tile_x'] / reference.shape[1])
            - rigid['shift_y']
            )
        expected_x = (
            2.5
            - np.sin(2 * np.pi * result['tile_y'] / reference.shape[0])
            - rigid['shift_x']
            )
        error = np.hypot(
            result['residual_y'] - expected_y,
            result['residual_x'] - expected_x,
            )

        self.assertLess(np.percentile(error, 95), 0.3)

    def test_local_field_refinement_rejects_a_false_peak(self):
        reference = _image((256, 256))
        y, x = np.indices(reference.shape, dtype=np.float32)
        observed_y = 1.2 * np.sin(2 * np.pi * x / reference.shape[1])
        observed_x = np.sin(2 * np.pi * y / reference.shape[0])
        frame = ndi.map_coordinates(
            reference,
            [y - observed_y, x - observed_x],
            order=1,
            mode='constant',
            ).astype(np.float32)
        rigid = estimate_shift(reference, frame, check_tiles=False)
        tiles = estimate_tile_shifts(
            reference,
            frame,
            (rigid['shift_y'], rigid['shift_x']),
            )
        bad_tile = len(tiles['tile_y']) // 2
        tiles['residual_y'][bad_tile] += 5
        tiles['residual_x'][bad_tile] -= 5

        field = refine_tile_field(tiles, reference.shape, 64)
        predicted_y, predicted_x = evaluate_tile_field(
            field, tiles['tile_y'], tiles['tile_x'])
        expected_y = (
            -1.2 * np.sin(2 * np.pi * tiles['tile_x'] / reference.shape[1])
            - rigid['shift_y']
            )
        expected_x = (
            -np.sin(2 * np.pi * tiles['tile_y'] / reference.shape[0])
            - rigid['shift_x']
            )
        error = np.hypot(predicted_y - expected_y, predicted_x - expected_x)

        self.assertFalse(field['accepted'][bad_tile])
        self.assertLess(np.percentile(error, 95), 0.3)

        registered = register_pair_piecewise(
            frame,
            frame,
            rigid['shift_y'],
            rigid['shift_x'],
            field,
            )
        valid = registered['control_valid']
        np.testing.assert_array_equal(
            registered['signal_valid'], registered['control_valid'])
        np.testing.assert_array_equal(
            np.isfinite(registered['control']), registered['control_valid'])
        self.assertGreater(np.corrcoef(
            reference[valid], registered['control'][valid])[0, 1], 0.98)

        focal = register_pair_piecewise(
            frame,
            frame,
            rigid['shift_y'],
            rigid['shift_x'],
            field,
            focal_loss=True,
            )
        self.assertEqual(focal['fallback_reason'], 'focal_loss')
        oversized = {**field, 'global_shift_y': np.float32(4)}
        oversized = assess_tile_field(oversized, reference.shape)
        self.assertEqual(oversized['fallback_reason'], 'field_overshoot')

    def test_automatic_registration_requires_every_piecewise_gain(self):
        rigid = {
            'gradient_ncc': 0.80,
            'residual_p95_px': 0.60,
            'valid_fraction': 0.96,
            'cross_channel_residual_px': 0.10,
            }
        piecewise = {
            'gradient_ncc': 0.82,
            'residual_p95_px': 0.40,
            'valid_fraction': 0.95,
            'cross_channel_residual_px': 0.12,
            'accepted_or_fallback_fraction': 0.98,
            }
        selected = select_registration_model(rigid, piecewise)
        self.assertEqual(selected['selected_model'], 'piecewise_rigid')

        piecewise['gradient_ncc'] = 0.805
        selected = select_registration_model(rigid, piecewise)
        self.assertEqual(selected['selected_model'], 'rigid')
        self.assertFalse(selected['passed']['gradient_ncc'])

    def test_automatic_registration_uses_held_out_tile_residuals(self):
        reference = _image((256, 256))
        y, x = np.indices(reference.shape, dtype=np.float32)
        observed_y = 1.5 * np.sin(2 * np.pi * x / reference.shape[1])
        observed_x = 1.2 * np.sin(2 * np.pi * y / reference.shape[0])
        frame = ndi.map_coordinates(
            reference,
            [y - observed_y, x - observed_x],
            order=1,
            mode='constant',
            ).astype(np.float32)
        rigid = estimate_shift(reference, frame, check_tiles=False)
        tiles = estimate_tile_shifts(
            reference,
            frame,
            (rigid['shift_y'], rigid['shift_x']),
            tile_size=64,
            )
        field = fit_tile_field(
            tiles,
            reference.shape,
            64,
            spatial_penalty=10,
            magnitude_penalty=1,
            )

        comparison = compare_registration_models(
            reference,
            [frame] * 8,
            [frame] * 8,
            [rigid] * 8,
            [tiles] * 8,
            [field] * 8,
            )

        self.assertEqual(comparison['selected_model'], 'piecewise_rigid')
        self.assertGreater(comparison['comparison']['residual_p95_gain_px'], 1)
        self.assertGreater(comparison['comparison']['gradient_ncc_gain'], 0.1)

        focal = compare_registration_models(
            reference,
            [frame] * 8,
            [frame] * 8,
            [rigid] * 8,
            [tiles] * 8,
            [field] * 8,
            focal_loss=np.ones(8, dtype=bool),
            )
        self.assertEqual(focal['selected_model'], 'rigid')
        self.assertEqual(focal['piecewise']['residual_p95_px'],
                         focal['rigid']['residual_p95_px'])

        empty_tiles = {
            **tiles,
            'accepted': np.zeros(len(tiles['tile_y']), dtype=bool),
            'precision': np.zeros((len(tiles['tile_y']), 2, 2), dtype=np.float32),
            }
        empty_field = {
            **field,
            'accepted': np.zeros(len(field['accepted']), dtype=bool),
            }
        no_local_evidence = compare_registration_models(
            reference,
            [frame] * 4,
            [frame] * 4,
            [rigid] * 4,
            [empty_tiles] * 4,
            [empty_field] * 4,
            )
        self.assertEqual(no_local_evidence['selected_model'], 'rigid')
        self.assertTrue(np.isnan(no_local_evidence['rigid']['residual_p95_px']))

    def test_two_pass_reference_recovers_the_unmoved_image(self):
        rng = np.random.default_rng(42)
        image = _image((64, 80))
        image[:, :2] += 5
        other_plane = np.flip(image, axis=0)
        frames = []
        for frame_i in range(80):
            shift = rng.uniform(-4, 4, 2)
            plane = other_plane if 28 <= frame_i < 30 else image
            frame = ndi.shift(plane, shift, order=1, mode='constant', cval=0)
            frames.append((1 + rng.uniform(-0.15, 0.15)) * frame + rng.normal(0, 0.03, image.shape))
        frames = np.asarray(frames, dtype=np.float32)

        result = make_reference(
            frames,
            max_frames=80,
            min_frames=40,
            min_peak_ratio=1.02,
            max_tile_disagreement=1.5,
            )
        unregistered = frames.mean(axis=0)
        registered_shift = estimate_shift(image, result['image'], check_tiles=False)
        mean_shift = estimate_shift(image, unregistered, check_tiles=False)
        registered, _ = warp_frame(
            result['image'], registered_shift['shift_y'], registered_shift['shift_x'])
        unregistered, _ = warp_frame(unregistered, mean_shift['shift_y'], mean_shift['shift_x'])
        interior = np.zeros(image.shape, dtype=bool)
        interior[8:-8, 8:-8] = True
        registered_valid = np.isfinite(registered) & interior
        mean_valid = np.isfinite(unregistered) & interior
        registered_ncc = np.corrcoef(image[registered_valid], registered[registered_valid])[0, 1]
        mean_ncc = np.corrcoef(image[mean_valid], unregistered[mean_valid])[0, 1]
        self.assertEqual(result['gradient_information'].shape, (80,))
        self.assertEqual(result['accepted'].sum(), 40)
        self.assertFalse(result['accepted'][28:30].any())
        self.assertGreater(registered_ncc, mean_ncc + 0.05)
        self.assertLess(np.max(np.abs(result['image'][8:-8, -4:])), 2)

    def test_signal_and_control_share_movement_after_channel_alignment(self):
        image = _image()
        channel_displacement = np.asarray([1.3, -2.2])
        control_references = np.repeat(image[None], 8, axis=0)
        signal_references = np.repeat(
            ndi.shift(image, channel_displacement, order=1, mode='constant', cval=0)[None],
            8,
            axis=0,
            )
        offset = estimate_channel_offset(signal_references, control_references)
        np.testing.assert_allclose(
            [offset['shift_y'], offset['shift_x']], -channel_displacement, atol=0.15)
        self.assertTrue(offset['accepted'])

        movement = np.asarray([2.4, -1.7])
        control = ndi.shift(image, movement, order=1, mode='constant', cval=0)
        signal = ndi.shift(
            image, movement + channel_displacement, order=1, mode='constant', cval=0)
        estimate = estimate_shift(image, control, check_tiles=False)
        result = register_pair(
            signal,
            control,
            estimate['shift_y'],
            estimate['shift_x'],
            signal_to_control_offset=(offset['shift_y'], offset['shift_x']),
            )
        valid = np.isfinite(result['signal']) & np.isfinite(result['control'])
        self.assertGreater(
            np.corrcoef(result['signal'][valid], result['control'][valid])[0, 1], 0.99)

        signal_reference = ndi.shift(
            image, channel_displacement, order=1, mode='constant', cval=0)
        signal_estimate = estimate_shift(signal_reference, signal, check_tiles=False)
        signal_led = register_pair(
            signal,
            control,
            signal_estimate['shift_y'],
            signal_estimate['shift_x'],
            signal_to_control_offset=(offset['shift_y'], offset['shift_x']),
            registration_channel='signal',
            )
        valid = np.isfinite(signal_led['signal']) & np.isfinite(signal_led['control'])
        self.assertGreater(np.corrcoef(
            signal_led['signal'][valid], signal_led['control'][valid])[0, 1], 0.99)

    def test_quality_separates_registered_and_focal_frames(self):
        reference = _image()
        blurred_plane = ndi.gaussian_filter(reference, 2)
        unrelated_plane = np.flip(reference, axis=0)
        frames = np.asarray([reference] * 20 + [blurred_plane] * 2 + [unrelated_plane])
        estimates = [{
            'shift_y': 0,
            'shift_x': 0,
            'peak_ratio': 2,
            'tile_disagreement': 0.1,
            'out_of_range': False,
            'search_boundary': False,
            }] * len(frames)

        quality = measure_quality(reference, frames, estimates, 30)

        self.assertEqual(
            list(quality['recommended_state'][-3:]),
            ['focal_loss', 'focal_loss', 'ambiguous'],
            )
        np.testing.assert_array_equal(
            quality['analysis_valid'],
            quality['recommended_state'] == 'accepted',
            )
        self.assertIn('low_canonical_similarity', quality['reason_code'][-3])
        self.assertIn('low_high_frequency_fraction', quality['reason_code'][-3])
        self.assertIn('low_control_gain', quality['reason_code'][-3])
        self.assertEqual(quality['reason_code'][-1], 'loss_of_correspondence')

    def test_quality_passes_registered_frames_to_the_writer(self):
        reference = _image()
        frames = np.asarray([reference, reference])
        estimates = [{
            'shift_y': 1.25,
            'shift_x': -0.75,
            'peak_ratio': 2,
            'tile_disagreement': 0.1,
            'out_of_range': False,
            'search_boundary': False,
            }] * len(frames)
        written = []

        measure_quality(
            reference, frames, estimates, 30,
            write_registered=lambda frame_i, frame, bounds: written.append(
                (frame_i, frame.copy(), bounds)))

        expected, expected_bounds = warp_frame(reference, 1.25, -0.75)
        self.assertEqual([frame_i for frame_i, _, _ in written], [0, 1])
        np.testing.assert_array_equal(written[0][1], expected)
        self.assertEqual(written[0][2], expected_bounds)

    def test_quality_marks_blank_and_saturated_frames(self):
        reference = ((_image() + 1) * 1000).astype(np.uint16)
        frames = np.asarray([
            np.zeros_like(reference),
            np.full_like(reference, np.iinfo(reference.dtype).max),
            reference,
            ])
        estimates = [{
            'shift_y': 0,
            'shift_x': 0,
            'peak_ratio': 2,
            'tile_disagreement': 0.1,
            'out_of_range': False,
            'search_boundary': False,
            }] * len(frames)

        quality = measure_quality(reference, frames, estimates, 30)

        np.testing.assert_array_equal(
            quality['detector_artifact'], [True, True, False])

    def test_quality_marks_abrupt_photometric_changes(self):
        reference = _image()
        frames = np.asarray([reference] * 15 + [reference * 3] + [reference] * 15)
        estimates = [{
            'shift_y': 0,
            'shift_x': 0,
            'peak_ratio': 2,
            'tile_disagreement': 0.1,
            'out_of_range': False,
            'search_boundary': False,
            }] * len(frames)

        quality = measure_quality(reference, frames, estimates, 30)

        self.assertEqual(quality['reason_code'][15], 'photometric_artifact')
        self.assertEqual(quality['reason_code'][16], 'photometric_artifact')
        self.assertFalse(quality['analysis_valid'][15])
        self.assertFalse(quality['analysis_valid'][16])

    def test_quality_requires_usable_calibration_frames(self):
        reference = _image()
        frames = np.asarray([reference] * 3)
        estimates = [{
            'shift_y': 0,
            'shift_x': 0,
            'peak_ratio': 2,
            'tile_disagreement': 0.1,
            'out_of_range': False,
            'search_boundary': False,
            }] * len(frames)
        written = []

        with self.assertRaisesRegex(ValueError, 'no usable frames'):
            measure_quality(
                reference, frames, estimates, 30,
                calibration_mask=np.zeros(len(frames), dtype=bool),
                write_registered=lambda *frame: written.append(frame))
        for calibration_mask in (True, [True]):
            with self.assertRaisesRegex(ValueError, 'one value per frame'):
                measure_quality(
                    reference, frames, estimates, 30,
                    calibration_mask=calibration_mask,
                    write_registered=lambda *frame: written.append(frame))
        self.assertEqual(written, [])

    def test_quality_marks_nonfinite_motion_confidence_ambiguous(self):
        reference = _image()
        blurred = ndi.gaussian_filter(reference, 2)
        frames = np.asarray([reference] * 21 + [blurred])
        estimates = [{
            'shift_y': 0,
            'shift_x': 0,
            'peak_ratio': 2,
            'tile_disagreement': 0.1,
            'out_of_range': False,
            'search_boundary': False,
            } for _ in frames]
        for estimate in estimates[-2:]:
            estimate['peak_ratio'] = np.nan
            estimate['tile_disagreement'] = np.nan

        quality = measure_quality(reference, frames, estimates, 30)

        self.assertEqual(quality['recommended_state'][-2], 'ambiguous')
        self.assertEqual(
            quality['reason_code'][-2], 'ambiguous_peak;tile_disagreement')
        self.assertEqual(quality['recommended_state'][-1], 'focal_loss')
        self.assertIn('ambiguous_peak', quality['reason_code'][-1])
        self.assertIn('low_high_frequency_fraction', quality['reason_code'][-1])
        self.assertFalse(quality['threshold_calibration'][-2:].any())

    def test_quality_marks_irregular_timestamps(self):
        reference = _image()
        frames = np.asarray([reference] * 8)
        estimates = [{
            'shift_y': 0,
            'shift_x': 0,
            'peak_ratio': 2,
            'tile_disagreement': 0.1,
            'out_of_range': False,
            'search_boundary': False,
        }] * len(frames)
        frame_period = 1 / 30
        interval_frames = np.asarray([1, 1, 2, 1, 1, 0, 1], dtype=float)
        timestamps = np.r_[0, np.cumsum(interval_frames)] * frame_period

        quality = measure_quality(
            reference, frames, estimates, 30, timestamps=timestamps)

        expected = np.zeros(len(frames), dtype=bool)
        expected[[3, 6]] = True
        np.testing.assert_array_equal(quality['timing_fault'], expected)

    def test_local_reference_follows_slow_image_change(self):
        reference = _image()
        blurred = ndi.gaussian_filter(reference, 2)
        frames = np.asarray([reference] * 30 + [blurred] * 90)
        timestamps = np.arange(120, dtype=float)
        timestamps[30:] += 90
        estimates = [{
            'shift_y': 0,
            'shift_x': 0,
            'peak_ratio': 2,
            'tile_disagreement': 0.1,
            'out_of_range': False,
            'search_boundary': False,
            }] * len(frames)

        local_references = make_local_references(
            reference, frames, estimates, 1,
            timestamps=timestamps, max_frames=30)
        quality = measure_quality(
            reference, frames, estimates, 1,
            local_references=local_references, timestamps=timestamps)

        self.assertEqual(local_references['images'].shape[0], 3)
        self.assertFalse(local_references['canonical_fallback'].any())
        self.assertGreater(
            quality['local_gradient_ncc'][90],
            quality['canonical_gradient_ncc'][90] + 0.01,
            )
        self.assertTrue(np.isfinite(quality['temporal_difference'][1]))

    def test_focal_episodes_keep_short_durations_and_merge_point_one_seconds(self):
        states = np.full(40, 'accepted', dtype=object)
        states[0:3] = 'focal_loss'
        states[6:9] = 'focal_loss'
        states[15:17] = 'focal_loss'
        states[22:40] = 'focal_loss'

        episodes = focal_loss_episodes(states, 30)

        self.assertEqual(
            [episode['duration_class'] for episode in episodes],
            ['transient', 'brief', 'sustained'],
            )
        self.assertEqual(episodes[0]['n_frames'], 6)
        self.assertAlmostEqual(episodes[0]['duration_s'], 0.3)

    def test_focal_episodes_use_timestamp_gap_when_timestamps_are_supplied(self):
        states = np.full(10, 'accepted', dtype=object)
        states[0:2] = 'focal_loss'
        states[5:7] = 'focal_loss'
        timestamps = np.arange(10, dtype=float) / 30

        episodes = focal_loss_episodes(states, 30, timestamps=timestamps)
        self.assertEqual(len(episodes), 1)

        timestamps[5:] += 0.04

        episodes = focal_loss_episodes(states, 30, timestamps=timestamps)

        self.assertEqual(len(episodes), 2)

    def test_axial_similarity_uses_sixty_seconds(self):
        similarity = np.ones(1801)
        similarity[-1] = 0

        axial_similarity = rolling_axial_similarity(similarity, 30)

        self.assertEqual(axial_similarity[900], 1)

        similarity = np.r_[np.zeros(40), np.ones(60)]
        timestamps = np.arange(100, dtype=float)
        timestamps[40:] += 40
        axial_similarity = rolling_axial_similarity(
            similarity, 1, timestamps=timestamps)

        self.assertEqual(axial_similarity[40], 1)

    def test_quality_uses_the_named_nwb_locations(self):
        from datetime import datetime, timezone
        from pynwb import NWBFile, NWBHDF5IO, validate

        reference = _image()
        blurred = ndi.gaussian_filter(reference, 2)
        frames = np.asarray([reference] * 20 + [blurred] * 2)
        estimates = [{
            'shift_y': 0,
            'shift_x': 0,
            'peak_ratio': 2,
            'tile_disagreement': 0.1,
            'out_of_range': False,
            'search_boundary': False,
            }] * len(frames)
        quality = measure_quality(reference, frames, estimates, 30)
        nwbfile = NWBFile(
            session_description='registration QC serialisation test',
            identifier='qc-test',
            session_start_time=datetime(2026, 8, 17, tzinfo=timezone.utc))

        add_quality_control_to_nwb(nwbfile, quality)
        path = self.root / 'quality.nwb'
        with NWBHDF5IO(path, 'w') as io:
            io.write(nwbfile)
        with NWBHDF5IO(path, 'r') as io:
            saved = io.read()
            module = saved.processing['quality_control']
            self.assertIn('registration_qc', module.data_interfaces)
            self.assertIn('timing_fault', module['registration_qc'].colnames)
            self.assertIn('threshold_calibration', module['registration_qc'].colnames)
            self.assertEqual(module['axial_similarity'].unit, 'dimensionless')
            self.assertAlmostEqual(
                module['registration_thresholds']['canonical_focal'][0],
                quality['thresholds']['canonical_focal'],
                )
            self.assertEqual(len(saved.intervals['focal_loss']), 1)
        self.assertEqual(validate(path=path), [])


if __name__ == '__main__':
    unittest.main()
