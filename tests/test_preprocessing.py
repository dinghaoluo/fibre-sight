'''
Created on 14 August 2026
Modified on 17 August 2026

check TIFF reading, rigid registration, and saved QC

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

from fibre_sight.preprocessing import (
    add_quality_control_to_nwb,
    estimate_channel_offset,
    estimate_shift,
    focal_loss_episodes,
    make_local_references,
    make_reference,
    measure_quality,
    read_tiffs,
    register_pair,
    rolling_axial_similarity,
    warp_frame,
    )


#%% helpers
def _description(time_s=None, frame=None):
    lines = []
    if frame is not None:
        lines.append(f'frameNumbers = {frame}')
    if time_s is not None:
        lines.append(f'frameTimestamps_sec = {time_s:.9f}')
    return '\n'.join(lines)


def _write_pages(path, pages, times=None, frames=None, channels=None):
    times = times if times is not None else [None] * len(pages)
    frames = frames if frames is not None else [None] * len(pages)
    software = None if channels is None else f'SI.hChannels.channelSave = {channels}'
    with tifffile.TiffWriter(path) as writer:
        for page, time_s, frame in zip(pages, times, frames):
            writer.write(
                page,
                description=_description(time_s, frame),
                software=software,
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

    def test_multiplexed_tiffs_follow_chunk_and_channel_order(self):
        early_path = self.root / 'recording_2.tif'
        late_path = self.root / 'recording_10.tif'
        early_signal = np.full((3, 5), 10, dtype=np.int16)
        early_control = np.full((3, 5), 20, dtype=np.int16)
        late_signal = np.full((3, 5), 11, dtype=np.int16)
        late_control = np.full((3, 5), 21, dtype=np.int16)
        frame_time = 1 / 30
        _write_pages(
            early_path,
            [early_control, early_signal],
            times=[0, 0],
            frames=[1, 1],
            channels='[2 1]',
            )
        _write_pages(
            late_path,
            [late_control, late_signal],
            times=[frame_time, frame_time],
            frames=[2, 2],
            channels='[2 1]',
            )

        frames = list(read_tiffs(
            [late_path, early_path],
            signal_channel=1,
            control_channel=2,
            sampling_frequency_hz=30,
            ))

        np.testing.assert_allclose(
            [frame['time_s'] for frame in frames],
            [0, frame_time],
            atol=1e-9,
            )
        np.testing.assert_array_equal(frames[0]['signal'], early_signal)
        np.testing.assert_array_equal(frames[0]['control'], early_control)
        np.testing.assert_array_equal(frames[1]['signal'], late_signal)
        np.testing.assert_array_equal(frames[1]['control'], late_control)

    def test_separate_tiffs_use_nominal_time(self):
        signal_path = self.root / 'signal_2.tif'
        control_path = self.root / 'control_2.tif'
        signal = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
        control = signal + 100
        _write_pages(signal_path, signal)
        _write_pages(control_path, control)

        frames = list(read_tiffs(
            [signal_path],
            control_tiffs=[control_path],
            multiplexed=False,
            signal_channel=1,
            control_channel=2,
            sampling_frequency_hz=20,
            ))

        self.assertEqual([frame['time_s'] for frame in frames], [0, 0.05])
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
                _write_pages(path, pages, channels='[1 2]')
                with self.assertRaises(ValueError):
                    list(read_tiffs(
                        [path], signal_channel=1, control_channel=2,
                        sampling_frequency_hz=30,
                        ))

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
            signal_offset=(offset['shift_y'], offset['shift_x']),
            )
        valid = np.isfinite(result['signal']) & np.isfinite(result['control'])
        self.assertGreater(
            np.corrcoef(result['signal'][valid], result['control'][valid])[0, 1], 0.99)

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
            self.assertIn('rigid_registration_qc', module.data_interfaces)
            self.assertIn('timing_fault', module['rigid_registration_qc'].colnames)
            self.assertIn('threshold_calibration', module['rigid_registration_qc'].colnames)
            self.assertEqual(module['axial_similarity'].unit, 'dimensionless')
            self.assertAlmostEqual(
                module['rigid_registration_thresholds']['canonical_focal'][0],
                quality['thresholds']['canonical_focal'],
                )
            self.assertEqual(len(saved.intervals['focal_loss']), 1)
        self.assertEqual(validate(path=path), [])


if __name__ == '__main__':
    unittest.main()
