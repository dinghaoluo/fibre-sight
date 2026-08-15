'''
Created on 14 August 2026

check TIFF channel pairing, timing, and frame consistency

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
    estimate_channel_offset,
    estimate_shift,
    make_reference,
    read_tiffs,
    register_pair,
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
        frames = []
        for _ in range(30):
            shift = rng.uniform(-4, 4, 2)
            frame = ndi.shift(image, shift, order=1, mode='constant', cval=0)
            frames.append((1 + rng.uniform(-0.15, 0.15)) * frame + rng.normal(0, 0.03, image.shape))
        frames = np.asarray(frames, dtype=np.float32)

        result = make_reference(
            frames,
            max_frames=30,
            min_frames=20,
            min_peak_ratio=1.02,
            max_tile_disagreement=1.5,
            )
        unregistered = frames.mean(axis=0)
        registered_shift = estimate_shift(image, result['image'], check_tiles=False)
        mean_shift = estimate_shift(image, unregistered, check_tiles=False)
        registered, _ = warp_frame(
            result['image'], registered_shift['shift_y'], registered_shift['shift_x'])
        unregistered, _ = warp_frame(unregistered, mean_shift['shift_y'], mean_shift['shift_x'])
        registered_valid = np.isfinite(registered)
        mean_valid = np.isfinite(unregistered)
        registered_ncc = np.corrcoef(image[registered_valid], registered[registered_valid])[0, 1]
        mean_ncc = np.corrcoef(image[mean_valid], unregistered[mean_valid])[0, 1]
        self.assertGreater(registered_ncc, mean_ncc + 0.05)

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


if __name__ == '__main__':
    unittest.main()
