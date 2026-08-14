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
import tifffile

from support import add_source_to_path

add_source_to_path()

from fibre_sight.preprocessing import read_tiffs


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


if __name__ == '__main__':
    unittest.main()
