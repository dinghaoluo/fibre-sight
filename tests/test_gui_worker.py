'''
Created on 21 August 2026

check automatic-session restart behaviour

@author: Dinghao Luo
'''

#%% imports
from pathlib import Path
from unittest import mock
import tempfile
import unittest

from fibre_sight import gui_worker


#%% tests
class AutomaticWorkerTests(unittest.TestCase):
    def test_dff_stage_messages_use_the_gui_label(self):
        config = {
            'source': {
                'multiplexed': True,
                },
            'signal_files': [{}, {}, {}],
            'control_files': [],
            'dff': {
                'baseline_window_s': 300.0,
                },
            }
        self.assertEqual(gui_worker._stage_display_name('dff'), 'dF/F')
        self.assertTrue(
            gui_worker._stage_start_text('dff', config).startswith('dF/F:'))

    def test_append_discards_an_incomplete_final_event(self):
        with tempfile.TemporaryDirectory(prefix='fibre sight worker ') as temp_dir:
            log_path = Path(temp_dir) / 'recording.fibresight.jsonl'
            gui_worker.append_session_event(log_path, {'event': 'configured'})
            with log_path.open('ab') as file:
                file.write(b'{"event": "stage_')

            gui_worker.append_session_event(
                log_path,
                {'event': 'stage_started', 'stage': 'preprocessing'},
                )

            events = gui_worker.read_session_events(log_path)
            self.assertEqual(
                [event['event'] for event in events],
                ['configured', 'stage_started'],
                )

    def test_provenance_failure_is_written_to_session_log(self):
        with tempfile.TemporaryDirectory(prefix='fibre sight worker ') as temp_dir:
            log_path = Path(temp_dir) / 'recording.fibresight.jsonl'
            with self.assertRaisesRegex(ValueError, 'different parameters'):
                gui_worker._run_stage(
                    log_path,
                    'segmentation',
                    mock.Mock(side_effect=ValueError('different parameters')),
                    mock.Mock(),
                    )

            self.assertEqual(
                gui_worker.session_stage_states(log_path)['segmentation'],
                'stage_failed',
                )

    def test_failed_stage_resumes_without_repeating_completed_work(self):
        with tempfile.TemporaryDirectory(prefix='fibre sight worker ') as temp_dir:
            root = Path(temp_dir)
            source_path = root / 'recording.tif'
            source_path.write_bytes(b'test recording')
            checkpoint_path = root / 'model.pt'
            checkpoint_path.write_bytes(b'test model')
            source_record = gui_worker.fingerprint_paths([source_path])[0]
            checkpoint_record = gui_worker.fingerprint_paths([checkpoint_path])[0]
            config = {
                'output_path': str(root / 'recording.nwb'),
                'source_files': [source_record],
                'signal_files': [source_record],
                'control_files': [],
                'source': {
                    'multiplexed': True,
                    'sampling_frequency_hz': 30.0,
                    'signal_channel': 1,
                    'control_channel': 2,
                    'signal_label': 'dLight',
                    'control_label': 'tdTomato',
                    'pixel_size_um': None,
                    },
                'registration': {'model': 'rigid', 'channel': 'control'},
                'segmentation': {
                    'run_name': 'proposal_auto',
                    'reference_channel': 'control',
                    'reference_low_percentile': 1.0,
                    'reference_high_percentile': 97.0,
                    'checkpoint_path': str(checkpoint_path),
                    'checkpoint_file': checkpoint_record,
                    'threshold': 0.25,
                    'min_size': 40,
                    'tta': True,
                    'device': 'cpu',
                    },
                'extraction': {
                    'run_name': 'fluorescence_auto',
                    'roi_run': 'proposal_auto',
                    'surround_method': 'adaptive',
                    'surround_inner_px': 5,
                    'surround_outer_px': 8,
                    'surround_min_pixels': 350,
                    },
                'dff': {
                    'run_name': 'dff_auto',
                    'fluorescence_run': 'fluorescence_auto',
                    'statistic': 'mean',
                    'baseline_percentile': 20.0,
                    'baseline_window_s': 300.0,
                    'surround_coefficient': 0.7,
                    'control_correction': 'none',
                    },
                }
            log_path = root / 'recording.fibresight.jsonl'
            gui_worker.append_session_event(
                log_path,
                {'event': 'configured', 'config': config},
                )

            with (
                    mock.patch.object(
                        gui_worker,
                        '_preprocessing_complete',
                        side_effect=[False, True],
                        ),
                    mock.patch.object(
                        gui_worker,
                        '_proposal_complete',
                        side_effect=[False, False],
                        ),
                    mock.patch.object(
                        gui_worker,
                        '_fluorescence_complete',
                        return_value=False,
                        ),
                    mock.patch.object(
                        gui_worker,
                        '_dff_complete',
                        return_value=False,
                        ),
                    mock.patch.object(
                        gui_worker,
                        'preprocess_recording',
                        return_value={'n_frames': 10},
                        ) as preprocess,
                    mock.patch.object(
                        gui_worker,
                        'segment_recording',
                        side_effect=[RuntimeError('interrupted'), {'roi_count': 2}],
                        ) as segment,
                    mock.patch.object(
                        gui_worker,
                        'extract_fluorescence',
                        return_value={'roi_count': 2},
                        ) as extract,
                    mock.patch.object(
                        gui_worker,
                        'calculate_dff',
                        return_value={'roi_count': 2},
                        ) as calculate,
                    ):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    gui_worker.run_session(log_path)
                gui_worker.run_session(log_path)

            self.assertEqual(preprocess.call_count, 1)
            self.assertEqual(segment.call_count, 2)
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(calculate.call_count, 1)
            self.assertEqual(gui_worker.session_stage_states(log_path), {
                'preprocessing': 'stage_skipped',
                'segmentation': 'stage_completed',
                'extraction': 'stage_completed',
                'dff': 'stage_completed',
                })


if __name__ == '__main__':
    unittest.main()
