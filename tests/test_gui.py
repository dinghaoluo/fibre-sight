'''
Created on 24 July 2026

Modified on 14 August 2026 to keep the GUI checks at the workflow level
Modified on 21 August 2026 for the automatic recording workflow

run GUI checks separately because QApplication state cannot be reset

@author: Dinghao Luo
'''

#%% imports
from importlib.util import find_spec
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import subprocess
import sys
import tempfile
import unittest

from tests.support import PROJECT_ROOT, source_environment


#%% fixtures
class _GuiModifiers:
    def __init__(self, modifiers):
        self._modifiers = modifiers

    def modifiers(self):
        return self._modifiers


def _click_event(window, xpix, ypix, modifiers):
    return SimpleNamespace(
        button=1,
        inaxes=window.ax,
        xdata=float(xpix),
        ydata=float(ypix),
        guiEvent=_GuiModifiers(modifiers),
        )


def _seed_canvas(window, include_probability=False):
    import numpy as np

    height, width = 80, 100
    ypix, xpix = np.mgrid[:height, :width]
    image = (0.2 * xpix + 0.1 * ypix).astype(np.float32)
    labelled = np.zeros((height, width), dtype=np.int32)
    labelled[1:6, 1:7] = 1
    labelled[34:42, 56:66] = 2

    window.image_line.setText('synthetic_ref_mat_ch2.npy')
    window.image_path = Path('synthetic_ref_mat_ch2.npy')
    window.recname = 'synthetic'
    window.ref_image = image
    window.labelled = labelled
    window.fixed_ids = {2}
    window.selected.clear()
    window.update_roi_dict()
    if include_probability:
        window.probability = np.linspace(
            0,
            1,
            image.size,
            dtype=np.float32,
            ).reshape(image.shape)
    window.plot_image()
    window.canvas.fit_to_image()
    window.refresh_status()
    return image


def _preprocessed_nwb(path, reference):
    from datetime import datetime, timezone

    from hdmf.backends.hdf5.h5_utils import H5DataIO
    from hdmf.common import DynamicTable
    import numpy as np
    from pynwb import NWBFile, NWBHDF5IO
    from pynwb.base import Images
    from pynwb.image import GrayscaleImage, ImageSeries

    height, width = reference.shape
    nwbfile = NWBFile(
        session_description='GUI curation adapter test',
        identifier=path.stem,
        session_start_time=datetime(2026, 8, 19, tzinfo=timezone.utc),
        )
    preprocessing = nwbfile.create_processing_module(
        name='preprocessing',
        description='minimal registered recording',
        )
    for channel_index, channel in enumerate(('signal', 'control')):
        movie = np.arange(3 * width * height, dtype=np.int16).reshape(
            3, width, height) + channel_index
        preprocessing.add(ImageSeries(
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
    preprocessing.add(Images(
        name='registration_references',
        images=[GrayscaleImage(
            name='control_reference',
            data=np.asarray(reference, dtype=np.float32).T,
            description='test reference',
            )],
        ))
    metadata = DynamicTable(
        name='recording_metadata',
        description='test metadata',
        )
    metadata.add_column(name='control_label', description='control label')
    metadata.add_column(name='pixel_size_um', description='pixel size')
    metadata.add_row(control_label='tdTomato', pixel_size_um=1.2)
    preprocessing.add(metadata)
    with NWBHDF5IO(path, 'w') as io:
        io.write(nwbfile)


#%% probes
def viewport(window):
    from PyQt5.QtCore import QPoint
    from PyQt5.QtWidgets import QApplication

    from fibre_sight.gui import GUI_FONT_SIZES

    app = QApplication.instance()
    window.resize(1000, 680)
    window.show()

    assert [
        window.tabs.tabText(index) for index in range(window.tabs.count())
        ] == ['AUTO', 'train', 'segment']

    auto_controls = (
        window.auto_tiff_dir_line,
        window.auto_acquisition_combo,
        window.auto_sampling_spin,
        )
    predict_controls = (
        window.threshold_spin,
        window.min_size_spin,
        window.predict_button,
        )
    curation_controls = (
        window.curate_buttons['select_all'],
        window.fix_selected_button,
        window.curate_buttons['merge'],
        window.curate_buttons['delete'],
        window.curate_buttons['undo'],
        window.segment_load_roi_button,
        window.curate_buttons['save_roi'],
        window.curate_buttons['zoom_out'],
        window.curate_buttons['fit_view'],
        window.curate_buttons['zoom_in'],
        )

    window.interface_font_actions[max(GUI_FONT_SIZES)].trigger()
    window.tabs.setCurrentIndex(0)
    app.processEvents()

    assert not window.resources_panel.isVisible()
    assert not window.persistent_panel.isVisible()
    assert not window.stage_header.isVisible()
    assert not window.right_stack.isVisible()
    assert window.activity_frame.isVisible()
    assert window.auto_run_button.text() == 'RUN PIPELINE'
    assert [
        window.auto_registration_model_combo.itemText(index)
        for index in range(window.auto_registration_model_combo.count())
        ] == ['rigid', 'piecewise', 'auto']
    assert window.auto_registration_model_combo.currentText() == 'auto'
    assert window.auto_sampling_spin.decimals() == 1
    assert window.auto_sampling_spin.text() == '30.0 Hz'
    assert not hasattr(window, 'auto_source_summary')
    assert all(
        widget.visibleRegion().contains(widget.rect())
        for widget in auto_controls
        )
    window.auto_advanced_button.setChecked(True)
    window.auto_tab_scroll.verticalScrollBar().setValue(
        window.auto_tab_scroll.verticalScrollBar().maximum())
    app.processEvents()
    assert window.auto_footer.isVisible()
    assert all(
        widget.visibleRegion().contains(widget.rect())
        for widget in (
            window.auto_state_label,
            window.auto_progress,
            window.auto_run_button,
            window.auto_resume_button,
            )
        )

    window.tabs.setCurrentIndex(2)
    app.processEvents()
    assert window.resources_panel.isVisible()
    assert window.persistent_panel.isVisible()
    assert window.stage_header.isVisible()
    assert window.right_stack.isVisible()
    assert window.roi_table.isVisible()
    assert all(
        widget.visibleRegion().contains(widget.rect())
        for widget in predict_controls
        )

    bar = window.curation_bar
    for button in curation_controls:
        top_left = button.mapTo(bar, QPoint(0, 0))
        assert top_left.x() >= 0
        assert top_left.y() >= 0
        assert top_left.x() + button.width() <= bar.width()
        assert top_left.y() + button.height() <= bar.height()

    window.mser_advanced_button.setChecked(True)
    app.processEvents()
    assert window.segment_button.isVisible()

    window.tabs.setCurrentIndex(1)
    app.processEvents()
    window.training_tab.verticalScrollBar().setValue(
        window.training_tab.verticalScrollBar().maximum())
    app.processEvents()
    assert not window.resources_panel.isVisible()
    assert not window.persistent_panel.isVisible()
    assert not window.curation_bar.isVisible()
    assert not window.stage_header.isVisible()
    assert not window.right_stack.isVisible()
    assert window.activity_frame.isVisible()
    assert window.train_output_dir_line.isVisible()
    assert window.train_model_button.isVisible()

    window.tabs.setCurrentIndex(2)
    app.processEvents()
    assert window.resources_panel.isVisible()
    assert window.persistent_panel.isVisible()
    assert window.curation_bar.isVisible()
    assert window.stage_header.isVisible()
    assert window.right_stack.isVisible()


def automatic_session(window):
    import numpy as np
    from PyQt5.QtCore import Qt

    from fibre_sight.gui_worker import (
        append_session_event,
        latest_session_config,
        session_stage_states,
        )

    with tempfile.TemporaryDirectory(prefix='fibre sight automatic ') as temp_dir:
        root = Path(temp_dir)
        tiff_dir = root / 'TIFFs'
        tiff_dir.mkdir()
        (tiff_dir / 'recording_10.tif').write_bytes(b'late')
        (tiff_dir / 'recording_2.tif').write_bytes(b'early')
        checkpoint = root / 'model.pt'
        checkpoint.write_bytes(b'model')

        window.auto_tiff_dir_line.setText(str(tiff_dir))
        window.auto_output_dir_line.setText(str(root))
        window.auto_session_line.setText('example')
        window.auto_checkpoint_line.setText(str(checkpoint))
        window.auto_reference_channel_combo.setCurrentText('signal')
        window.auto_reference_high_spin.setValue(97)
        window.auto_threshold_spin.setValue(0.31)
        window.auto_min_size_spin.setValue(52)
        config = window.auto_session_config()

        assert [Path(record['path']).name for record in config['signal_files']] == [
            'recording_2.tif',
            'recording_10.tif',
            ]
        assert config['segmentation']['reference_channel'] == 'signal'
        assert config['segmentation']['reference_high_percentile'] == 97
        assert config['segmentation']['threshold'] == 0.31
        assert config['segmentation']['min_size'] == 52
        assert config['extraction']['roi_run'] == 'proposal_auto'
        assert config['dff']['fluorescence_run'] == 'fluorescence_auto'

        log_path = root / 'example.fibresight.jsonl'
        append_session_event(log_path, {'event': 'configured', 'config': config})
        append_session_event(
            log_path,
            {'event': 'stage_completed', 'stage': 'preprocessing'},
            )
        assert latest_session_config(log_path) == config
        assert session_stage_states(log_path)['preprocessing'] == 'stage_completed'

        window.auto_log_path = log_path
        window.auto_loaded_config = config
        window.auto_session_line.setText('next_example')
        with mock.patch.object(window, 'start_process', return_value=True) as start:
            window.start_auto_session()
        next_log_path = root / 'next_example.fibresight.jsonl'
        assert window.auto_log_path == next_log_path.resolve()
        assert Path(latest_session_config(next_log_path)['output_path']).name == (
            'next_example.nwb')
        start.assert_called_once()

        window.trace_cache = {
            'run_name': 'dff_auto',
            'timestamps': np.arange(20, dtype=float),
            'analysis_valid': np.ones(20, dtype=bool),
            'fluorescence': {'traces': {
                'signal_roi_mean': np.arange(20, dtype=float)[:, None],
                'signal_surround_mean': np.ones((20, 1), dtype=float),
                }},
            'dff': {
                'roi_ids': np.asarray([7]),
                'provenance': {'statistic': 'mean', 'surround_coefficient': 0.7},
                'traces': {
                    'signal_surround_corrected_dff': np.zeros((20, 1)),
                    'control_surround_corrected_dff': np.ones((20, 1)),
                    },
                },
            }
        window.trace_roi_combo.addItem('7', 7)
        window.draw_trace_inspector()
        assert len(window.trace_figure.axes) == 1
        assert window.trace_figure.axes[0].get_title() == 'ROI 7 dF/F'
        assert len(window.trace_figure.axes[0].lines) == 2
        assert all(
            abs(line.get_linewidth() - 0.9) < 1e-6
            for line in window.trace_figure.axes[0].lines
            )
        window.trace_signal_check.setChecked(False)
        assert len(window.trace_figure.axes[0].lines) == 1
        window.trace_signal_check.setChecked(True)
        window.trace_zoom_in_button.click()
        assert window.trace_figure.axes[0].get_xlim()[1] < 20
        window.nwb_path = Path(window.auto_loaded_config['output_path'])
        window.controls_tab_changed(0)
        assert not window.right_stack.isHidden()
        assert window.right_stack.currentIndex() == 0
        assert [
            window.activity_tabs.tabText(index)
            for index in range(window.activity_tabs.count())
            ] == ['trace inspector', 'console']
        assert window.activity_tabs.currentIndex() == 0
        window.activity_tabs.setCurrentIndex(1)
        assert window.activity_tabs.currentIndex() == 1
        window.activity_tabs.setCurrentIndex(0)

        window.ref_image = np.ones((6, 8), dtype=float)
        window.labelled = np.zeros((6, 8), dtype=np.int32)
        window.labelled[1:4, 2:5] = 7
        window.roi_dict = {
            7: {
                'xpix': np.array([2, 3, 4]),
                'ypix': np.array([1, 2, 3]),
                },
        }
        window.selected.clear()
        window.refresh_roi_table()
        window.activity_tabs.setCurrentIndex(1)
        window.on_click(_click_event(window, 3, 2, Qt.NoModifier))
        assert window.selected == {7}
        assert window.trace_roi_combo.currentData() == 7
        assert window.activity_tabs.currentIndex() == 0

        window.load_auto_session_log(next_log_path)
        assert window.trace_cache is None
        assert window.trace_roi_combo.count() == 0
        assert window.right_stack.isHidden()


def image_prediction(window):
    import numpy as np

    with tempfile.TemporaryDirectory(prefix='fibre sight prediction ') as temp_dir:
        temp_dir = Path(temp_dir)
        first_path = temp_dir / 'first_ref_mat_ch2.npy'
        second_path = temp_dir / 'second_ref_mat_ch2.npy'
        checkpoint_path = temp_dir / 'best.pt'
        first = np.arange(120, dtype=np.float32).reshape(10, 12)
        second = np.full((8, 9), 7, dtype=np.float32)
        np.save(first_path, first)
        np.save(second_path, second)
        checkpoint_path.touch()

        window.image_line.setText(str(first_path))
        window.load_channel_image()
        np.testing.assert_array_equal(window.ref_image, first)

        window.checkpoint_line.setText(str(checkpoint_path))

        labelled = np.zeros_like(window.ref_image, dtype=np.int32)
        labelled[2:5, 3:7] = 1
        ypix, xpix = np.nonzero(labelled == 1)
        probability = np.zeros_like(window.ref_image, dtype=np.float32)
        probability[2:5, 3:7] = 0.9
        probability[7:9, 8:10] = 0.6
        roi_dict = {1: {'xpix': xpix, 'ypix': ypix}}

        class FakePredictor:
            def __init__(self):
                self.checkpoint_path = checkpoint_path
                self.device = 'cpu'
                self.threshold = 0.25
                self.min_size = 1
                self.max_size = None
                self.tta = True
                self.calls = 0

            def predict_image(self, _image):
                self.calls += 1
                return roi_dict, labelled, probability

        predictor = FakePredictor()
        window.predictor = predictor
        window.min_size_spin.setValue(1)
        window.predict_rois()

        assert len(window.roi_dict) == 1
        assert predictor.calls == 1

        before = window.labelled.copy()
        window.threshold_spin.setValue(0.5)
        window.rebuild_rois_from_probability()
        assert predictor.calls == 1
        assert len(window.roi_dict) == 2
        window.undo()
        np.testing.assert_array_equal(window.labelled, before)

        window.set_display_mode('confidence')
        window.checkpoint_line.setText(str(checkpoint_path) + '.changed')
        assert window.probability is None
        assert window.display_mode == 'image'

        window.checkpoint_line.setText(str(checkpoint_path))
        window.probability = probability
        window.set_display_mode('confidence')
        window.image_line.setText(str(second_path))
        window.load_channel_image()
        np.testing.assert_array_equal(window.ref_image, second)
        assert window.roi_dict == {}
        assert not np.any(window.labelled)
        assert window.probability is None
        assert window.display_mode == 'image'


def roi_editing(window):
    import numpy as np
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QFileDialog

    from fibre_sight.roi_io import load_roi_dict

    with tempfile.TemporaryDirectory(prefix='fibre sight roi editing ') as temp_dir:
        temp_dir = Path(temp_dir)
        image_path = temp_dir / 'session_ref_mat_ch2.npy'
        np.save(image_path, _seed_canvas(window))
        window.image_path = image_path
        window.image_line.setText(str(image_path))
        window.recname = 'session'
        before = window.labelled.copy()

        window.on_click(_click_event(window, 2, 2, Qt.NoModifier))
        window.on_click(_click_event(window, 60, 36, Qt.ShiftModifier))
        assert window.selected == {1, 2}

        window.merge_selected()
        assert len(window.roi_dict) == 1
        assert window.fixed_ids == {1}
        window.undo()
        np.testing.assert_array_equal(window.labelled, before)
        assert window.selected == {1, 2}
        assert window.fixed_ids == {2}

        window.save_roi_file()
        saved = load_roi_dict(window.default_roi_path())
        assert set(saved) == {1, 2}

        imported_path = temp_dir / 'imported_ROI_dict.npy'
        np.save(imported_path, {
            7: {
                'xpix': np.array([1, 2, 2]),
                'ypix': np.array([1, 1, 2]),
                },
            })
        with mock.patch.object(
                QFileDialog,
                'getOpenFileName',
                return_value=(str(imported_path), 'NumPy dict (*.npy)'),
                ):
            window.load_roi_file()

        expected = np.zeros_like(window.labelled)
        expected[1, 1:3] = 1
        expected[2, 2] = 1
        np.testing.assert_array_equal(window.labelled, expected)
        assert window.probability is None


def segmentation_fixed(window):
    import numpy as np

    import fibre_sight.gui as gui
    from fibre_sight.roi_io import labels_to_roi_dict

    with tempfile.TemporaryDirectory(prefix='fibre sight segmentation ') as temp_dir:
        image_path = Path(temp_dir) / 'session_ref_mat_ch2.npy'
        np.save(image_path, np.arange(120, dtype=np.float32).reshape(10, 12))
        window.image_line.setText(str(image_path))
        window.load_channel_image()

        window.labelled[1:4, 1:4] = 1
        window.update_roi_dict()
        window.selected = {1}
        window.fixed_ids = {1}
        window.probability = np.ones_like(window.ref_image, dtype=np.float32)
        window.refresh_status()
        window.set_display_mode('confidence')

        before = window.labelled.copy()
        expected_ypix, expected_xpix = np.where(before == 1)
        returned = before.copy()
        returned[6:9, 7:10] = 2
        returned_rois = labels_to_roi_dict(returned)

        def fake_segment(_image, _params, fixed_rois=None):
            np.testing.assert_array_equal(
                fixed_rois[1]['xpix'],
                expected_xpix,
                )
            np.testing.assert_array_equal(
                fixed_rois[1]['ypix'],
                expected_ypix,
                )
            return (
                returned_rois,
                returned,
                {1},
                {'MSER regions': 2, 'kept ROIs': 2, 'fixed ROIs': 1},
                )

        with mock.patch.object(gui, 'segment_mser', side_effect=fake_segment):
            window.segment_rois()

        assert window.fixed_ids == {1}
        np.testing.assert_array_equal(window.labelled[1:4, 1:4], 1)
        assert window.probability is None
        assert window.display_mode == 'image'

        window.undo()
        np.testing.assert_array_equal(window.labelled, before)
        assert window.selected == {1}
        assert window.fixed_ids == {1}


def nwb_curation(window):
    import numpy as np
    from PyQt5.QtWidgets import QApplication

    import fibre_sight.api as api
    from fibre_sight.api import list_roi_runs, load_roi_run

    with tempfile.TemporaryDirectory(prefix='fibre sight NWB curation ') as temp_dir:
        temp_dir = Path(temp_dir)
        nwb_path = temp_dir / 'recording.nwb'
        reference = np.arange(24, dtype=np.float32).reshape(4, 6)
        _preprocessed_nwb(nwb_path, reference)

        class FakePredictor:
            def __init__(self, **_kwargs):
                self.checkpoint_path = Path(__file__)
                self.threshold = 0.25
                self.min_size = 2
                self.max_size = 100
                self.tta = True
                self.device = 'cpu'

            def predict_image(self, image):
                probability = np.asarray(image, dtype=np.float32) / np.max(image)
                roi_dict = {
                    4: {
                        'xpix': np.asarray([1, 2, 2]),
                        'ypix': np.asarray([1, 1, 2]),
                        },
                    }
                return roi_dict, np.zeros(image.shape, dtype=np.int32), probability

        with mock.patch.object(api, 'ROIPredictor', FakePredictor):
            api.segment_recording(nwb_path, 'proposal')
            api.segment_recording(nwb_path, 'second_proposal')

        window.image_line.setText(str(nwb_path))
        window.load_channel_image()
        assert window.ref_image is None
        assert window.source_proposal_run is None
        assert not window.predict_button.isEnabled()
        window.tabs.setCurrentIndex(2)
        window.proposal_run_combo.setCurrentText('proposal')
        window.resize(1000, 680)
        window.show()
        QApplication.processEvents()
        assert window.proposal_run_combo.visibleRegion().contains(
            window.proposal_run_combo.rect()
            )
        assert window.curated_run_line.visibleRegion().contains(
            window.curated_run_line.rect()
            )
        np.testing.assert_array_equal(window.ref_image, reference)
        np.testing.assert_array_equal(window.probability, reference / 23)
        assert window.source_proposal_run == 'proposal'
        assert window.proposal_run_combo.currentData() == 'proposal'
        assert window.predict_button.isEnabled()
        window.threshold_spin.setValue(0.9)
        window.min_size_spin.setValue(1)
        window.rebuild_rois_from_probability()
        assert window.roi_dict
        window.proposal_run_combo.setCurrentText('second_proposal')
        window.proposal_run_combo.setCurrentText('proposal')
        assert set(zip(
            window.roi_dict[1]['xpix'],
            window.roi_dict[1]['ypix'],
            )) == {(1, 1), (2, 1), (2, 2)}

        window.labelled[2, 3] = 1
        window.update_roi_dict()
        window.curated_run_line.setText('curated')
        window.save_roi_file()

        proposal = load_roi_run(nwb_path, 'proposal')
        curated = load_roi_run(nwb_path, 'curated')
        assert set(zip(
            proposal['roi_dict'][4]['xpix'],
            proposal['roi_dict'][4]['ypix'],
            )) == {(1, 1), (2, 1), (2, 2)}
        assert set(zip(
            curated['roi_dict'][1]['xpix'],
            curated['roi_dict'][1]['ypix'],
            )) == {(1, 1), (2, 1), (2, 2), (3, 2)}
        assert curated['provenance']['source_run'] == 'proposal'

        window.labelled.fill(0)
        window.update_roi_dict()
        window.curated_run_line.setText('empty_curated')
        assert window.curate_buttons['save_roi'].isEnabled()
        window.save_roi_file()
        assert load_roi_run(nwb_path, 'empty_curated')['roi_dict'] == {}

        window.curated_run_line.setText('curated')
        window.save_roi_file()
        assert [run['run_name'] for run in list_roi_runs(nwb_path)] == [
            'proposal',
            'second_proposal',
            'curated',
            'empty_curated',
            ]
        assert 'ROI run already exists: curated' in window.output_box.toPlainText()


def training_configuration(window):
    from fibre_sight.config import save_recipe

    with tempfile.TemporaryDirectory(prefix='fibre sight training ') as temp_dir:
        root = Path(temp_dir)
        manifest = root / 'manifest.csv'
        manifest.touch()
        output_dir = root / 'runs'
        run_dir = output_dir / 'gui_test'

        window.source_root_line.setText(str(root))
        window.manifest_line.setText(str(manifest))
        window.train_output_dir_line.setText(str(output_dir))
        window.run_name_line.setText('gui_test')
        window.epochs_spin.setValue(3)
        window.threshold_spin.setValue(0.21)
        window.min_size_spin.setValue(7)

        with mock.patch('fibre_sight.gui.save_recipe') as save:
            window.write_training_config()
        recipe = save.call_args.args[0]
        assert recipe['train']['out_dir'] == str(output_dir)
        assert recipe['train']['run_name'] == 'gui_test'
        assert recipe['train']['epochs'] == 3
        assert recipe['postprocess']['threshold'] == 0.5
        assert recipe['postprocess']['min_size'] == 30
        assert window.training_checkpoint_path() == run_dir / 'best.pt'

        run_dir.mkdir(parents=True)
        (run_dir / 'best.pt').touch()
        save_recipe(recipe, run_dir / 'config.yaml')
        window.refresh_status()
        assert window.evaluate_model_button.isEnabled()
        with mock.patch.object(window, 'start_process', return_value=True) as start:
            window.evaluate_model()
        args = start.call_args.args[2]
        assert args[args.index('--checkpoint') + 1] == str(run_dir / 'best.pt')
        assert args[args.index('--threshold') + 1] == '0.5'
        assert args[args.index('--min-size') + 1] == '30'


PROBES = {
    'viewport': viewport,
    'automatic_session': automatic_session,
    'training_configuration': training_configuration,
    'image_prediction': image_prediction,
    'roi_editing': roi_editing,
    'segmentation_fixed': segmentation_fixed,
    'nwb_curation': nwb_curation,
    }


#%% tests
@unittest.skipUnless(find_spec('PyQt5') is not None, 'PyQt5 is not installed')
class GUIProbeTests(unittest.TestCase):
    def run_probe(self, name, scale_factor='1'):
        env = source_environment()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        env['QT_SCALE_FACTOR'] = scale_factor
        env['MPLCONFIGDIR'] = str(
            Path(tempfile.gettempdir()) / 'fibre-sight-matplotlib'
            )
        env['XDG_CACHE_HOME'] = str(
            Path(tempfile.gettempdir()) / 'fibre-sight-cache'
            )
        result = subprocess.run(
            [sys.executable, '-m', 'tests.test_gui', name],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_viewport(self):
        self.run_probe('viewport', '1.5')

    def test_automatic_session_configuration_and_trace_view(self):
        self.run_probe('automatic_session')

    def test_training_configuration(self):
        self.run_probe('training_configuration')

    def test_image_prediction_and_confidence(self):
        self.run_probe('image_prediction')

    def test_roi_editing_and_files(self):
        self.run_probe('roi_editing')

    def test_segmentation_preserves_fixed_rois(self):
        self.run_probe('segmentation_fixed')

    def test_nwb_proposal_curation(self):
        self.run_probe('nwb_curation')


#%% entry point
if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in PROBES:
        from PyQt5.QtCore import QSettings, Qt
        from PyQt5.QtWidgets import QApplication

        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        app = QApplication([])
        app.setStyle('Fusion')

        from fibre_sight.gui import FibreSightGUI

        with tempfile.TemporaryDirectory(prefix='fibre sight settings ') as temp_dir:
            settings = QSettings(
                str(Path(temp_dir) / 'settings.ini'),
                QSettings.IniFormat,
                )
            window = FibreSightGUI(settings=settings)
            PROBES[sys.argv[1]](window)
            window.close()
        app.quit()
    else:
        unittest.main()
