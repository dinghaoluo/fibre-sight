'''
Created on 24 July 2026

Modified on 14 August 2026 to keep the GUI checks at the workflow level

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

    window.tabs.setCurrentIndex(1)
    app.processEvents()
    assert window.segment_button.isVisible()

    window.tabs.setCurrentIndex(2)
    app.processEvents()
    assert window.train_model_button.isVisible()


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

        window.save_roi_file()
        assert [run['run_name'] for run in list_roi_runs(nwb_path)] == [
            'proposal',
            'second_proposal',
            'curated',
            ]
        assert 'ROI run already exists: curated' in window.output_box.toPlainText()


PROBES = {
    'viewport': viewport,
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
