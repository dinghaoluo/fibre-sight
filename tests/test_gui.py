'''
Created on 24 July 2026

Modified on 25 July 2026 to run each workbench check as a named probe

the workbench needs its own QApplication, so each probe runs in a separate
interpreter; this file is both the unittest entry point and the probe script

@author: Dinghao Luo
'''

#%% imports
from importlib.util import find_spec
from pathlib import Path
import subprocess
import sys
import unittest

from support import PROJECT_ROOT, source_environment

TEST_IMAGE = Path('examples/demo_test_ref_mat_ch2.npy')
SECOND_IMAGE = Path('examples/demo_train_02_ref_mat_ch2.npy')
THIRD_IMAGE = Path('examples/demo_train_01_ref_mat_ch2.npy')


#%% probes
def layout(window):
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QFontInfo
    from PyQt5.QtWidgets import QApplication, QScrollArea

    assert not window.isMaximized()
    assert window.minimumWidth() <= 1000
    assert window.minimumHeight() <= 680

    assert window.tabs.count() == 3
    assert [window.tabs.tabText(i) for i in range(3)] == ['Predict', 'Label', 'Train']
    assert all(isinstance(window.tabs.widget(i), QScrollArea) for i in range(3))

    # the bundled font has to survive packaging, so check the resolved family
    assert QFontInfo(window.font()).family() == 'mononoki'
    assert QFontInfo(window.font()).fixedPitch()

    assert window.activity_splitter.orientation() == Qt.Vertical
    assert window.activity_splitter.isCollapsible(1)
    assert window.activity_splitter.sizes()[1] == 0
    assert window.apply_settings_button.isHidden()
    assert 'QPushButton:focus' in window.styleSheet()

    # 1000 x 680 is the smallest laptop screen the workbench has to fit
    window.resize(1000, 680)
    window.show()
    QApplication.instance().processEvents()

    assert window.minimumSizeHint().height() <= 680
    tab_bar = window.tabs.tabBar()
    assert all(tab_bar.tabRect(i).right() < tab_bar.width() for i in range(3))
    visible = window.canvas.visibleRegion().boundingRect()
    assert visible.width() == window.canvas.width()
    assert visible.height() == window.canvas.height()

    window.dark_mode_check.setChecked(True)
    QApplication.instance().processEvents()
    assert window.dark_mode


def image_loading(window):
    window.image_line.setText(str(TEST_IMAGE))
    window.load_channel_image()
    assert window.image_path == TEST_IMAGE
    assert window.ref_image is not None
    assert window.recname == 'demo_test'
    assert window.default_roi_path() == (
        Path.cwd() / 'workspace/output/demo_predicted_ROI_dict.npy'
        )

    assert window.image_matches(TEST_IMAGE)
    assert window.image_matches(Path.cwd() / TEST_IMAGE)
    assert not window.image_matches(SECOND_IMAGE)

    window.browse_image = lambda: (
        window.image_line.setText(str(SECOND_IMAGE)) or str(SECOND_IMAGE)
        )
    window.segment_load_image_button.click()
    assert window.image_path == SECOND_IMAGE


def probability_invalidation(window):
    import numpy as np

    window.image_line.setText(str(TEST_IMAGE))
    window.load_channel_image()
    checkpoint_path = window.checkpoint_line.text()

    # a new checkpoint, image or device all make the retained map stale
    changes = [
        lambda: window.checkpoint_line.setText(checkpoint_path + '.changed'),
        lambda: window.image_line.setText(str(THIRD_IMAGE)),
        lambda: window.device_combo.setCurrentIndex(1),
        ]
    for change in changes:
        window.probability = np.zeros((8, 8), dtype=np.float32)
        window.labelled = np.zeros((8, 8), dtype=np.int32)
        window.labelled[1:3, 1:3] = 1
        window.update_roi_dict()
        window.refresh_status()
        assert window.apply_settings_button.isEnabled()
        assert not window.apply_settings_button.isHidden()

        change()
        assert window.probability is None
        assert not window.apply_settings_button.isEnabled()
        assert window.apply_settings_button.isHidden()

        window.checkpoint_line.setText(checkpoint_path)
        window.image_line.setText(str(TEST_IMAGE))
        window.device_combo.setCurrentIndex(0)


def undo_after_apply_settings(window):
    import numpy as np

    window.ref_image = np.zeros((8, 8), dtype=np.float32)
    window.labelled = np.zeros((8, 8), dtype=np.int32)
    window.labelled[1:3, 1:3] = 1
    window.update_roi_dict()
    window.probability = np.zeros((8, 8), dtype=np.float32)
    window.probability[4:6, 4:6] = 1
    window.min_size_spin.setValue(1)

    undo_count = len(window.undo_stack)
    window.apply_probability_threshold()
    assert len(window.undo_stack) == undo_count + 1
    # the rebuilt ROI follows the confidence map, not the ROI it replaced
    assert window.labelled[4, 4] > 0
    assert window.labelled[1, 1] == 0

    window.undo()
    assert len(window.undo_stack) == undo_count
    assert window.labelled[1, 1] > 0
    assert window.labelled[4, 4] == 0


def failed_training_run(window):
    from PyQt5.QtCore import QProcess

    checkpoint_path = window.checkpoint_line.text()
    window.checkpoint_line.clear()
    window.current_process_name = 'train'
    window.pending_checkpoint_path = Path(checkpoint_path)

    window.process_finished(1, QProcess.CrashExit)
    assert window.checkpoint_line.text() == ''
    assert window.last_saved_model_path is None
    assert window.pending_checkpoint_path is None


PROBES = {
    'layout': layout,
    'image_loading': image_loading,
    'probability_invalidation': probability_invalidation,
    'undo_after_apply_settings': undo_after_apply_settings,
    'failed_training_run': failed_training_run,
    }


#%% tests
@unittest.skipUnless(find_spec('PyQt5') is not None, 'PyQt5 is not installed')
class GUIProbeTests(unittest.TestCase):
    def run_probe(self, name):
        env = source_environment()
        env['QT_QPA_PLATFORM'] = 'offscreen'
        result = subprocess.run(
            [sys.executable, __file__, name],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_layout_fits_the_smallest_supported_window(self):
        self.run_probe('layout')

    def test_loading_an_image_tracks_the_selected_path(self):
        self.run_probe('image_loading')

    def test_changed_inputs_discard_the_confidence_map(self):
        self.run_probe('probability_invalidation')

    def test_apply_settings_can_be_undone(self):
        self.run_probe('undo_after_apply_settings')

    def test_failed_training_run_leaves_no_checkpoint(self):
        self.run_probe('failed_training_run')


#%% entry point
if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in PROBES:
        from PyQt5.QtWidgets import QApplication

        from fibre_sight.fibre_sight_workbench_gui import FibreSightWorkbench

        app = QApplication([])
        window = FibreSightWorkbench()
        PROBES[sys.argv[1]](window)
        window.close()
        app.quit()
    else:
        unittest.main()
