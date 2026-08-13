'''
Created on 24 July 2026

Modified on 25 July 2026 to run each workbench check as a named probe
Modified on 29 July 2026 to cover the revised workbench interactions

the workbench needs its own QApplication, so each probe runs in a separate
interpreter; this file is both the unittest entry point and the probe script

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

from support import PROJECT_ROOT, source_environment

TEST_IMAGE = Path('examples/demo_test_ref_mat_ch2.npy')
SECOND_IMAGE = Path('examples/demo_train_02_ref_mat_ch2.npy')
THIRD_IMAGE = Path('examples/demo_train_01_ref_mat_ch2.npy')


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


def _seed_canvas(window, include_probability=True):
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
    window.selected = {1}
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


def _assert_limits_equal(observed, expected, atol=1e-7):
    import numpy as np

    np.testing.assert_allclose(observed[0], expected[0], atol=atol, rtol=0)
    np.testing.assert_allclose(observed[1], expected[1], atol=atol, rtol=0)


def _span(limits):
    return abs(float(limits[1]) - float(limits[0]))


def _data_point(window, xdata, ydata):
    from PyQt5.QtCore import QPoint

    xdisplay, ydisplay = window.ax.transData.transform((xdata, ydata))
    return QPoint(
        int(round(xdisplay)),
        int(round(window.canvas.height() - ydisplay)),
        )


#%% probes
def layout(window):
    from PyQt5.QtCore import QPoint, Qt
    from PyQt5.QtGui import QColor, QFont, QFontInfo, QPalette
    from PyQt5.QtWidgets import (
        QAbstractButton,
        QApplication,
        QLabel,
        QProgressBar,
        QScrollArea,
        QSizePolicy,
        QToolTip,
        QWidget,
        )

    from fibre_sight.fibre_sight_workbench_gui import (
        GUI_FONT_SIZE,
        GUI_FONT_SIZES,
        )

    assert GUI_FONT_SIZE == (12.0 if sys.platform == 'darwin' else 9.0)

    def relative_luminance(colour):
        channels = [channel / 255 for channel in colour.getRgb()[:3]]
        channels = [
            channel / 12.92
            if channel <= 0.04045 else
            ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
            ]
        return (
            0.2126 * channels[0] +
            0.7152 * channels[1] +
            0.0722 * channels[2]
            )

    def contrast(first, second):
        first_luminance = relative_luminance(first)
        second_luminance = relative_luminance(second)
        lighter = max(first_luminance, second_luminance)
        darker = min(first_luminance, second_luminance)
        return (lighter + 0.05) / (darker + 0.05)

    assert QApplication.testAttribute(Qt.AA_EnableHighDpiScaling)
    assert QApplication.testAttribute(Qt.AA_UseHighDpiPixmaps)
    assert not window.isMaximized()
    assert window.minimumWidth() <= 1000
    assert window.minimumHeight() <= 680

    assert window.tabs.count() == 3
    assert [window.tabs.tabText(i) for i in range(3)] == ['Predict', 'Label', 'Train']
    assert isinstance(window.predict_tab, QScrollArea)
    assert not isinstance(window.mser_tab, QScrollArea)
    assert isinstance(window.mser_scroll, QScrollArea)
    assert isinstance(window.training_tab, QScrollArea)

    ordinary_widgets = [
        window.tabs.tabBar(),
        window.image_line,
        window.checkpoint_line,
        window.predict_button,
        window.segment_button,
        window.train_model_button,
        window.dark_mode_check,
        window.roi_table,
        window.roi_table.horizontalHeader(),
        window.output_box,
        ]
    for widget in ordinary_widgets:
        info = QFontInfo(widget.font())
        assert info.family().casefold() == 'mononoki'
        assert widget.font().weight() == QFont.Normal
        assert info.weight() == QFont.Normal
        assert info.styleName().casefold() == 'regular'

    assert window.font().pointSizeF() == GUI_FONT_SIZE
    assert all(
        widget.font().pointSizeF() == GUI_FONT_SIZE
        for widget in ordinary_widgets
        )
    assert window.ax.texts
    assert min(
        text.get_fontsize()
        for text in window.ax.texts
        ) == GUI_FONT_SIZE
    assert QToolTip.font().pointSizeF() == GUI_FONT_SIZE

    headings = [
        window.state_label,
        *[
            label
            for label in window.findChildren(QLabel)
            if label.objectName() == 'sectionHeading'
            ],
        ]
    assert headings
    for heading in headings:
        info = QFontInfo(heading.font())
        assert info.family().casefold() == 'mononoki'
        assert heading.font().weight() == QFont.Bold
        assert info.weight() == QFont.Bold
        assert info.styleName().casefold() == 'bold'

    for widget in [window, *window.findChildren(QWidget)]:
        assert widget.font().weight() in {QFont.Normal, QFont.Bold}
        assert QFontInfo(widget.font()).weight() in {QFont.Normal, QFont.Bold}
    assert 'font-weight:' not in window.styleSheet()
    assert 'font-size:' not in window.styleSheet()

    assert window.dark_mode
    assert window.dark_mode_check.isChecked()
    theme = window._theme()
    assert theme['canvas'] == '#1a1a1c'
    assert theme['surface'] == '#232326'
    assert theme['surface_alt'] == '#2b2b2f'
    assert theme['text'] == '#e8e6e0'
    assert theme['muted'] == '#aaa7a0'
    assert theme['border'] == '#3a3a3f'

    palette = window.palette()
    assert palette.color(QPalette.Window).name() == theme['window']
    assert palette.color(QPalette.Button).name() == theme['surface']
    assert palette.color(QPalette.Text).name() == theme['text']
    assert contrast(
        palette.color(QPalette.Text),
        palette.color(QPalette.Base),
        ) >= 7
    assert contrast(
        palette.color(QPalette.Disabled, QPalette.ButtonText),
        palette.color(QPalette.Disabled, QPalette.Button),
        ) >= 3
    assert QColor(theme['border_strong']) != QColor(theme['surface'])
    assert 'QPushButton:focus' in window.styleSheet()
    assert 'QCheckBox:disabled' in window.styleSheet()

    assert window.interface_font_button.text() == 'Aa'
    assert window.interface_font_button.accessibleName() == 'interface text size'
    assert [
        action.text()
        for action in window.interface_font_menu.actions()
        ] == ['9 pt', '10 pt', '11 pt', '12 pt', '13 pt']
    assert window.interface_font_actions[GUI_FONT_SIZE].isChecked()
    window.interface_font_actions[11].trigger()
    QApplication.instance().processEvents()
    assert all(
        button.font().pointSizeF() == 11.0
        for button in window.findChildren(QAbstractButton)
        )
    assert all(widget.font().pointSizeF() == 11.0 for widget in ordinary_widgets)
    assert all(heading.font().pointSizeF() == 11.0 for heading in headings)
    assert QToolTip.font().pointSizeF() == 11.0
    assert all(
        action.font().pointSizeF() == 11.0
        for action in window.interface_font_menu.actions()
        )
    assert all(
        window.roi_table.horizontalHeaderItem(column).font().pointSizeF() == 11.0
        for column in range(window.roi_table.columnCount())
        )
    assert min(text.get_fontsize() for text in window.ax.texts) == 11.0
    assert window.interface_font_actions[11].isChecked()
    window.interface_font_actions[GUI_FONT_SIZE].trigger()
    QApplication.instance().processEvents()

    assert window.activity_splitter.orientation() == Qt.Vertical
    assert window.activity_splitter.isCollapsible(1)
    assert window.activity_splitter.sizes()[1] == 0
    assert window.rebuild_rois_button.isHidden()
    assert window.rebuild_rois_button.text() == 'rebuild ROIs'
    assert window.rebuild_rois_button.toolTip() == (
        'rebuild all ROIs from the cached confidence map using the '
        'threshold and minimum area; prediction does not run again; '
        'this replaces the current ROIs; Undo restores them'
        )
    assert not window.findChildren(QProgressBar)
    assert window.curate_buttons['delete'].property('role') == 'dangerQuiet'
    assert [
        window.predict_button.text(),
        window.segment_button.text(),
        window.train_model_button.text(),
        ] == ['predict', 'SEGMENT', 'TRAIN MODEL']
    assert window.threshold_spin.accessibleName() == 'prediction threshold'
    assert window.threshold_spin.accessibleDescription() == (
        'higher values retain fewer, more confident ROIs'
        )
    assert window.min_size_spin.accessibleName() == 'minimum ROI area (pixels)'
    assert window.min_size_spin.accessibleDescription() == (
        'discard connected components below this area'
        )

    window.resize(1280, 820)
    window.show()
    for _ in range(3):
        QApplication.instance().processEvents()

    wide_curation_bar = window.curation_bar
    button_position = lambda button: button.mapTo(
        wide_curation_bar,
        QPoint(0, 0),
        )
    assert window._curation_layout_is_wide
    assert button_position(window.fix_selected_button).y() == (
        button_position(window.segment_load_roi_button).y()
        )
    assert button_position(window.fix_selected_button).y() == (
        button_position(window.curate_buttons['save_roi']).y()
        )
    assert button_position(window.curate_buttons['zoom_out']).y() > (
        button_position(window.fix_selected_button).y()
        )

    window.tabs.setCurrentIndex(0)
    for _ in range(2):
        QApplication.instance().processEvents()
    predict_upper = window.controls_splitter.sizes()[0]
    predict_table_height = window.roi_table.height()
    assert window.predict_tab.verticalScrollBar().maximum() == 0

    window.tabs.setCurrentIndex(1)
    for _ in range(2):
        QApplication.instance().processEvents()
    label_upper = window.controls_splitter.sizes()[0]
    label_table_height = window.roi_table.height()
    assert label_upper - predict_upper >= 100
    assert predict_table_height - label_table_height >= 100
    assert window.mser_scroll.verticalScrollBar().maximum() > 0
    assert window.roi_table.height() >= 140
    label_field_right_edges = {
        widget.mapTo(window.mser_tab_content, QPoint(widget.width(), 0)).x()
        for widget in window.segment_param_widgets.values()
        }
    assert len(label_field_right_edges) == 1
    label_heading = next(
        label
        for label in window.mser_tab.findChildren(QLabel)
        if label.objectName() == 'sectionHeading' and label.text() == 'preprocessing'
        )
    assert label_heading.height() > label_heading.fontMetrics().height()
    assert 'QWidget#labelTab QLabel#sectionHeading' in window.styleSheet()
    footer_position = window.mser_action_bar.mapTo(window.mser_tab, QPoint(0, 0))
    segment_position = window.segment_button.mapTo(window.mser_tab, QPoint(0, 0))
    assert footer_position.y() + window.mser_action_bar.height() <= window.mser_tab.height()
    assert segment_position.y() >= footer_position.y()
    window.mser_scroll.verticalScrollBar().setValue(
        window.mser_scroll.verticalScrollBar().maximum()
        )
    QApplication.instance().processEvents()
    assert window.mser_action_bar.mapTo(window.mser_tab, QPoint(0, 0)) == footer_position
    assert window.segment_button.mapTo(window.mser_tab, QPoint(0, 0)) == segment_position

    window.tabs.setCurrentIndex(2)
    for _ in range(2):
        QApplication.instance().processEvents()
    assert window.training_tab.verticalScrollBar().maximum() > 0
    assert window.roi_table.height() >= 140

    # test fontsize with resizing, 13 Aug 2026
    for point_size in GUI_FONT_SIZES:
        window.interface_font_actions[point_size].trigger()
        window.resize(1000, 680)
        window.tabs.setCurrentIndex(0)
        for _ in range(3):
            QApplication.instance().processEvents()

        assert window.size().width() == 1000
        assert window.size().height() == 680
        assert window.minimumSizeHint().width() <= 1000
        assert window.minimumSizeHint().height() <= 680
        assert window.roi_table.isVisible()
        assert window.roi_table.height() >= 140
        assert window.black_slider.visibleRegion().boundingRect().width() > 0
        assert window.white_slider.visibleRegion().boundingRect().width() > 0
        assert window.predict_tab.verticalScrollBar().maximum() == 0
        for widget in [
            window.interface_font_button,
            window.roi_overlay_check,
            window.dark_mode_check,
            ]:
            assert widget.visibleRegion().boundingRect().width() == widget.width()
            assert widget.width() >= widget.sizeHint().width()

        for button in [
            window.fix_selected_button,
            window.segment_load_roi_button,
            *window.curate_buttons.values(),
            ]:
            assert button.width() >= button.sizeHint().width()

        window.tabs.setCurrentIndex(1)
        for _ in range(2):
            QApplication.instance().processEvents()
        assert window.mser_scroll.verticalScrollBar().maximum() > 0
        assert window.roi_table.height() >= 140

        window.tabs.setCurrentIndex(2)
        for _ in range(2):
            QApplication.instance().processEvents()
        assert window.training_tab.verticalScrollBar().maximum() > 0
        assert window.roi_table.height() >= 140

    window.interface_font_actions[GUI_FONT_SIZE].trigger()
    window.tabs.setCurrentIndex(0)
    for _ in range(3):
        QApplication.instance().processEvents()

    assert window.state_label.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored
    assert window.model_label.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored

    curation_bar = window.curation_bar
    curation_buttons = [
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
        ]
    for button in curation_buttons:
        top_left = button.mapTo(curation_bar, QPoint(0, 0))
        assert top_left.x() >= 0
        assert top_left.x() + button.width() <= curation_bar.width()
    assert not window._curation_layout_is_wide
    assert curation_bar.height() < 90
    row_positions = {
        button.mapTo(curation_bar, QPoint(0, 0)).y()
        for button in curation_buttons
        }
    assert len(row_positions) == 2

    divider = window.curation_history_divider
    divider_position = divider.mapTo(curation_bar, QPoint(0, 0))
    delete_position = window.curate_buttons['delete'].mapTo(
        curation_bar,
        QPoint(0, 0),
        )
    undo_position = window.curate_buttons['undo'].mapTo(
        curation_bar,
        QPoint(0, 0),
        )
    assert divider.width() == 1
    assert divider.height() == 20
    assert delete_position.x() < divider_position.x() < undo_position.x()
    import_position = window.segment_load_roi_button.mapTo(
        curation_bar,
        QPoint(0, 0),
        )
    export_position = window.curate_buttons['save_roi'].mapTo(
        curation_bar,
        QPoint(0, 0),
        )
    assert window.segment_load_roi_button.property('role') == 'secondary'
    assert window.curate_buttons['save_roi'].property('role') == 'secondary'
    assert import_position.y() == export_position.y()
    assert import_position.x() < export_position.x()
    assert import_position.y() == window.curate_buttons['zoom_out'].mapTo(
        curation_bar,
        QPoint(0, 0),
        ).y()

    reset = window.reset_display_button
    assert reset.text() == 'reset'
    assert reset.accessibleName() == 'reset display'
    for button in [
        reset,
        window.image_view_button,
        window.confidence_view_button,
        ]:
        assert button.width() >= button.sizeHint().width()

    tab_bar = window.tabs.tabBar()
    assert all(tab_bar.tabRect(i).right() < tab_bar.width() for i in range(3))
    assert QLabel.text(window.canvas_hint) == window.canvas_hint.text()
    visible = window.canvas.visibleRegion().boundingRect()
    assert visible.width() == window.canvas.width()
    assert visible.height() == window.canvas.height()

    window.tabs.setCurrentIndex(1)
    for _ in range(2):
        QApplication.instance().processEvents()
    assert window.segment_load_roi_button.isVisible()
    window.tabs.setCurrentIndex(2)
    for _ in range(2):
        QApplication.instance().processEvents()
    assert window.training_tab.verticalScrollBar().maximum() > 0

    window.dark_mode_check.setChecked(False)
    QApplication.instance().processEvents()
    light_theme = window._theme()
    assert light_theme['window'] == '#f0eee9'
    assert light_theme['surface'] == '#faf9f6'
    assert light_theme['canvas'] == '#1a1a1c'


def interface_font_persistence(window):
    from PyQt5.QtWidgets import QApplication, QToolTip

    from fibre_sight.fibre_sight_workbench_gui import FibreSightWorkbench

    window.interface_font_actions[10].trigger()
    QApplication.instance().processEvents()
    assert window.settings.value('interface/font_size', type=float) == 10.0

    restored = FibreSightWorkbench(settings=window.settings)
    assert restored.interface_font_size == 10.0
    assert restored.interface_font_actions[10].isChecked()
    assert restored.predict_button.font().pointSizeF() == 10.0
    assert QToolTip.font().pointSizeF() == 10.0
    assert all(
        restored.roi_table.horizontalHeaderItem(column).font().pointSizeF() == 10.0
        for column in range(restored.roi_table.columnCount())
        )
    restored.close()


def image_loading(window):
    import numpy as np
    from PyQt5.QtWidgets import QFileDialog, QPushButton

    import fibre_sight.fibre_sight_workbench_gui as gui

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

    window.labelled[1:3, 1:3] = 1
    window.update_roi_dict()
    window.probability = np.ones_like(window.ref_image, dtype=np.float32)
    window.set_display_range(12.3, 78.9, schedule_redraw=False)
    window.refresh_status()
    window.set_display_mode('confidence')

    image_browse = next(
        button
        for button in window.findChildren(QPushButton)
        if button.accessibleName() == 'browse for channel-2 image'
        )
    with mock.patch.object(
            QFileDialog,
            'getOpenFileName',
            return_value=(str(SECOND_IMAGE), 'NumPy arrays (*.npy)'),
            ):
        image_browse.click()

    assert window.image_path == SECOND_IMAGE
    assert window.image_matches(SECOND_IMAGE)
    assert window.roi_dict == {}
    assert window.probability is None
    assert window.display_mode == 'image'
    assert window.display_black == 1.0
    assert window.display_white == 99.7

    window.image_line.setText(str(THIRD_IMAGE))
    window.image_line.editingFinished.emit()
    assert window.image_path == THIRD_IMAGE
    assert window.image_matches(THIRD_IMAGE)

    window.image_line.setText('missing_ref_mat_ch2.npy')
    assert not window.segment_button.isEnabled()
    assert not window.segment_load_roi_button.isEnabled()
    assert not window.curate_buttons['save_roi'].isEnabled()
    with (
        mock.patch.object(gui, 'run_mser_segmentation') as segment,
        mock.patch.object(gui, 'save_roi_dict') as export,
    ):
        window.segment_rois()
        window.save_roi_file()
    segment.assert_not_called()
    export.assert_not_called()


def global_resources(window):
    import numpy as np
    from PyQt5.QtWidgets import QApplication, QLabel, QWidget

    from fibre_sight.fibre_sight_workbench_gui import PathLineEdit

    def has_ancestor(widget, object_name):
        parent = widget.parentWidget()
        while parent is not None:
            if parent.objectName() == object_name:
                return True
            parent = parent.parentWidget()
        return False

    labels = window.findChildren(QLabel)
    for text in ('channel-2 image', 'trained model'):
        assert sum(label.text() == text for label in labels) == 1
    assert not any(label.text() == 'compute device' for label in labels)

    assert has_ancestor(window.image_line, 'resourcesPanel')
    assert has_ancestor(window.checkpoint_line, 'resourcesPanel')
    assert window.image_line not in window.predict_tab_content.findChildren(PathLineEdit)
    assert window.checkpoint_line not in window.predict_tab_content.findChildren(PathLineEdit)
    assert window.selected_device() == 'auto'
    assert not hasattr(window, 'device_combo')

    window.checkpoint_line.clear()
    window.refresh_status()
    assert window.model_label.text() == 'model: not selected'

    with tempfile.TemporaryDirectory(prefix='fibre sight model ') as temp_dir:
        checkpoint = Path(temp_dir) / 'best.pt'
        checkpoint.touch()
        window.checkpoint_line.setText(str(checkpoint))
        window.predictor = None
        window.refresh_status()
        assert window.model_label.text() == 'model: best.pt · awaiting image'

        missing_checkpoint = Path(temp_dir) / 'missing.pt'
        window.checkpoint_line.setText(str(missing_checkpoint))
        window.refresh_status()
        assert window.model_label.text() == 'model: missing.pt · checkpoint missing'

        image_path = Path(temp_dir) / 'session_ref_mat_ch2.npy'
        np.save(image_path, np.zeros((4, 4), dtype=np.float32))
        window.checkpoint_line.setText(str(checkpoint))
        window.image_line.setText(str(image_path))
        window.load_channel_image()
        window.refresh_status()
        assert window.model_label.text() == 'model: best.pt · ready to predict'

        window.predictor = SimpleNamespace(
            checkpoint_path=checkpoint,
            device='cpu',
            )
        window.refresh_status()
        assert window.model_label.text() == 'model: best.pt · cpu'

    window.resize(1000, 680)
    window.show()
    QApplication.instance().processEvents()
    assert window.model_label.isVisible()
    assert window.model_separator.text() == '·'
    assert not hasattr(window, 'device_label')
    stage_header = window.findChild(QWidget, 'stageHeader')
    assert stage_header is not None
    assert all(
        not label.text().startswith('device:')
        for label in stage_header.findChildren(QLabel)
        )


def probability_invalidation(window):
    import numpy as np

    _seed_canvas(window, include_probability=True)
    checkpoint_path = window.checkpoint_line.text()
    image_path = window.image_line.text()
    changes = [
        lambda: window.checkpoint_line.setText(checkpoint_path + '.changed'),
        lambda: window.image_line.setText(str(THIRD_IMAGE)),
        ]
    for change in changes:
        window.probability = np.zeros_like(window.ref_image, dtype=np.float32)
        window.refresh_status()
        window.set_display_mode('confidence')
        assert window.display_mode == 'confidence'
        assert window.rebuild_rois_button.isEnabled()
        assert window.confidence_view_button.isEnabled()

        change()
        assert window.probability is None
        assert window.display_mode == 'image'
        assert window.image_view_button.isChecked()
        assert not window.confidence_view_button.isEnabled()
        assert not window.rebuild_rois_button.isEnabled()
        assert window.rebuild_rois_button.isHidden()

        window.checkpoint_line.setText(checkpoint_path)
        window.image_line.setText(image_path)


def undo_after_rebuild_rois(window):
    import numpy as np

    window.ref_image = np.zeros((8, 8), dtype=np.float32)
    window.labelled = np.zeros((8, 8), dtype=np.int32)
    window.labelled[1:3, 1:3] = 1
    window.update_roi_dict()
    window.selected = {1}
    window.fixed_ids = {1}
    window.probability = np.zeros((8, 8), dtype=np.float32)
    window.probability[4:6, 4:6] = 1
    retained_probability = window.probability
    window.min_size_spin.setValue(1)

    undo_count = len(window.undo_stack)
    window.rebuild_rois_from_probability()
    assert len(window.undo_stack) == undo_count + 1
    assert window.probability is retained_probability
    assert window.confidence_view_button.isEnabled()
    assert window.labelled[4, 4] > 0
    assert window.labelled[1, 1] == 0
    assert not window.selected
    assert not window.fixed_ids

    window.undo()
    assert len(window.undo_stack) == undo_count
    assert window.labelled[1, 1] > 0
    assert window.labelled[4, 4] == 0
    assert window.selected == {1}
    assert window.fixed_ids == {1}


def training_handoff(window):
    from PyQt5.QtCore import QProcess

    class DummyPredictor:
        def __init__(self, path):
            self.checkpoint_path = path
            self.device = 'cpu'

    with tempfile.TemporaryDirectory(prefix='fibre sight training ') as temp_dir:
        temp_dir = Path(temp_dir)
        previous = temp_dir / 'previous.pt'
        new_checkpoint = temp_dir / 'best.pt'
        previous.touch()
        new_checkpoint.touch()

        previous_predictor = DummyPredictor(previous)
        window.checkpoint_line.setText(str(previous))
        window.predictor = previous_predictor
        window.last_saved_model_path = previous
        window.current_process_name = 'train'
        window.pending_checkpoint_path = new_checkpoint
        window._process_was_stopped = False
        window.process = None
        window.process_finished(0, QProcess.NormalExit)
        assert Path(window.checkpoint_line.text()) == new_checkpoint
        assert window.last_saved_model_path == new_checkpoint
        assert window.predictor is None
        assert window.model_label.text() == 'model: best.pt · awaiting image'

        same_path_predictor = DummyPredictor(new_checkpoint)
        window.predictor = same_path_predictor
        window.current_process_name = 'train'
        window.pending_checkpoint_path = new_checkpoint
        window._process_was_stopped = False
        window.process_finished(0, QProcess.NormalExit)
        assert Path(window.checkpoint_line.text()) == new_checkpoint
        assert window.predictor is None

        for stopped, exit_code, exit_status in (
                (False, 2, QProcess.CrashExit),
                (True, 0, QProcess.NormalExit),
                ):
            preserved_predictor = DummyPredictor(previous)
            window.checkpoint_line.setText(str(previous))
            window.predictor = preserved_predictor
            window.last_saved_model_path = previous
            window.current_process_name = 'train'
            window.pending_checkpoint_path = new_checkpoint
            window._process_was_stopped = stopped
            window.process = None
            window.process_finished(exit_code, exit_status)
            assert Path(window.checkpoint_line.text()) == previous
            assert window.last_saved_model_path == previous
            assert window.predictor is preserved_predictor


def process_streaming(window):
    from PyQt5.QtCore import QProcess
    from PyQt5.QtWidgets import QProgressBar

    import fibre_sight.fibre_sight_workbench_gui as gui

    class FakeSignal:
        def connect(self, slot):
            self.slot = slot

    class FakeProcess:
        NotRunning = QProcess.NotRunning

        def __init__(self, _parent):
            self.readyReadStandardOutput = FakeSignal()
            self.readyReadStandardError = FakeSignal()
            self.finished = FakeSignal()
            self.program = None
            self.arguments = None
            self.stdout_chunk = b''
            self.stderr_chunk = b''

        def state(self):
            return self.NotRunning

        def setWorkingDirectory(self, path):
            self.working_directory = path

        def setProgram(self, program):
            self.program = program

        def setArguments(self, arguments):
            self.arguments = list(arguments)

        def readAllStandardOutput(self):
            chunk = self.stdout_chunk
            self.stdout_chunk = b''
            return chunk

        def readAllStandardError(self):
            chunk = self.stderr_chunk
            self.stderr_chunk = b''
            return chunk

        def start(self):
            self.started = True

    with (
        tempfile.TemporaryDirectory(prefix='fibre sight process ') as temp_dir,
        mock.patch.object(gui, 'WORKSPACE_ROOT', Path(temp_dir)),
        mock.patch.object(gui, 'QProcess', FakeProcess),
    ):
        assert window.start_process('probe', 'train_unet', ['--config', 'recipe.yaml'])
        process = window.process
        assert process.program == sys.executable
        assert process.arguments[:3] == ['-u', '-m', 'fibre_sight.train_unet']

        window.output_box.clear()
        process.stdout_chunk = b'epoch 001 | train'
        window.read_process_stdout()
        process.stderr_chunk = b'warn'
        window.read_process_stderr()
        assert window.output_box.toPlainText() == ''

        process.stdout_chunk = b' loss 0.5000\nnext'
        window.read_process_stdout()
        process.stderr_chunk = b'ing\n'
        window.read_process_stderr()
        text = window.output_box.toPlainText()
        assert text.count('epoch 001 | train loss 0.5000') == 1
        assert text.count('warning') == 1
        assert 'next' not in text

        window.flush_process_buffers()
        text = window.output_box.toPlainText()
        assert text.count('epoch 001 | train loss 0.5000') == 1
        assert text.count('warning') == 1
        assert text.count('next') == 1
        assert window._process_stdout_buffer == ''
        assert window._process_stderr_buffer == ''

        window.output_box.clear()
        process.stdout_chunk = b'epoch 002\r'
        window.read_process_stdout()
        assert window.output_box.toPlainText() == ''
        process.stdout_chunk = b'\n'
        window.read_process_stdout()
        assert window.output_box.toPlainText().splitlines() == ['epoch 002']

        assert not window.findChildren(QProgressBar)
        window.process = None


def display_range(window):
    import numpy as np
    from PyQt5.QtTest import QSignalSpy, QTest

    from fibre_sight.fibre_sight_workbench_gui import (
        DISPLAY_BLACK_DEFAULT,
        DISPLAY_MIN_GAP,
        DISPLAY_REDRAW_MS,
        DISPLAY_WHITE_DEFAULT,
        )
    from fibre_sight.gui_canvas import normalise_for_display

    finite = np.arange(5, dtype=np.float64)
    finite_before = finite.copy()
    np.testing.assert_allclose(
        normalise_for_display(finite, 0, 100),
        finite.astype(np.float32) / 4,
        )
    np.testing.assert_array_equal(finite, finite_before)

    integer = np.arange(9, dtype=np.uint16).reshape(3, 3)
    integer_before = integer.copy()
    integer_out = normalise_for_display(integer)
    assert integer_out.dtype == np.float32
    assert np.all(np.isfinite(integer_out))
    np.testing.assert_array_equal(integer, integer_before)

    constant = np.full((3, 4), 7, dtype=np.float32)
    np.testing.assert_array_equal(
        normalise_for_display(constant),
        np.zeros_like(constant),
        )

    mixed = np.array([[np.nan, 0], [5, 10]], dtype=np.float32)
    mixed_before = mixed.copy()
    mixed_out = normalise_for_display(mixed, 0, 100)
    assert np.all(np.isfinite(mixed_out))
    assert mixed_out[0, 0] == 0
    assert mixed_out[0, 1] == 0
    assert mixed_out[1, 1] == 1
    np.testing.assert_equal(mixed, mixed_before)

    all_nan = np.full((2, 3), np.nan, dtype=np.float32)
    np.testing.assert_array_equal(
        normalise_for_display(all_nan),
        np.zeros_like(all_nan),
        )

    source = _seed_canvas(window, include_probability=True)
    source_before = source.copy()
    labelled_before = window.labelled.copy()
    probability_before = window.probability.copy()
    roi_before = {
        roi_id: {
            'xpix': roi['xpix'].copy(),
            'ypix': roi['ypix'].copy(),
            }
        for roi_id, roi in window.roi_dict.items()
        }
    selected_before = set(window.selected)

    window.set_display_range(
        99.9,
        window.display_white,
        changed='black',
        schedule_redraw=False,
        )
    assert window.display_white - window.display_black >= DISPLAY_MIN_GAP
    window.set_display_range(
        0.5,
        0.2,
        changed='white',
        schedule_redraw=False,
        )
    assert window.display_white - window.display_black >= DISPLAY_MIN_GAP

    assert window.display_redraw_timer.isSingleShot()
    assert window.display_redraw_timer.interval() == DISPLAY_REDRAW_MS
    timeout_spy = QSignalSpy(window.display_redraw_timer.timeout)
    window.black_slider.setValue(20)
    window.black_slider.setValue(21)
    window.black_slider.setValue(22)
    QTest.qWait(15)
    assert len(timeout_spy) == 0
    QTest.qWait(DISPLAY_REDRAW_MS + 15)
    assert len(timeout_spy) == 1

    np.testing.assert_array_equal(window.ref_image, source_before)
    np.testing.assert_array_equal(window.labelled, labelled_before)
    np.testing.assert_array_equal(window.probability, probability_before)
    assert window.selected == selected_before
    for roi_id, roi in roi_before.items():
        np.testing.assert_array_equal(window.roi_dict[roi_id]['xpix'], roi['xpix'])
        np.testing.assert_array_equal(window.roi_dict[roi_id]['ypix'], roi['ypix'])

    window.reset_display_button.click()
    assert window.display_black == DISPLAY_BLACK_DEFAULT
    assert window.display_white == DISPLAY_WHITE_DEFAULT
    assert window.black_value.value() == DISPLAY_BLACK_DEFAULT
    assert window.white_value.value() == DISPLAY_WHITE_DEFAULT

    window.set_display_range(12.3, 78.9, schedule_redraw=False)
    with tempfile.TemporaryDirectory(prefix='fibre sight display ') as temp_dir:
        image_path = Path(temp_dir) / 'new_ref_mat_ch2.npy'
        np.save(image_path, np.ones((12, 14), dtype=np.float32))
        window.image_line.setText(str(image_path))
        window.load_channel_image()
    assert window.display_black == DISPLAY_BLACK_DEFAULT
    assert window.display_white == DISPLAY_WHITE_DEFAULT


def confidence_view(window):
    import numpy as np

    from fibre_sight.fibre_sight_workbench_gui import (
        SELECTION_COLOUR_WIDTH,
        SELECTION_INNER_WIDTH,
        SELECTION_OUTER_WIDTH,
        )

    _seed_canvas(window, include_probability=True)
    assert window.display_mode == 'image'
    assert window.confidence_view_button.isEnabled()
    assert window.confidence_view_button.toolTip() == (
        'show per-pixel model confidence from 0 to 1'
        )

    window.canvas.zoom_in(3)
    limits_before = window.get_current_view()
    selection_before = set(window.selected)
    probability = window.probability
    window.set_display_mode('confidence')

    assert window.display_mode == 'confidence'
    assert window.confidence_view_button.isChecked()
    assert not window.image_view_button.isChecked()
    assert window.selected == selection_before
    _assert_limits_equal(window.get_current_view(), limits_before)
    assert len(window.ax.images) == 1
    np.testing.assert_array_equal(window.ax.images[0].get_array(), probability)
    assert window.ax.images[0].get_clim() == (0.0, 1.0)
    assert window.ax.images[0].get_cmap().name == 'gray'
    assert len(window.ax.collections) == 5

    linewidths = [
        float(collection.get_linewidths()[0])
        for collection in window.ax.collections
        ]
    assert SELECTION_OUTER_WIDTH in linewidths
    assert SELECTION_INNER_WIDTH in linewidths
    assert SELECTION_COLOUR_WIDTH in linewidths
    assert 1.8 in linewidths
    assert any(
        collection.get_linestyles()[0][1] is not None
        for collection in window.ax.collections
        )

    for widget in (
        window.black_slider,
        window.white_slider,
        window.black_value,
        window.white_value,
        window.reset_display_button,
    ):
        assert not widget.isEnabled()

    retained_probability = window.probability
    window.min_size_spin.setValue(1)
    window.rebuild_rois_from_probability()
    assert window.probability is retained_probability
    assert window.confidence_view_button.isEnabled()
    assert window.display_mode == 'confidence'

    window.checkpoint_line.setText(window.checkpoint_line.text() + '.changed')
    assert window.probability is None
    assert window.display_mode == 'image'
    assert window.image_view_button.isChecked()
    assert not window.confidence_view_button.isEnabled()
    assert window.black_slider.isEnabled()
    assert window.white_slider.isEnabled()

    with tempfile.TemporaryDirectory(prefix='fibre sight prediction ') as temp_dir:
        temp_dir = Path(temp_dir)
        image_path = temp_dir / 'session_ref_mat_ch2.npy'
        checkpoint_path = temp_dir / 'best.pt'
        np.save(image_path, np.arange(120, dtype=np.float32).reshape(10, 12))
        checkpoint_path.touch()
        window.image_line.setText(str(image_path))
        window.load_channel_image()
        window.checkpoint_line.setText(str(checkpoint_path))

        labelled = np.zeros_like(window.ref_image, dtype=np.int32)
        labelled[2:5, 3:7] = 1
        ypix, xpix = np.nonzero(labelled == 1)
        predicted_probability = np.linspace(
            0,
            1,
            labelled.size,
            dtype=np.float32,
            ).reshape(labelled.shape)
        prediction = SimpleNamespace(
            roi_dict={1: {'xpix': xpix, 'ypix': ypix}},
            labelled=labelled,
            probability=predicted_probability,
            threshold=0.25,
            min_size=1,
            )
        window.predictor = SimpleNamespace(
            checkpoint_path=checkpoint_path,
            device='cpu',
            threshold=0.25,
            min_size=1,
            tta=True,
            predict_image=lambda _image: prediction,
            )
        window.probability = np.ones_like(window.ref_image, dtype=np.float32)
        window.refresh_status()
        window.set_display_mode('confidence')
        window.predict_rois()

    assert window.display_mode == 'image'
    assert window.image_view_button.isChecked()
    assert not window.confidence_view_button.isChecked()
    assert window.black_slider.isEnabled()
    assert 'predicted 1 ROIs' in window.output_box.toPlainText()
    assert window.statusBar().currentMessage() == ''


def canvas_navigation(window):
    import numpy as np
    from PyQt5.QtCore import QEvent, QPoint, QPointF, Qt
    from PyQt5.QtGui import QKeySequence, QMouseEvent
    from PyQt5.QtWidgets import QApplication, QShortcut

    _seed_canvas(window, include_probability=False)
    window.resize(1000, 680)
    window.show()
    QApplication.instance().processEvents()
    window.canvas.draw()

    window.selected.clear()
    window.on_click(_click_event(window, 2, 2, Qt.NoModifier))
    assert window.selected == {1}
    window.on_click(_click_event(window, 60, 36, Qt.ShiftModifier))
    assert window.selected == {1, 2}

    window.canvas.zoom_in(3)
    limits_before_drag = window.get_current_view()
    selected_before_drag = set(window.selected)
    start = _data_point(window, 50, 40)
    end = QPoint(start)
    end.setX(start.x() + QApplication.startDragDistance() + 35)
    press = QMouseEvent(
        QEvent.MouseButtonPress,
        QPointF(start),
        Qt.LeftButton,
        Qt.LeftButton,
        Qt.NoModifier,
        )
    move = QMouseEvent(
        QEvent.MouseMove,
        QPointF(end),
        Qt.NoButton,
        Qt.LeftButton,
        Qt.NoModifier,
        )
    release = QMouseEvent(
        QEvent.MouseButtonRelease,
        QPointF(end),
        Qt.LeftButton,
        Qt.NoButton,
        Qt.NoModifier,
        )
    QApplication.sendEvent(window.canvas, press)
    QApplication.sendEvent(window.canvas, move)
    assert window.canvas.cursor().shape() == Qt.ClosedHandCursor
    QApplication.sendEvent(window.canvas, release)
    assert window.canvas.cursor().shape() == Qt.CrossCursor
    assert window.selected == selected_before_drag
    assert not window.canvas.release_was_dragged
    assert not np.allclose(
        window.get_current_view()[0],
        limits_before_drag[0],
        )

    image_limits = window.canvas._image_limits()
    assert image_limits is not None
    full_x_span = _span(image_limits[0])
    full_y_span = _span(image_limits[1])
    window.canvas.fit_to_image()
    for _ in range(100):
        window.canvas.zoom_in()
    assert np.isclose(_span(window.ax.get_xlim()), full_x_span / 32)
    assert np.isclose(_span(window.ax.get_ylim()), full_y_span / 32)
    for _ in range(100):
        window.canvas.zoom_out()
    assert np.isclose(_span(window.ax.get_xlim()), full_x_span)
    assert np.isclose(_span(window.ax.get_ylim()), full_y_span)

    shortcuts = window.findChildren(QShortcut)

    def shortcut(sequence):
        return next(
            item
            for item in shortcuts
            if item.key() == QKeySequence(sequence)
            )

    shortcut('+').activated.emit()
    enlarged_span = _span(window.ax.get_xlim())
    assert enlarged_span < full_x_span
    shortcut('-').activated.emit()
    assert _span(window.ax.get_xlim()) > enlarged_span
    shortcut('+').activated.emit()
    shortcut('0').activated.emit()
    assert np.isclose(_span(window.ax.get_xlim()), full_x_span)
    assert np.isclose(_span(window.ax.get_ylim()), full_y_span)
    assert window.canvas.accessibleDescription() == (
        'click ROI to select · drag to pan · scroll or ± to zoom · '
        'Shift-click to add an ROI to the selection'
        )
    assert window.canvas_hint.text() == 'Shift-click to add'
    assert window.canvas.toolTip() == ''
    assert window.canvas_hint.toolTip() == ''


def roi_table(window):
    import numpy as np
    from PyQt5.QtCore import QItemSelectionModel, Qt
    from PyQt5.QtGui import QColor
    from PyQt5.QtTest import QSignalSpy, QTest
    from PyQt5.QtWidgets import QAbstractItemView, QApplication

    from fibre_sight.gui_canvas import generate_distinct_colours

    _seed_canvas(window, include_probability=False)
    window.labelled[60:64, 80:85] = 3
    window.update_roi_dict()
    window.plot_image()
    window.resize(1000, 680)
    window.show()
    QApplication.instance().processEvents()
    window.canvas.draw()

    assert window.roi_table.rowCount() == 3
    assert [window.roi_table.item(row, 1).text() for row in range(3)] == ['1', '2', '3']
    assert [window.roi_table.item(row, 2).text() for row in range(3)] == ['30', '80', '20']
    assert [window.roi_table.item(row, 3).text() for row in range(3)] == ['', 'yes', '']
    assert not window.roi_table.isSortingEnabled()
    assert window.roi_table.editTriggers() == QAbstractItemView.NoEditTriggers
    assert window.roi_table.selectionMode() == QAbstractItemView.ExtendedSelection
    assert (
        window.roi_table.horizontalHeaderItem(2).textAlignment() ==
        int(Qt.AlignRight | Qt.AlignVCenter)
        )

    expected_colour = generate_distinct_colours(3)[0]
    observed_colour = window.roi_table.item(0, 0).background().color().getRgbF()[:3]
    np.testing.assert_allclose(observed_colour, expected_colour, atol=1 / 255)
    swatch = window.roi_table.cellWidget(0, 0)
    assert swatch.text() == '■'
    assert QColor.fromRgbF(*expected_colour).name() in swatch.styleSheet()
    assert window.fix_selected_button.parentWidget().objectName() == 'curationBar'
    assert window.fix_selected_button.minimumWidth() >= 55
    assert window.fix_selected_button.isEnabled()
    assert window.fix_selected_button.text() == 'fix'
    window.fix_selected_button.click()
    assert window.fixed_ids == {1, 2}
    assert window.roi_table.item(0, 3).text() == 'yes'
    assert window.fix_selected_button.text() == 'unfix'
    window.fix_selected_button.click()
    assert window.fixed_ids == {2}
    assert window.roi_table.item(0, 3).text() == ''
    assert window.fix_selected_button.text() == 'fix'

    row_three_rect = window.roi_table.visualItemRect(window.roi_table.item(2, 1))
    QTest.mouseClick(
        window.roi_table.viewport(),
        Qt.LeftButton,
        Qt.NoModifier,
        row_three_rect.center(),
        )
    assert window.selected == {3}
    window.selected = {1}
    window.plot_image(preserve_view=True)
    assert window.roi_table.currentRow() == 0
    row_two_rect = window.roi_table.visualItemRect(window.roi_table.item(1, 1))
    QTest.mouseClick(
        window.roi_table.viewport(),
        Qt.LeftButton,
        Qt.ShiftModifier,
        row_two_rect.center(),
        )
    assert window.selected == {1, 2}

    window.canvas.zoom_in(4)
    view_before_selection = window.get_current_view()
    row_one_rect = window.roi_table.visualItemRect(window.roi_table.item(0, 1))
    QTest.mouseClick(
        window.roi_table.viewport(),
        Qt.LeftButton,
        Qt.NoModifier,
        row_one_rect.center(),
        )
    assert window.selected == {1}
    _assert_limits_equal(window.get_current_view(), view_before_selection)

    selection_model = window.roi_table.selectionModel()
    selection_model.select(
        window.roi_table.model().index(1, 0),
        QItemSelectionModel.Select | QItemSelectionModel.Rows,
        )
    assert window.selected == {1, 2}
    _assert_limits_equal(window.get_current_view(), view_before_selection)

    selection_spy = QSignalSpy(window.roi_table.itemSelectionChanged)
    window.refresh_roi_table()
    assert len(selection_spy) == 0

    row_two = 1
    span_before_centre = (
        _span(window.ax.get_xlim()),
        _span(window.ax.get_ylim()),
        )
    window.centre_roi_from_table(row_two, 1)
    span_after_centre = (
        _span(window.ax.get_xlim()),
        _span(window.ax.get_ylim()),
        )
    np.testing.assert_allclose(span_after_centre, span_before_centre)
    roi_two = window.roi_dict[2]
    expected_x = float(np.mean(roi_two['xpix']))
    expected_y = float(np.mean(roi_two['ypix']))
    assert np.isclose(sum(window.ax.get_xlim()) / 2, expected_x)
    assert np.isclose(sum(window.ax.get_ylim()) / 2, expected_y)

    window.centre_roi_from_table(0, 1)
    edge_x = float(np.mean(window.roi_dict[1]['xpix']))
    edge_y = float(np.mean(window.roi_dict[1]['ypix']))
    xlow, xhigh = sorted(window.ax.get_xlim())
    ylow, yhigh = sorted(window.ax.get_ylim())
    assert xlow <= edge_x <= xhigh
    assert ylow <= edge_y <= yhigh
    np.testing.assert_allclose(
        (_span(window.ax.get_xlim()), _span(window.ax.get_ylim())),
        span_before_centre,
        )


def roi_io_paths(window):
    import numpy as np
    from PyQt5.QtWidgets import QApplication, QFileDialog

    import fibre_sight.fibre_sight_workbench_gui as gui
    from fibre_sight.fibre_sight_workbench_gui import display_path

    window.resize(1000, 680)
    window.show()
    QApplication.instance().processEvents()

    with tempfile.TemporaryDirectory(prefix='fibre sight io ') as temp_dir:
        temp_dir = Path(temp_dir)
        image_path = temp_dir / 'session_ref_mat_ch2.npy'
        np.save(image_path, np.arange(120, dtype=np.float32).reshape(10, 12))
        window.image_line.setText(str(image_path))
        window.load_channel_image()
        window.labelled[2:5, 3:7] = 1
        window.update_roi_dict()
        window.plot_image()
        window.refresh_status()

        out_path = window.default_roi_path()
        assert str(out_path) in window.curate_buttons['save_roi'].toolTip()
        assert window.curate_buttons['save_roi'].isEnabled()
        assert window.segment_load_roi_button.isEnabled()
        assert (
            window.segment_load_roi_button.parentWidget() is
            window.curate_buttons['save_roi'].parentWidget()
            )
        assert (
            window.segment_load_roi_button.property('role') ==
            window.curate_buttons['save_roi'].property('role')
            )
        with (
            mock.patch.object(gui, 'save_roi_dict') as save_mock,
            mock.patch.object(
                QFileDialog,
                'getSaveFileName',
                side_effect=AssertionError('export must not open a picker'),
                ),
        ):
            window.curate_buttons['save_roi'].click()
        assert save_mock.call_count == 1
        saved_roi_dict, saved_path = save_mock.call_args.args
        assert saved_roi_dict is window.roi_dict
        assert saved_path == out_path
        assert 'exported ROIs to' in window.output_box.toPlainText()
        assert display_path(out_path) in window.output_box.toPlainText()

        roi_path = temp_dir / 'imported_ROI_dict.npy'
        roi_data = {
            7: {
                'xpix': np.array([1, 2, 2]),
                'ypix': np.array([1, 1, 2]),
                },
            }
        np.save(roi_path, roi_data)
        window.probability = np.ones_like(window.ref_image, dtype=np.float32)
        window.set_display_mode('confidence')
        with mock.patch.object(
                QFileDialog,
                'getOpenFileName',
                return_value=(str(roi_path), 'NumPy dict (*.npy)'),
                ) as open_mock:
            window.segment_load_roi_button.click()
        assert open_mock.call_args.args[1] == 'import ROIs'
        assert len(window.roi_dict) == 1
        assert window.probability is None
        assert window.display_mode == 'image'

    workspace_path = gui.WORKSPACE_ROOT / 'output' / 'nested' / 'result.npy'
    assert display_path(workspace_path) == str(Path('output', 'nested', 'result.npy'))

    home_path = Path.home() / 'private' / 'long-session-name' / 'result.npy'
    shortened = display_path(home_path)
    assert shortened == str(Path('…', 'private', 'long-session-name', 'result.npy'))
    assert str(Path.home()) not in shortened

    window.image_line.resize(150, window.image_line.height())
    window.image_line.setText(str(home_path))
    window.checkpoint_line.setFocus()
    QApplication.instance().processEvents()
    assert window.image_line.toolTip() == str(home_path)
    assert window.image_line._overlay.isVisible()
    assert window.image_line._overlay.text() != str(home_path)
    assert '…' in window.image_line._overlay.text()

    window.image_line.setFocus()
    QApplication.instance().processEvents()
    assert not window.image_line._overlay.isVisible()
    assert window.image_line.text() == str(home_path)

    window.print_log(f'external result: {home_path}')
    log_text = window.output_box.toPlainText()
    assert str(Path.home()) not in log_text
    assert display_path(home_path) in log_text


PROBES = {
    'layout': layout,
    'interface_font_persistence': interface_font_persistence,
    'image_loading': image_loading,
    'global_resources': global_resources,
    'probability_invalidation': probability_invalidation,
    'undo_after_rebuild_rois': undo_after_rebuild_rois,
    'training_handoff': training_handoff,
    'process_streaming': process_streaming,
    'display_range': display_range,
    'confidence_view': confidence_view,
    'canvas_navigation': canvas_navigation,
    'roi_table': roi_table,
    'roi_io_paths': roi_io_paths,
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
            [sys.executable, __file__, name],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_layout_typography_and_theme(self):
        for scale_factor in ('1', '1.25', '1.5'):
            with self.subTest(scale_factor=scale_factor):
                self.run_probe('layout', scale_factor)

    def test_interface_font_size_persists(self):
        self.run_probe('interface_font_persistence')

    def test_loading_an_image_tracks_the_global_path(self):
        self.run_probe('image_loading')

    def test_global_resources_and_header_states(self):
        self.run_probe('global_resources')

    def test_changed_inputs_discard_the_confidence_map(self):
        self.run_probe('probability_invalidation')

    def test_rebuild_rois_can_be_undone_without_losing_probability(self):
        self.run_probe('undo_after_rebuild_rois')

    def test_training_handoff_and_failure_preservation(self):
        self.run_probe('training_handoff')

    def test_process_output_is_unbuffered_and_line_buffered(self):
        self.run_probe('process_streaming')

    def test_percentile_display_range_and_immutability(self):
        self.run_probe('display_range')

    def test_confidence_view_rendering_and_state(self):
        self.run_probe('confidence_view')

    def test_canvas_selection_drag_and_zoom(self):
        self.run_probe('canvas_navigation')

    def test_roi_table_selection_and_centring(self):
        self.run_probe('roi_table')

    def test_roi_io_and_path_presentation(self):
        self.run_probe('roi_io_paths')


#%% entry point
if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] in PROBES:
        from PyQt5.QtCore import QSettings, Qt
        from PyQt5.QtWidgets import QApplication

        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        app = QApplication([])
        app.setStyle('Fusion')

        from fibre_sight.fibre_sight_workbench_gui import FibreSightWorkbench

        with tempfile.TemporaryDirectory(prefix='fibre sight settings ') as temp_dir:
            settings = QSettings(
                str(Path(temp_dir) / 'settings.ini'),
                QSettings.IniFormat,
                )
            window = FibreSightWorkbench(settings=settings)
            PROBES[sys.argv[1]](window)
            window.close()
        app.quit()
    else:
        unittest.main()
