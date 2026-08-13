'''
Created on 12 May 2026

Modified on 13 May 2026 whilst wiring the MSER controls into the workbench
Modified on 21 May 2026 to separate labelling, training, prediction, and editing
Modified on 2 June 2026 during the first command-line and diagnostics pass
Modified on 23 June 2026 to bring the MSER editor and model workflow together
Modified on 24 July 2026 to load packaged assets and the bundled model
Modified on 25 July 2026 to separate the workbench tasks and reduce the control density
fibre-sight workbench for labelling, training, prediction, and ROI curation

@author: Dinghao Luo
'''


#%% imports
from pathlib import Path
import re
import shlex
import sys

import matplotlib

matplotlib.use('Qt5Agg')

import numpy as np
from matplotlib import font_manager
from matplotlib.figure import Figure
from PyQt5.QtCore import (
    QByteArray,
    QItemSelectionModel,
    QProcess,
    QSignalBlocker,
    QTimer,
    Qt,
    )
from PyQt5.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontDatabase,
    QIcon,
    QKeySequence,
    QPalette,
    QTextCursor,
    )
from PyQt5.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSpinBox,
    QAction,
    QActionGroup,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QPlainTextEdit,
    QShortcut,
    QSizePolicy,
    QScrollArea,
    QSlider,
    QSplitter,
    QSpinBox,
    QStatusBar,
    QTabBar,
    QTableWidget,
    QTableWidgetItem,
    QTableView,
    QTabWidget,
    QToolButton,
    QToolTip,
    QVBoxLayout,
    QWidget,
    )

from ._repo import (
    default_output_root,
    default_source_root,
    get_workspace_root,
    package_path,
    )
from .api import AxonROIPredictor, get_default_checkpoint, get_model_entry
from .config import load_config, save_config
from .gui_canvas import (
    ZoomableCanvas,
    generate_distinct_colours,
    normalise_for_display,
    squeeze_image,
    )
from .mser_segmenter import (
    PARAMETER_SPECS,
    PARAMETER_TOOLTIPS,
    run_mser_segmentation,
    )
from .postprocess import probability_to_roi_dict
from .roi_io import labels_to_roi_dict, load_roi_dict, roi_dict_to_label, save_roi_dict


WORKSPACE_ROOT = get_workspace_root()
APP_ICON_PATH = package_path('assets', 'fibresight_icon.ico')
MONONOKI_FONT_DIR = package_path('assets', 'fonts', 'mononoki')
MONONOKI_FONT_FAMILY = 'mononoki'
GUI_FONT_SIZE = 12.0 if sys.platform == 'darwin' else 9.0
GUI_FONT_SIZES = range(9, 14)
DISPLAY_BLACK_DEFAULT = 1.0
DISPLAY_WHITE_DEFAULT = 99.7
DISPLAY_MIN_GAP = 1.0
DISPLAY_REDRAW_MS = 33
CURATION_WIDE_MIN_WIDTH = 650
SELECTION_OUTER_COLOUR = '#1a1a1c'
SELECTION_INNER_COLOUR = '#f4f2ec'
SELECTION_OUTER_WIDTH = 4.2
SELECTION_INNER_WIDTH = 2.4
SELECTION_COLOUR_WIDTH = 1.0

SEGMENT_GROUPS = (
    (
        'preprocessing',
        (
            ('tophat kernel', 'tophat kernel'),
            ('clahe clip', 'CLAHE clip'),
            ('clip-percentile', 'intensity clip'),
        ),
    ),
    (
        'MSER detection',
        (
            ('MSER threshold', 'brightness threshold'),
            ('MSER delta', 'delta'),
            ('MSER max variation', 'maximum variation'),
            ('MSER min area', 'candidate area minimum'),
            ('MSER max area', 'candidate area maximum'),
        ),
    ),
    (
        'ROI filters',
        (
            ('area min', 'ROI area minimum'),
            ('solidity min', 'solidity minimum'),
            ('eccentricity min', 'eccentricity minimum'),
            ('thinness max', 'thinness maximum'),
            ('aspect ratio min', 'aspect-ratio minimum'),
        ),
    ),
)

_GUI_FONT = None
_GUI_BOLD_FONT = None
_GUI_FONT_FAMILY = None


#%% helpers
def load_gui_font(size=None, bold=False):
    global _GUI_FONT, _GUI_BOLD_FONT, _GUI_FONT_FAMILY
    if _GUI_FONT is None:
        families = set()
        for font_path in sorted(MONONOKI_FONT_DIR.glob('*.ttf')):
            # loading from data avoids macOS quarantine blocking application fonts
            font_data = QByteArray(font_path.read_bytes())
            font_id = QFontDatabase.addApplicationFontFromData(font_data)
            if font_id >= 0:
                families.update(QFontDatabase.applicationFontFamilies(font_id))
            font_manager.fontManager.addfont(str(font_path))

        expected = MONONOKI_FONT_FAMILY.casefold()
        matches = [family for family in families if family.casefold() == expected]
        family = matches[0] if matches else None
        if family is not None:
            styles = {style.casefold() for style in QFontDatabase().styles(family)}
            if not {'regular', 'bold'}.issubset(styles):
                family = None

        if family is None:
            fallback = QFontDatabase.systemFont(QFontDatabase.FixedFont)
            family = fallback.family()

        _GUI_FONT_FAMILY = family
        _GUI_FONT = QFont(family)
        _GUI_FONT.setPointSizeF(GUI_FONT_SIZE)
        _GUI_FONT.setWeight(QFont.Normal)
        _GUI_BOLD_FONT = QFont(family)
        _GUI_BOLD_FONT.setPointSizeF(GUI_FONT_SIZE)
        _GUI_BOLD_FONT.setWeight(QFont.Bold)
        for font in (_GUI_FONT, _GUI_BOLD_FONT):
            font.setStyleStrategy(QFont.PreferAntialias)

        matplotlib.rcParams['font.family'] = family
        matplotlib.rcParams['font.monospace'] = [
            family,
            'Menlo',
            'Consolas',
            'Courier New',
            ]
        matplotlib.rcParams['lines.antialiased'] = True
        matplotlib.rcParams['patch.antialiased'] = True
        matplotlib.rcParams['text.antialiased'] = True

    source = _GUI_BOLD_FONT if bold else _GUI_FONT
    font = QFont(source)
    font.setPointSizeF(GUI_FONT_SIZE if size is None else float(size))
    return font


def display_path(path):
    path = Path(path)
    try:
        return str(path.resolve().relative_to(WORKSPACE_ROOT.resolve()))
    except (OSError, ValueError):
        try:
            parts = path.resolve().relative_to(Path.home().resolve()).parts
        except (OSError, ValueError):
            parts = path.parts
            if path.anchor and parts and parts[0] == path.anchor:
                parts = parts[1:]
        return str(Path('…', *parts[-3:]))


def elide_path(path_text, metrics, width):
    if not path_text or metrics.horizontalAdvance(path_text) <= width:
        return path_text

    path = Path(path_text)
    parts = list(path.parts)
    if len(parts) <= 1:
        return metrics.elidedText(path_text, Qt.ElideMiddle, width)

    filename = parts[-1]
    prefix = parts[0] if parts[0] not in {'/', '\\'} else ''
    candidates = []
    if prefix and prefix != '…':
        candidates.append(str(Path(prefix, '…', filename)))
    candidates.append(str(Path('…', filename)))
    for candidate in candidates:
        if metrics.horizontalAdvance(candidate) <= width:
            return candidate
    return metrics.elidedText(filename, Qt.ElideMiddle, width)


def format_log_text(text):
    text = str(text)
    try:
        workspace = str(WORKSPACE_ROOT.resolve())
    except OSError:
        workspace = str(WORKSPACE_ROOT)
    text = text.replace(workspace, '.')

    quoted_path = re.compile(r'(?P<quote>[\'"])(?P<path>/[^\'"]+)(?P=quote)')
    def replace_quoted_path(match):
        quote = match.group('quote')
        shortened_path = display_path(match.group('path'))
        return f'{quote}{shortened_path}{quote}'

    text = quoted_path.sub(replace_quoted_path, text)

    home = str(Path.home())
    lines = []
    for line in text.splitlines(keepends=True):
        line_end = len(line.rstrip('\r\n'))
        content = line[:line_end]
        path_start = None
        for marker in (' from ', ' to ', ': '):
            marker_idx = content.rfind(marker)
            if marker_idx < 0:
                continue
            candidate = content[marker_idx + len(marker):]
            if candidate.startswith('/') or re.match(r'^[A-Za-z]:[\\/]', candidate):
                path_start = marker_idx + len(marker)
                break

        if path_start is None:
            home_idx = content.find(home)
            if home_idx >= 0:
                path_start = home_idx

        if path_start is not None:
            candidate = content[path_start:]
            line = (
                line[:path_start] +
                display_path(candidate) +
                line[line_end:]
                )
        lines.append(line)
    text = ''.join(lines)

    path_pattern = re.compile(r'(?<![\w.…])/(?:[^\s|,;:()]+/)*[^\s|,;:()]+')
    return path_pattern.sub(lambda match: display_path(match.group(0)), text)


class ElidedLabel(QLabel):
    def __init__(self, text='', mode=Qt.ElideMiddle):
        super().__init__()
        self._full_text = ''
        self._elide_mode = mode
        self.setMinimumWidth(0)
        self.setText(text)

    def text(self):
        return self._full_text

    def setText(self, text):
        self._full_text = str(text)
        self.setToolTip(self._full_text)
        self._update_elision()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_elision()

    def _update_elision(self):
        shown = self.fontMetrics().elidedText(
            self._full_text,
            self._elide_mode,
            max(0, self.width()),
            )
        QLabel.setText(self, shown)


class PathLineEdit(QLineEdit):
    def __init__(self, text=''):
        super().__init__()
        self._overlay = QLabel(self)
        self._overlay.setObjectName('pathOverlay')
        self._overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._overlay.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.textChanged.connect(self._update_path_display)
        self.setText(text)

    def focusInEvent(self, event):
        self._overlay.hide()
        super().focusInEvent(event)

    def focusOutEvent(self, event):
        super().focusOutEvent(event)
        self._update_path_display()
        self._overlay.show()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_overlay()
        self._update_path_display()

    def showEvent(self, event):
        super().showEvent(event)
        self._position_overlay()
        self._update_path_display()
        if not self.hasFocus():
            self._overlay.show()

    def _position_overlay(self):
        frame = self.style().pixelMetric(self.style().PM_DefaultFrameWidth, None, self)
        self._overlay.setGeometry(
            frame + 5,
            frame,
            max(0, self.width() - 2 * frame - 10),
            max(0, self.height() - 2 * frame),
            )

    def _update_path_display(self, _text=None):
        text = self.text()
        self.setToolTip(text)
        shown = display_path(text) if text else ''
        metrics = self._overlay.fontMetrics()
        self._overlay.setText(elide_path(shown, metrics, self._overlay.width()))


#%% main window
class FibreSightWorkbench(QMainWindow):
    def __init__(self):
        app = QApplication.instance()
        if app is not None:
            app.setStyle('Fusion')
        gui_font = load_gui_font() if app is not None else None
        super().__init__()
        if app is not None and gui_font is not None:
            app.setFont(gui_font)
            QToolTip.setFont(gui_font)
        self.setWindowTitle('FibreSight')
        self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.resize(1280, 820)
        self.setMinimumSize(1000, 680)

        self.ref_image = None
        self.image_path = None
        self.recname = None
        self.roi_dict = {}
        self.labelled = None
        self.selected = set()
        self.fixed_ids = set()
        self.undo_stack = []
        self.probability = None
        self.predictor = None
        self.process = None
        self.current_process_name = None
        self.pending_checkpoint_path = None
        self.last_saved_model_path = None
        self.dark_mode = True
        self.display_black = DISPLAY_BLACK_DEFAULT
        self.display_white = DISPLAY_WHITE_DEFAULT
        self.display_mode = 'image'
        self.interface_font_size = GUI_FONT_SIZE
        self._syncing_roi_table = False
        self._process_stdout_buffer = ''
        self._process_stderr_buffer = ''
        self._process_was_stopped = False
        self._controls_split_positions = {}
        self._sizing_controls_splitter = False

        self._build_widgets()
        self._build_layout()
        self._apply_palette_and_style()
        self._connect_shortcuts()
        self.plot_image()
        self.refresh_status()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.schedule_curation_layout()
        if hasattr(self, 'controls_split_timer'):
            self.controls_split_timer.start(0)

    #%% setup
    def _build_widgets(self):
        self.fig = Figure(dpi=100, facecolor=self._theme()['canvas'])
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
        self.ax.axis('off')
        self.canvas = ZoomableCanvas(self.fig, self.ax)
        self.canvas.setMinimumSize(300, 300)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.canvas.setAccessibleName('ROI image canvas')
        self.canvas.setAccessibleDescription(
            'click ROI to select · drag to pan · scroll or ± to zoom · '
            'Shift-click to add an ROI to the selection'
            )
        self.canvas.setToolTip('')
        self.canvas.mpl_connect('button_release_event', self.on_click)

        self.state_label = ElidedLabel('image: not loaded')
        self.state_label.setObjectName('stateSummary')
        self.state_label.setFont(load_gui_font(bold=True))
        self.state_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
            )
        self.roi_label = QLabel('0 ROIs | 0 selected | 0 fixed')
        self.roi_label.setObjectName('panelValue')
        self.model_separator = QLabel('·')
        self.model_separator.setObjectName('panelValue')
        self.model_label = ElidedLabel('model: not selected')
        self.model_label.setObjectName('panelValue')
        self.model_label.setSizePolicy(
            QSizePolicy.Ignored,
            QSizePolicy.Preferred,
            )

        self.tabs = QTabWidget()
        self.predict_tab, self.predict_tab_content = self._make_scroll_tab()
        (
            self.mser_tab,
            self.mser_scroll,
            self.mser_tab_content,
            self.mser_action_bar,
            self.mser_action_layout,
            ) = self._make_label_tab()
        self.training_tab, self.training_tab_content = self._make_scroll_tab()
        self.tabs.addTab(self.predict_tab, 'Predict')
        self.tabs.addTab(self.mser_tab, 'Label')
        self.tabs.addTab(self.training_tab, 'Train')
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.tabs.currentChanged.connect(self.controls_tab_changed)
        self._build_training_widgets()
        self._build_prediction_widgets()
        self._build_segment_widgets()
        self._build_editing_widgets()
        self._build_persistent_widgets()
        self.output_box = self.make_log_box()

        self.roi_overlay_check = QCheckBox('ROI on')
        self.roi_overlay_check.setChecked(True)
        self.roi_overlay_check.stateChanged.connect(self.set_roi_overlay_visible)
        self.dark_mode_check = QCheckBox('dark mode')
        self.dark_mode_check.setChecked(True)
        self.dark_mode_check.stateChanged.connect(self.set_dark_mode)

        self.interface_font_button = QToolButton()
        self.interface_font_button.setText('Aa')
        self.interface_font_button.setAccessibleName('interface text size')
        self.interface_font_button.setToolTip('interface text size')
        self.interface_font_button.setPopupMode(QToolButton.InstantPopup)
        self.interface_font_menu = QMenu(self.interface_font_button)
        self.interface_font_group = QActionGroup(self)
        self.interface_font_group.setExclusive(True)
        self.interface_font_actions = {}
        for size in GUI_FONT_SIZES:
            action = QAction(f'{size} pt', self.interface_font_group)
            action.setCheckable(True)
            action.setChecked(size == GUI_FONT_SIZE)
            action.triggered.connect(
                lambda _checked, point_size=size: self.set_interface_font_size(
                    point_size
                    )
                )
            self.interface_font_menu.addAction(action)
            self.interface_font_actions[size] = action
        self.interface_font_button.setMenu(self.interface_font_menu)

    def _make_scroll_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName('tabContent')
        scroll.setWidget(content)
        return scroll, content

    def _make_label_tab(self):
        tab = QWidget()
        tab.setObjectName('labelTab')
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        scroll, content = self._make_scroll_tab()
        action_bar = QFrame()
        action_bar.setObjectName('labelActionBar')
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(10, 8, 10, 8)
        action_layout.setSpacing(6)

        layout.addWidget(scroll, 1)
        layout.addWidget(action_bar)
        return tab, scroll, content, action_bar, action_layout

    def _build_training_widgets(self):
        self.source_root_line = PathLineEdit(str(default_source_root()))
        self.manifest_line = PathLineEdit(str(WORKSPACE_ROOT / 'manifests' / 'ch2_manifest.csv'))
        self.config_line = PathLineEdit(
            str(package_path('configs', 'ch2_unet.yaml'))
            )
        self.run_name_line = QLineEdit('ch2_unet')
        self.source_root_line.textChanged.connect(lambda _: self.refresh_status())
        self.manifest_line.textChanged.connect(lambda _: self.refresh_status())
        self.config_line.textChanged.connect(lambda _: self.refresh_status())
        self.run_name_line.textChanged.connect(lambda _: self.refresh_status())

        self.val_fraction_spin = QDoubleSpinBox()
        self.val_fraction_spin.setRange(0, 0.5)
        self.val_fraction_spin.setSingleStep(0.01)
        self.val_fraction_spin.setDecimals(2)
        self.val_fraction_spin.setValue(0.15)
        self.prepare_value_control(self.val_fraction_spin)

        self.test_fraction_spin = QDoubleSpinBox()
        self.test_fraction_spin.setRange(0, 0.5)
        self.test_fraction_spin.setSingleStep(0.01)
        self.test_fraction_spin.setDecimals(2)
        self.test_fraction_spin.setValue(0.15)
        self.prepare_value_control(self.test_fraction_spin)

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(80)
        self.prepare_value_control(self.epochs_spin)

        self.build_manifest_button = QPushButton('scan labelled sessions')
        self.train_model_button = QPushButton('TRAIN MODEL')
        self.evaluate_model_button = QPushButton('score model')
        self.stop_process_button = QPushButton('stop process')
        self.inspect_manifest_button = QPushButton('dataset summary')
        self.preview_training_button = QPushButton('save label preview')
        self.preview_predictions_button = QPushButton('save prediction preview')
        self.set_button_role(self.train_model_button, 'primary')
        self.set_button_role(self.stop_process_button, 'danger')
        self.set_button_role(self.inspect_manifest_button, 'quiet')
        self.set_button_role(self.evaluate_model_button, 'quiet')
        self.set_button_role(self.preview_training_button, 'quiet')
        self.set_button_role(self.preview_predictions_button, 'quiet')
        self.evaluate_model_button.setToolTip('score the current trained model on held-out labelled sessions')
        self.preview_predictions_button.setToolTip('save example overlays comparing model ROIs with held-out labels')
        self.stop_process_button.hide()

        self.build_manifest_button.clicked.connect(self.build_manifest)
        self.train_model_button.clicked.connect(self.train_model)
        self.evaluate_model_button.clicked.connect(self.evaluate_model)
        self.inspect_manifest_button.clicked.connect(self.inspect_manifest)
        self.preview_training_button.clicked.connect(self.preview_training_labels)
        self.preview_predictions_button.clicked.connect(self.preview_model_predictions)
        self.stop_process_button.clicked.connect(self.stop_process)

    def _build_prediction_widgets(self):
        self.image_line = PathLineEdit()
        self.checkpoint_line = PathLineEdit(str(get_default_checkpoint()))
        self.image_line.textChanged.connect(self.prediction_inputs_changed)
        self.image_line.editingFinished.connect(self.load_edited_channel_image)
        self.checkpoint_line.textChanged.connect(self.prediction_inputs_changed)

        model_entry = get_model_entry()
        # these stayed visible because they were the useful controls during tuning
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.99)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(float(model_entry['threshold']))
        self.prepare_value_control(self.threshold_spin)

        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(1, 100000)
        self.min_size_spin.setValue(int(model_entry['min_size']))
        self.prepare_value_control(self.min_size_spin)

        self.predict_button = QPushButton('predict')
        self.rebuild_rois_button = QPushButton('rebuild ROIs')
        self.set_button_role(self.predict_button, 'primary')
        self.set_button_role(self.rebuild_rois_button, 'secondary')
        self.predict_button.setToolTip(
            'load the selected image and model, then predict ROIs'
            )
        self.rebuild_rois_button.setToolTip(
            'rebuild all ROIs from the cached confidence map using the '
            'threshold and minimum area; prediction does not run again; '
            'this replaces the current ROIs; Undo restores them'
            )
        self.rebuild_rois_button.hide()

        self.threshold_spin.setToolTip('higher values keep only stronger model detections')
        self.min_size_spin.setToolTip('smallest ROI size to keep')

        self.predict_button.clicked.connect(self.predict_rois)
        self.rebuild_rois_button.clicked.connect(
            self.rebuild_rois_from_probability
            )


    def _build_segment_widgets(self):
        self.segment_param_widgets = {}
        for spec in PARAMETER_SPECS:
            self.segment_param_widgets[spec['name']] = self._make_segment_param_widget(spec)

        self.segment_button = QPushButton('SEGMENT')
        self.reset_segment_button = QPushButton('reset params')
        self.fix_selected_button = QPushButton('fix')
        self.clear_fixed_button = QPushButton('clear fixed')
        self.clear_unfixed_button = QPushButton('clear unfixed')
        self.segment_load_roi_button = QPushButton('import ROIs')

        self.set_button_role(self.segment_button, 'primary')
        self.set_button_role(self.segment_load_roi_button, 'secondary')
        self.set_button_role(self.reset_segment_button, 'quiet')
        self.set_button_role(self.fix_selected_button, 'quiet')
        self.set_button_role(self.clear_fixed_button, 'quiet')
        self.set_button_role(self.clear_unfixed_button, 'dangerQuiet')

        self.fix_selected_button.setMinimumWidth(55)
        self.fix_selected_button.setToolTip('keep selected ROIs when segmentation is run again')
        self.clear_fixed_button.setToolTip('remove all fixed marks without deleting ROIs')
        self.clear_unfixed_button.setToolTip('remove all ROIs except fixed ROIs')

        self.segment_button.clicked.connect(self.segment_rois)
        self.reset_segment_button.clicked.connect(self.reset_segment_parameters)
        self.fix_selected_button.clicked.connect(self.toggle_selected_fixed)
        self.clear_fixed_button.clicked.connect(self.clear_fixed)
        self.clear_unfixed_button.clicked.connect(self.clear_unfixed)
        self.segment_load_roi_button.clicked.connect(self.load_roi_file)

    def _build_editing_widgets(self):
        self.curate_buttons = {
            'select_all': QPushButton('select all'),
            'delete': QPushButton('delete'),
            'merge': QPushButton('merge'),
            'undo': QPushButton('undo'),
            'zoom_out': QPushButton('−'),
            'fit_view': QPushButton('fit'),
            'zoom_in': QPushButton('+'),
            'save_roi': QPushButton('export ROIs'),
        }
        self.set_button_role(self.curate_buttons['delete'], 'dangerQuiet')
        self.set_button_role(self.curate_buttons['select_all'], 'quiet')
        self.set_button_role(self.curate_buttons['merge'], 'quiet')
        self.set_button_role(self.curate_buttons['undo'], 'quiet')
        self.set_button_role(self.curate_buttons['zoom_out'], 'quiet')
        self.set_button_role(self.curate_buttons['fit_view'], 'quiet')
        self.set_button_role(self.curate_buttons['zoom_in'], 'quiet')
        self.set_button_role(self.curate_buttons['save_roi'], 'secondary')

        self.curate_buttons['select_all'].clicked.connect(self.select_all)
        self.curate_buttons['delete'].clicked.connect(self.delete_selected)
        self.curate_buttons['merge'].clicked.connect(self.merge_selected)
        self.curate_buttons['undo'].clicked.connect(self.undo)
        self.curate_buttons['zoom_out'].clicked.connect(self.zoom_out)
        self.curate_buttons['fit_view'].clicked.connect(self.reset_view)
        self.curate_buttons['zoom_in'].clicked.connect(self.zoom_in)
        self.curate_buttons['save_roi'].clicked.connect(self.save_roi_file)
        self.update_export_tooltip()

    def _build_persistent_widgets(self):
        self.image_view_button = QPushButton('image')
        self.confidence_view_button = QPushButton('confidence')
        for button in (self.image_view_button, self.confidence_view_button):
            button.setCheckable(True)
            self.set_button_role(button, 'viewMode')
        self.image_view_button.setChecked(True)
        self.confidence_view_button.setEnabled(False)
        self.confidence_view_button.setToolTip(
            'available after prediction produces a confidence map'
            )

        self.display_mode_group = QButtonGroup(self)
        self.display_mode_group.setExclusive(True)
        self.display_mode_group.addButton(self.image_view_button)
        self.display_mode_group.addButton(self.confidence_view_button)
        self.image_view_button.clicked.connect(lambda: self.set_display_mode('image'))
        self.confidence_view_button.clicked.connect(lambda: self.set_display_mode('confidence'))

        self.black_slider = QSlider(Qt.Horizontal)
        self.white_slider = QSlider(Qt.Horizontal)
        self.black_value = QDoubleSpinBox()
        self.white_value = QDoubleSpinBox()
        for slider in (self.black_slider, self.white_slider):
            slider.setRange(0, 1000)
            slider.setSingleStep(1)
            slider.setPageStep(10)
        for spin in (self.black_value, self.white_value):
            spin.setRange(0, 100)
            spin.setDecimals(1)
            spin.setSingleStep(0.1)
            spin.setSuffix('%')
            spin.setKeyboardTracking(False)
            spin.setButtonSymbols(QAbstractSpinBox.NoButtons)
            spin.setMaximumWidth(70)

        self.black_slider.setValue(int(round(DISPLAY_BLACK_DEFAULT * 10)))
        self.white_slider.setValue(int(round(DISPLAY_WHITE_DEFAULT * 10)))
        self.black_value.setValue(DISPLAY_BLACK_DEFAULT)
        self.white_value.setValue(DISPLAY_WHITE_DEFAULT)
        self.black_slider.valueChanged.connect(self.display_slider_changed)
        self.white_slider.valueChanged.connect(self.display_slider_changed)
        self.black_value.valueChanged.connect(self.display_spin_changed)
        self.white_value.valueChanged.connect(self.display_spin_changed)

        self.display_redraw_timer = QTimer(self)
        self.display_redraw_timer.setSingleShot(True)
        self.display_redraw_timer.setInterval(DISPLAY_REDRAW_MS)
        self.display_redraw_timer.timeout.connect(
            lambda: self.plot_image(preserve_view=True)
            )

        self.reset_display_button = QPushButton('reset')
        self.set_button_role(self.reset_display_button, 'quiet')
        self.reset_display_button.setAccessibleName('reset display')
        self.reset_display_button.setToolTip(
            'restore the default black and white points'
            )
        self.reset_display_button.clicked.connect(self.reset_display_range)

        self.roi_table = QTableWidget(0, 4)
        self.roi_table.setObjectName('roiTable')
        self.roi_table.setHorizontalHeaderLabels(['colour', 'ID', 'pixels', 'fixed'])
        self.roi_table.horizontalHeaderItem(2).setTextAlignment(
            Qt.AlignRight | Qt.AlignVCenter
            )
        self.roi_table.verticalHeader().hide()
        self.roi_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.roi_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.roi_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.roi_table.setSortingEnabled(False)
        self.roi_table.setShowGrid(False)
        self.roi_table.setMinimumHeight(140)
        self.roi_table.setAccessibleName('ROI table')
        self.roi_table.setAccessibleDescription(
            'select rows to select ROIs; double-click a row to centre it'
            )
        header = self.roi_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.roi_table.itemSelectionChanged.connect(self.roi_table_selection_changed)
        self.roi_table.cellDoubleClicked.connect(self.centre_roi_from_table)

        self.roi_empty_label = QLabel('no ROIs')
        self.roi_empty_label.setObjectName('emptyState')
        self.roi_empty_label.setAlignment(Qt.AlignCenter)

    def _build_layout(self):
        canvas_layout = QVBoxLayout()
        canvas_layout.setContentsMargins(3, 3, 3, 3)
        canvas_layout.addWidget(self.canvas)
        canvas_frame = QFrame()
        canvas_frame.setObjectName('canvasFrame')
        canvas_frame.setLayout(canvas_layout)

        self.curation_bar = QFrame()
        self.curation_bar.setObjectName('curationBar')
        self.curation_layout = QVBoxLayout(self.curation_bar)
        self.curation_layout.setContentsMargins(7, 5, 7, 5)
        self.curation_layout.setSpacing(3)

        self.curation_primary_row = QHBoxLayout()
        self.curation_primary_row.setSpacing(3)
        self.curation_history_divider = QFrame()
        self.curation_history_divider.setObjectName('curationHistoryDivider')
        self.curation_history_divider.setFixedWidth(1)
        self.curation_history_divider.setFixedHeight(20)
        self.curation_navigation_row = QHBoxLayout()
        self.curation_navigation_row.setSpacing(3)
        self.canvas_hint = ElidedLabel('Shift-click to add', mode=Qt.ElideRight)
        self.canvas_hint.setObjectName('canvasHint')
        self.canvas_hint.setToolTip('')
        self.canvas_hint.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
            )
        self.curation_layout.addLayout(self.curation_primary_row)
        self.curation_layout.addLayout(self.curation_navigation_row)
        self._curation_layout_is_wide = None
        self.curation_layout_timer = QTimer(self)
        self.curation_layout_timer.setSingleShot(True)
        self.curation_layout_timer.timeout.connect(self.update_curation_layout)
        self.update_curation_layout()

        canvas_stage = QWidget()
        canvas_stage.setObjectName('canvasStage')
        canvas_stage_layout = QVBoxLayout(canvas_stage)
        canvas_stage_layout.setContentsMargins(0, 0, 0, 0)
        canvas_stage_layout.setSpacing(8)
        canvas_stage_layout.addWidget(self.curation_bar)
        canvas_stage_layout.addWidget(canvas_frame, 1)

        activity_frame = QFrame()
        activity_frame.setObjectName('activityFrame')
        activity_layout = QVBoxLayout(activity_frame)
        activity_layout.setContentsMargins(8, 8, 8, 8)
        activity_layout.addWidget(self.output_box, 1)

        self.activity_splitter = QSplitter(Qt.Vertical)
        self.activity_splitter.setObjectName('activitySplitter')
        self.activity_splitter.addWidget(canvas_stage)
        self.activity_splitter.addWidget(activity_frame)
        self.activity_splitter.setChildrenCollapsible(True)
        self.activity_splitter.setCollapsible(0, False)
        self.activity_splitter.setCollapsible(1, True)
        self.activity_splitter.setStretchFactor(0, 1)
        self.activity_splitter.setStretchFactor(1, 0)
        self.activity_splitter.setSizes([755, 0])

        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.activity_splitter)
        right_pane = QWidget()
        right_pane.setObjectName('mainPane')
        right_pane.setLayout(right_layout)

        self._layout_prediction_tab()
        self._layout_mser_tab()
        self._layout_training_tab()

        self.resources_panel = QWidget()
        self.resources_panel.setObjectName('resourcesPanel')
        resources_layout = QVBoxLayout(self.resources_panel)
        resources_layout.setContentsMargins(9, 9, 9, 7)
        resources_layout.setSpacing(5)
        resources_layout.addWidget(self.make_section_label('resources'))
        resources_form = QFormLayout()
        resources_form.setHorizontalSpacing(8)
        resources_form.setVerticalSpacing(5)
        resources_form.addRow(
            self.make_form_label(
                'channel-2 image',
                'channel-2 reference image used across prediction and labelling',
                self.image_line,
                ),
            self._path_row(self.image_line, self.choose_channel_image),
            )
        resources_form.addRow(
            self.make_form_label(
                'trained model',
                'selected model used for prediction and scoring',
                self.checkpoint_line,
                ),
            self._path_row(self.checkpoint_line, self.browse_checkpoint),
            )
        resources_layout.addLayout(resources_form)

        self.upper_controls = QWidget()
        self.upper_controls.setObjectName('upperControls')
        self.upper_controls_layout = QVBoxLayout(self.upper_controls)
        self.upper_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.upper_controls_layout.setSpacing(6)
        self.upper_controls_layout.addWidget(self.resources_panel)
        self.upper_controls_layout.addWidget(self.tabs, 1)

        self.persistent_panel = QWidget()
        self.persistent_panel.setObjectName('persistentPanel')
        persistent_layout = QVBoxLayout(self.persistent_panel)
        persistent_layout.setContentsMargins(9, 5, 9, 5)
        persistent_layout.setSpacing(3)
        display_header = QHBoxLayout()
        display_header.addWidget(self.make_section_label('display'))
        display_header.addStretch(1)
        display_header.addWidget(self.reset_display_button)
        display_header.addWidget(self.image_view_button)
        display_header.addWidget(self.confidence_view_button)
        persistent_layout.addLayout(display_header)

        black_row = QHBoxLayout()
        black_row.addWidget(QLabel('black point'))
        black_row.addWidget(self.black_slider, 1)
        black_row.addWidget(self.black_value)
        persistent_layout.addLayout(black_row)
        white_row = QHBoxLayout()
        white_row.addWidget(QLabel('white point'))
        white_row.addWidget(self.white_slider, 1)
        white_row.addWidget(self.white_value)
        persistent_layout.addLayout(white_row)
        persistent_layout.addWidget(self.make_section_label('ROIs'))
        persistent_layout.addWidget(self.roi_table, 1)
        persistent_layout.addWidget(self.roi_empty_label)

        self.controls_splitter = QSplitter(Qt.Vertical)
        self.controls_splitter.setObjectName('controlsSplitter')
        self.controls_splitter.addWidget(self.upper_controls)
        self.controls_splitter.addWidget(self.persistent_panel)
        self.controls_splitter.setChildrenCollapsible(False)
        self.controls_splitter.setStretchFactor(0, 3)
        self.controls_splitter.setStretchFactor(1, 2)
        self.controls_splitter.setSizes([430, 240])
        self.controls_splitter.splitterMoved.connect(
            self.remember_controls_split_position
            )
        self.controls_split_timer = QTimer(self)
        self.controls_split_timer.setSingleShot(True)
        self.controls_split_timer.timeout.connect(
            self.size_controls_for_current_tab
            )

        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(0)
        controls_layout.addWidget(self.controls_splitter, 1)

        controls_widget = QWidget()
        controls_widget.setObjectName('controlsPanel')
        controls_widget.setLayout(controls_layout)
        controls_widget.setMinimumWidth(400)
        controls_widget.setMaximumWidth(520)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setObjectName('mainSplitter')
        self.main_splitter.addWidget(controls_widget)
        self.main_splitter.addWidget(right_pane)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setSizes([420, 840])
        self.main_splitter.splitterMoved.connect(self.schedule_curation_layout)

        stage_header = QFrame()
        stage_header.setObjectName('stageHeader')
        stage_layout = QHBoxLayout(stage_header)
        stage_layout.setContentsMargins(6, 7, 6, 7)
        stage_layout.setSpacing(4)

        stage_layout.addWidget(self.state_label, 1)
        stage_layout.addWidget(self.roi_label)
        stage_layout.addWidget(self.model_separator)
        stage_layout.addWidget(self.model_label, 1)
        stage_layout.addSpacing(8)
        stage_layout.addWidget(self.roi_overlay_check)
        stage_layout.addWidget(self.dark_mode_check)
        stage_layout.addWidget(self.interface_font_button)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(7)
        main_layout.addWidget(stage_header)
        main_layout.addWidget(self.main_splitter, 1)

        container = QWidget()
        container.setObjectName('centralWidget')
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.controls_split_timer.start(0)

    def controls_tab_changed(self, _index):
        self.refresh_status()
        if hasattr(self, 'controls_split_timer'):
            self.controls_split_timer.start(0)

    def schedule_curation_layout(self, *_args):
        if hasattr(self, 'curation_layout_timer'):
            self.curation_layout_timer.start(0)

    @staticmethod
    def _clear_layout(layout):
        while layout.count():
            layout.takeAt(0)

    def update_curation_layout(self):
        wide = self.curation_bar.width() >= CURATION_WIDE_MIN_WIDTH
        if wide == self._curation_layout_is_wide:
            return

        self._clear_layout(self.curation_primary_row)
        self._clear_layout(self.curation_navigation_row)

        for button in (
            self.curate_buttons['select_all'],
            self.fix_selected_button,
            self.curate_buttons['merge'],
            self.curate_buttons['delete'],
            ):
            self.curation_primary_row.addWidget(button)
        self.curation_primary_row.addSpacing(2)
        self.curation_primary_row.addWidget(self.curation_history_divider)
        self.curation_primary_row.addSpacing(2)
        self.curation_primary_row.addWidget(self.curate_buttons['undo'])

        if wide:
            self.curation_primary_row.addStretch(1)
            self.curation_primary_row.addWidget(self.segment_load_roi_button)
            self.curation_primary_row.addWidget(self.curate_buttons['save_roi'])
        else:
            self.curation_primary_row.addStretch(1)

        self.curation_navigation_row.addWidget(self.canvas_hint, 1)
        if not wide:
            self.curation_navigation_row.addWidget(self.segment_load_roi_button)
            self.curation_navigation_row.addWidget(self.curate_buttons['save_roi'])
        self.curation_navigation_row.addWidget(self.curate_buttons['zoom_out'])
        self.curation_navigation_row.addWidget(self.curate_buttons['fit_view'])
        self.curation_navigation_row.addWidget(self.curate_buttons['zoom_in'])
        self._curation_layout_is_wide = wide
        self.curation_layout.invalidate()
        self.curation_bar.updateGeometry()
        if hasattr(self, 'activity_splitter'):
            self.activity_splitter.updateGeometry()
        if hasattr(self, 'main_splitter'):
            self.main_splitter.updateGeometry()
        if self.centralWidget() is not None:
            self.centralWidget().updateGeometry()

    def remember_controls_split_position(self, _position, _index):
        if self._sizing_controls_splitter:
            return
        sizes = self.controls_splitter.sizes()
        if sizes:
            self._controls_split_positions[self.tabs.currentIndex()] = sizes[0]

    def size_controls_for_current_tab(self):
        sizes = self.controls_splitter.sizes()
        if len(sizes) != 2:
            return

        total = sum(sizes)
        if total <= 0:
            return

        tab_index = self.tabs.currentIndex()
        stored_height = self._controls_split_positions.get(tab_index)
        if stored_height is None:
            current_scroll = (
                self.mser_scroll
                if self.tabs.currentWidget() is self.mser_tab else
                self.tabs.currentWidget()
                )
            tab_chrome = max(
                0,
                self.tabs.height() - current_scroll.viewport().height(),
                )
            stored_height = (
                self.resources_panel.sizeHint().height() +
                self.upper_controls_layout.spacing() +
                tab_chrome +
                max(
                    current_scroll.widget().sizeHint().height(),
                    current_scroll.widget().minimumSizeHint().height(),
                    )
                )

        min_upper = self.upper_controls.minimumSizeHint().height()
        max_upper = total - self.persistent_panel.minimumSizeHint().height()
        target = int(np.clip(stored_height, min_upper, max_upper))
        self._sizing_controls_splitter = True
        self.controls_splitter.setSizes([target, total - target])
        self._sizing_controls_splitter = False

    def _layout_mser_tab(self):
        layout = QVBoxLayout(self.mser_tab_content)
        layout.setSpacing(6)
        layout.setContentsMargins(10, 10, 10, 10)
        self._layout_mser_section(layout)
        self.mser_action_layout.addWidget(self.segment_button, 1)
        self.mser_action_layout.addWidget(self.reset_segment_button)

    def _layout_prediction_tab(self):
        layout = QVBoxLayout(self.predict_tab_content)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        self._layout_prediction_section(layout)
        layout.addStretch(1)

    def _layout_training_tab(self):
        layout = QVBoxLayout(self.training_tab_content)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        self._layout_training_section(layout)
        layout.addStretch(1)

    def _layout_mser_section(self, layout):
        specs_by_name = {spec['name']: spec for spec in PARAMETER_SPECS}
        for group_index, (group_name, entries) in enumerate(SEGMENT_GROUPS):
            if group_index:
                layout.addSpacing(10)
            layout.addWidget(self.make_section_label(group_name))
            param_form = QFormLayout()
            param_form.setHorizontalSpacing(12)
            param_form.setVerticalSpacing(10)
            for internal_name, display_name in entries:
                spec = specs_by_name[internal_name]
                tooltip = PARAMETER_TOOLTIPS.get(spec['name'], '')
                widget = self.segment_param_widgets[spec['name']]
                widget.setToolTip(tooltip)
                field = QWidget()
                field_layout = QHBoxLayout(field)
                field_layout.setContentsMargins(0, 0, 0, 0)
                field_layout.addStretch(1)
                field_layout.addWidget(widget)
                param_form.addRow(
                    self.make_form_label(display_name, tooltip, widget),
                    field,
                    )
            layout.addLayout(param_form)
            if group_index < len(SEGMENT_GROUPS) - 1:
                layout.addStretch(1)

        buttons = QGridLayout()
        buttons.setHorizontalSpacing(6)
        buttons.setVerticalSpacing(6)
        buttons.addWidget(self.clear_fixed_button, 0, 0)
        buttons.addWidget(self.clear_unfixed_button, 0, 1)
        layout.addLayout(buttons)

    def _layout_training_section(self, layout):
        layout.addWidget(self.make_section_label('data'))
        layout.addWidget(self.make_form_label(
            'labelled sessions',
            'folder containing processed sessions for training',
            self.source_root_line,
            ))
        layout.addWidget(self._path_row(self.source_root_line, self.browse_source_root))
        layout.addWidget(self.make_form_label(
            'dataset table',
            'CSV index of labelled images and ROI dicts',
            self.manifest_line,
            ))
        layout.addWidget(self._path_row(self.manifest_line, self.browse_manifest_out))
        split_form = QFormLayout()
        split_form.setVerticalSpacing(5)
        split_form.addRow(
            self.make_form_label(
                'validation split',
                'fraction used for tuning during training',
                self.val_fraction_spin,
                ),
            self.val_fraction_spin,
            )
        split_form.addRow(
            self.make_form_label(
                'test split',
                'held-out fraction used for scoring',
                self.test_fraction_spin,
                ),
            self.test_fraction_spin,
            )
        layout.addLayout(split_form)
        data_actions = QGridLayout()
        data_actions.setHorizontalSpacing(6)
        data_actions.setVerticalSpacing(6)
        data_actions.addWidget(self.build_manifest_button, 0, 0)
        data_actions.addWidget(self.inspect_manifest_button, 0, 1)
        data_actions.addWidget(self.preview_training_button, 1, 0, 1, 2)
        layout.addLayout(data_actions)

        layout.addWidget(self.make_section_label('training'))
        layout.addWidget(self.make_form_label(
            'training recipe',
            'YAML settings used for model training',
            self.config_line,
            ))
        layout.addWidget(self._path_row(self.config_line, self.browse_config))
        form = QFormLayout()
        form.setVerticalSpacing(5)
        form.addRow(
            self.make_form_label(
                'run name',
                'name used for the output training folder',
                self.run_name_line,
                ),
            self.run_name_line,
            )
        form.addRow(
            self.make_form_label(
                'epochs',
                'full passes through the training set',
                self.epochs_spin,
                ),
            self.epochs_spin,
            )
        layout.addLayout(form)

        training_actions = QHBoxLayout()
        training_actions.setSpacing(6)
        training_actions.addWidget(self.train_model_button, 1)
        training_actions.addWidget(self.stop_process_button)
        layout.addLayout(training_actions)

        layout.addWidget(self.make_section_label('evaluation'))
        evaluation_actions = QGridLayout()
        evaluation_actions.setHorizontalSpacing(6)
        evaluation_actions.setVerticalSpacing(6)
        evaluation_actions.addWidget(self.evaluate_model_button, 0, 0)
        evaluation_actions.addWidget(self.preview_predictions_button, 0, 1)
        layout.addLayout(evaluation_actions)

    def _layout_prediction_section(self, layout):
        form = QFormLayout()
        form.setVerticalSpacing(5)
        form.addRow(
            self.make_form_label(
                'prediction threshold',
                'higher values retain fewer, more confident ROIs',
                self.threshold_spin,
                ),
            self.threshold_spin,
            )
        form.addRow(
            self.make_form_label(
                'minimum ROI area (pixels)',
                'discard connected components below this area',
                self.min_size_spin,
                ),
            self.min_size_spin,
            )
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(self.predict_button, 1)
        buttons.addWidget(self.rebuild_rois_button)
        layout.addLayout(buttons)

    def _path_row(self, line_edit, browse_slot):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        line_edit.setProperty('field', 'path')
        line_edit.style().unpolish(line_edit)
        line_edit.style().polish(line_edit)
        browse = QPushButton('browse')
        self.set_button_role(browse, 'small')
        browse.clicked.connect(browse_slot)
        target = line_edit.accessibleName() or 'path'
        browse.setAccessibleName(f'browse for {target}')
        layout.addWidget(line_edit, 1)
        layout.addWidget(browse)
        return row

    def make_log_box(self):
        box = QPlainTextEdit()
        box.setObjectName('logBox')
        box.setReadOnly(True)
        box.setMaximumBlockCount(3000)
        box.setMinimumHeight(72)
        box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        box.setPlaceholderText('console')
        box.setAccessibleName('activity log')
        return box

    def make_form_label(self, text, tooltip, buddy=None):
        label = QLabel(text)
        label.setToolTip(tooltip.rstrip('.'))
        if buddy is not None:
            label.setBuddy(buddy)
            if not buddy.accessibleName():
                buddy.setAccessibleName(text)
            if tooltip and not buddy.accessibleDescription():
                buddy.setAccessibleDescription(tooltip.rstrip('.'))
        return label

    @staticmethod
    def make_section_label(text):
        label = QLabel(text)
        label.setObjectName('sectionHeading')
        label.setFont(load_gui_font(bold=True))
        return label

    @staticmethod
    def set_button_role(button, role):
        if button.property('role') == role:
            return
        button.setProperty('role', role)
        button.style().unpolish(button)
        button.style().polish(button)

    @staticmethod
    def prepare_value_control(control):
        control.setButtonSymbols(QAbstractSpinBox.NoButtons)
        control.setKeyboardTracking(False)

    def _make_segment_param_widget(self, spec):
        if spec['kind'] == 'int':
            widget = QSpinBox()
            widget.setRange(int(spec['minimum']), int(spec['maximum']))
            widget.setSingleStep(int(spec['step']))
            widget.setValue(int(spec['default']))
        else:
            widget = QDoubleSpinBox()
            widget.setDecimals(spec['decimals'])
            widget.setRange(float(spec['minimum']), float(spec['maximum']))
            widget.setSingleStep(float(spec['step']))
            widget.setValue(float(spec['default']))
        self.prepare_value_control(widget)
        widget.setAlignment(Qt.AlignRight)
        widget.setMinimumWidth(58)
        widget.setMaximumWidth(74)
        return widget

    def _connect_shortcuts(self):
        shortcuts = [
            ('Ctrl+O', self.choose_channel_image),
            ('Ctrl+M', self.load_model),
            ('Ctrl+P', self.predict_rois),
            ('Ctrl+S', self.save_roi_file),
            ('Ctrl+R', self.segment_rois),
            ('Ctrl+F', self.toggle_selected_fixed),
            ('Ctrl+Shift+F', self.clear_fixed),
            ('Ctrl+A', self.select_all),
            ('Ctrl+I', self.invert_selection),
            ('Escape', self.clear_selection),
            ('Delete', self.delete_selected),
            ('Ctrl+Z', self.undo),
            ('Backspace', self.undo),
            ('-', self.zoom_out),
            ('0', self.reset_view),
            ('+', self.zoom_in),
            ]
        for sequence, slot in shortcuts:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(slot)

    def _theme(self):
        if self.dark_mode:
            return {
                'window': '#1d1d1f',
                'surface': '#232326',
                'surface_alt': '#2b2b2f',
                'surface_strong': '#343439',
                'surface_hover': '#303034',
                'border': '#3a3a3f',
                'border_strong': '#6a6864',
                'text': '#e8e6e0',
                'muted': '#aaa7a0',
                'disabled': '#8c8982',
                'primary': '#e8e6e0',
                'primary_hover': '#f4f2ec',
                'primary_text': '#232326',
                'danger': '#8f4b42',
                'danger_text': '#faf9f6',
                'selection': '#4a494e',
                'canvas': '#1a1a1c',
                }

        return {
            'window': '#f0eee9',
            'surface': '#faf9f6',
            'surface_alt': '#f0eee9',
            'surface_strong': '#e2dfd8',
            'surface_hover': '#eae7e0',
            'border': '#cbc7bf',
            'border_strong': '#77736d',
            'text': '#292826',
            'muted': '#6d6963',
            'disabled': '#827e77',
            'primary': '#2b2b2f',
            'primary_hover': '#1f1f22',
            'primary_text': '#faf9f6',
            'danger': '#8f4b42',
            'danger_text': '#faf9f6',
            'selection': '#d9d5cd',
            'canvas': '#1a1a1c',
            }

    def _apply_palette_and_style(self):
        theme = self._theme()

        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(theme['window']))
        palette.setColor(QPalette.Base, QColor(theme['surface_alt']))
        palette.setColor(QPalette.Button, QColor(theme['surface']))
        palette.setColor(QPalette.ButtonText, QColor(theme['text']))
        palette.setColor(QPalette.Text, QColor(theme['text']))
        palette.setColor(QPalette.WindowText, QColor(theme['text']))
        palette.setColor(QPalette.Highlight, QColor(theme['selection']))
        palette.setColor(QPalette.HighlightedText, QColor(theme['text']))
        palette.setColor(QPalette.Disabled, QPalette.Button, QColor(theme['surface_alt']))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(theme['disabled']))
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(theme['disabled']))
        palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(theme['disabled']))
        self.setPalette(palette)

        self.setStyleSheet(
            f'''
            QMainWindow, QWidget#centralWidget {{
                background: {theme['window']};
            }}
            QFrame#canvasFrame {{
                border: 1px solid {theme['border']};
                border-radius: 2px;
                background: {theme['canvas']};
            }}
            QFrame#curationBar, QFrame#activityFrame {{
                border: 1px solid {theme['border']};
                border-radius: 3px;
                background: {theme['surface']};
            }}
            QFrame#curationHistoryDivider {{
                border: none;
                background: {theme['border']};
            }}
            QFrame#stageHeader {{
                border: none;
                background: transparent;
            }}
            QSplitter::handle {{
                background: transparent;
            }}
            QSplitter::handle:hover {{
                background: {theme['surface_hover']};
                border-radius: 2px;
            }}
            QSplitter::handle:horizontal {{
                width: 7px;
            }}
            QSplitter::handle:vertical {{
                height: 7px;
            }}
            QWidget#controlsPanel, QWidget#mainPane, QWidget#canvasStage {{
                background: transparent;
            }}
            QWidget#resourcesPanel, QWidget#persistentPanel {{
                border: 1px solid {theme['border']};
                border-radius: 3px;
                background: {theme['surface']};
            }}
            QScrollArea, QWidget#tabContent, QWidget#labelTab {{
                border: none;
                background: {theme['surface']};
            }}
            QFrame#labelActionBar {{
                border: none;
                border-top: 1px solid {theme['border']};
                background: {theme['surface']};
            }}
            QTabWidget::pane {{
                border: 1px solid {theme['border']};
                border-radius: 3px;
                background: {theme['surface']};
            }}
            QTabBar::tab {{
                background: {theme['surface_alt']};
                border: 1px solid {theme['border']};
                border-bottom: none;
                padding: 5px 9px;
                margin-right: 2px;
                border-top-left-radius: 3px;
                border-top-right-radius: 3px;
                color: {theme['muted']};
                min-width: 58px;
            }}
            QTabBar::tab:hover {{
                background: {theme['surface_hover']};
                color: {theme['text']};
            }}
            QTabBar::tab:selected {{
                background: {theme['surface']};
                border-color: {theme['border_strong']};
                color: {theme['text']};
            }}
            QLabel {{
                color: {theme['text']};
            }}
            QLabel#stateSummary {{
                color: {theme['text']};
                padding: 0;
            }}
            QLabel#panelValue, QLabel#canvasHint, QLabel#emptyState {{
                color: {theme['muted']};
                padding: 0;
            }}
            QLabel#sectionHeading {{
                color: {theme['text']};
                border-left: 2px solid {theme['border_strong']};
                border-bottom: 1px solid {theme['border']};
                padding: 3px 0 4px 6px;
            }}
            QWidget#labelTab QLabel#sectionHeading {{
                background: {theme['surface_alt']};
            }}
            QLabel#pathOverlay {{
                background: {theme['surface_alt']};
                color: {theme['text']};
            }}
            QLineEdit, QDoubleSpinBox, QSpinBox {{
                border: 1px solid {theme['border']};
                border-radius: 2px;
                padding: 3px 6px;
                background: {theme['surface_alt']};
                color: {theme['text']};
                selection-background-color: {theme['selection']};
                min-height: 23px;
            }}
            QLineEdit:hover, QDoubleSpinBox:hover, QSpinBox:hover {{
                border-color: {theme['border_strong']};
                background: {theme['surface']};
            }}
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
                border-color: {theme['text']};
                background: {theme['surface']};
            }}
            QLineEdit:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled {{
                border-color: {theme['border']};
                background: {theme['surface_alt']};
                color: {theme['disabled']};
            }}
            QCheckBox {{
                spacing: 6px;
                color: {theme['text']};
            }}
            QCheckBox:hover {{
                color: {theme['text']};
            }}
            QCheckBox:focus {{
                color: {theme['text']};
            }}
            QCheckBox:disabled {{
                color: {theme['disabled']};
            }}
            QToolButton {{
                border: 1px solid transparent;
                border-radius: 2px;
                padding: 2px 5px;
                background: transparent;
                color: {theme['muted']};
            }}
            QToolButton:hover, QToolButton:focus {{
                border-color: {theme['border']};
                background: {theme['surface_hover']};
                color: {theme['text']};
            }}
            QToolButton::menu-indicator {{
                image: none;
            }}
            QMenu {{
                border: 1px solid {theme['border_strong']};
                background: {theme['surface']};
                color: {theme['text']};
            }}
            QMenu::item {{
                padding: 4px 18px 4px 8px;
            }}
            QMenu::item:selected {{
                background: {theme['surface_hover']};
            }}
            QPushButton {{
                border: 1px solid {theme['border']};
                border-radius: 2px;
                padding: 3px 6px;
                background: {theme['surface']};
                color: {theme['text']};
                min-height: 23px;
            }}
            QPushButton:hover {{
                background: {theme['surface_hover']};
                border-color: {theme['border_strong']};
            }}
            QPushButton:pressed {{
                background: {theme['surface_strong']};
            }}
            QPushButton:focus {{
                border: 2px solid {theme['text']};
                padding: 2px 5px;
            }}
            QPushButton[role='primary'] {{
                background: {theme['primary']};
                border-color: {theme['primary']};
                color: {theme['primary_text']};
            }}
            QPushButton[role='primary']:hover {{
                background: {theme['primary_hover']};
                border-color: {theme['primary_hover']};
            }}
            QPushButton[role='primary']:disabled {{
                background: {theme['surface_alt']};
                border-color: {theme['border']};
                color: {theme['disabled']};
            }}
            QPushButton[role='danger']:hover,
            QPushButton[role='danger']:focus,
            QPushButton[role='dangerQuiet']:hover,
            QPushButton[role='dangerQuiet']:focus {{
                background: {theme['danger']};
                border-color: {theme['danger']};
                color: {theme['danger_text']};
            }}
            QPushButton[role='dangerQuiet'] {{
                background: transparent;
                border-color: transparent;
                color: {theme['muted']};
            }}
            QPushButton[role='secondary'] {{
                background: {theme['surface']};
                border-color: {theme['border']};
                color: {theme['text']};
            }}
            QPushButton[role='secondary']:hover {{
                background: {theme['surface_hover']};
                border-color: {theme['border_strong']};
            }}
            QPushButton[role='quiet'] {{
                background: transparent;
                border-color: transparent;
                color: {theme['muted']};
            }}
            QPushButton[role='quiet']:hover {{
                background: {theme['surface_hover']};
                border-color: {theme['border']};
                color: {theme['text']};
            }}
            QPushButton[role='small'] {{
                min-height: 20px;
                padding: 2px 7px;
                color: {theme['muted']};
            }}
            QPushButton[role='viewMode'] {{
                min-height: 20px;
                padding: 2px 7px;
                background: transparent;
                color: {theme['muted']};
            }}
            QPushButton[role='viewMode']:checked {{
                background: {theme['surface_strong']};
                border-color: {theme['border_strong']};
                color: {theme['text']};
            }}
            QPushButton:disabled {{
                background: {theme['surface_alt']};
                border-color: {theme['border']};
                color: {theme['disabled']};
            }}
            QSlider::groove:horizontal {{
                height: 3px;
                border: none;
                background: {theme['border']};
            }}
            QSlider::handle:horizontal {{
                width: 11px;
                margin: -5px 0;
                border: 1px solid {theme['border_strong']};
                border-radius: 2px;
                background: {theme['text']};
            }}
            QSlider::groove:horizontal:disabled,
            QSlider::handle:horizontal:disabled {{
                background: {theme['surface_strong']};
                border-color: {theme['border']};
            }}
            QTableWidget#roiTable {{
                border: 1px solid {theme['border']};
                border-radius: 2px;
                background: {theme['surface_alt']};
                color: {theme['text']};
                selection-background-color: {theme['selection']};
                selection-color: {theme['text']};
            }}
            QTableWidget#roiTable::item {{
                border: none;
                padding: 3px 5px;
            }}
            QHeaderView::section {{
                border: none;
                border-bottom: 1px solid {theme['border']};
                padding: 3px 5px;
                background: {theme['surface']};
                color: {theme['muted']};
            }}
            QPlainTextEdit#logBox {{
                border: none;
                border-radius: 2px;
                background: {theme['surface_alt']};
                color: {theme['text']};
                padding: 6px;
                selection-background-color: {theme['selection']};
            }}
            QToolTip {{
                border: 1px solid {theme['border_strong']};
                border-radius: 2px;
                padding: 4px 6px;
                background: {theme['surface']};
                color: {theme['text']};
            }}
            QStatusBar {{
                background: {theme['surface']};
                color: {theme['muted']};
                border-top: 1px solid {theme['border']};
            }}
            '''
            )
        self.refresh_status()

    def set_roi_overlay_visible(self, state):
        self.roi_overlay_check.setText('ROI on' if state else 'ROI off')
        self.plot_image(preserve_view=True)

    def set_dark_mode(self, state):
        self.dark_mode = bool(state)
        self._apply_palette_and_style()
        self.plot_image(preserve_view=True)

    def set_interface_font_size(self, point_size):
        if point_size not in GUI_FONT_SIZES:
            raise ValueError(f'unsupported interface font size: {point_size}')

        text_widget_types = (
            QAbstractButton,
            QAbstractSpinBox,
            QHeaderView,
            QLabel,
            QLineEdit,
            QMenu,
            QPlainTextEdit,
            QStatusBar,
            QTabBar,
            QTableView,
            )
        for widget in self.findChildren(QWidget):
            if not isinstance(widget, text_widget_types):
                continue
            bold = widget.font().weight() == QFont.Bold
            widget.setFont(load_gui_font(size=point_size, bold=bold))
        QToolTip.setFont(load_gui_font(size=point_size))
        for action in self.interface_font_menu.actions():
            action.setFont(load_gui_font(size=point_size))
        for column in range(self.roi_table.columnCount()):
            self.roi_table.horizontalHeaderItem(column).setFont(
                load_gui_font(size=point_size)
                )
        self.interface_font_size = point_size
        self.interface_font_actions[point_size].setChecked(True)
        self.plot_image(preserve_view=True)
        self.controls_split_timer.start(0)
        self.schedule_curation_layout()

    @staticmethod
    def selected_device():
        return 'auto'

    def prediction_inputs_changed(self, _text=None):
        self.invalidate_probability()
        self.refresh_status()

    def invalidate_probability(self, redraw=True):
        needs_redraw = self.probability is not None or self.display_mode == 'confidence'
        self.probability = None
        self.display_mode = 'image'
        if hasattr(self, 'image_view_button'):
            blockers = [
                QSignalBlocker(self.image_view_button),
                QSignalBlocker(self.confidence_view_button),
                ]
            self.image_view_button.setChecked(True)
            self.confidence_view_button.setChecked(False)
            self.confidence_view_button.setEnabled(False)
            self.confidence_view_button.setToolTip(
                'available after prediction produces a confidence map'
                )
            del blockers
            self.update_display_control_state()
        if redraw and needs_redraw:
            self.plot_image(preserve_view=True)

    def set_display_mode(self, mode):
        if mode == 'confidence' and self.probability is None:
            mode = 'image'
        if mode not in {'image', 'confidence'}:
            raise ValueError(f'unknown display mode: {mode}')

        self.display_mode = mode
        blockers = [
            QSignalBlocker(self.image_view_button),
            QSignalBlocker(self.confidence_view_button),
            ]
        self.image_view_button.setChecked(mode == 'image')
        self.confidence_view_button.setChecked(mode == 'confidence')
        del blockers
        self.update_display_control_state()
        self.plot_image(preserve_view=True)

    def update_display_control_state(self):
        image_controls_enabled = (
            self.ref_image is not None and
            self.display_mode == 'image'
            )
        for widget in (
            self.black_slider,
            self.white_slider,
            self.black_value,
            self.white_value,
            self.reset_display_button,
        ):
            widget.setEnabled(image_controls_enabled)

    def display_slider_changed(self, _value=None):
        sender = self.sender()
        if sender is self.black_slider:
            self.set_display_range(
                self.black_slider.value() / 10,
                self.display_white,
                changed='black',
                )
        elif sender is self.white_slider:
            self.set_display_range(
                self.display_black,
                self.white_slider.value() / 10,
                changed='white',
                )

    def display_spin_changed(self, _value=None):
        sender = self.sender()
        if sender is self.black_value:
            self.set_display_range(
                self.black_value.value(),
                self.display_white,
                changed='black',
                )
        elif sender is self.white_value:
            self.set_display_range(
                self.display_black,
                self.white_value.value(),
                changed='white',
                )

    def set_display_range(
            self,
            black_percentile,
            white_percentile,
            changed=None,
            schedule_redraw=True,
            ):
        black_percentile = float(np.clip(black_percentile, 0, 100))
        white_percentile = float(np.clip(white_percentile, 0, 100))
        if changed == 'black':
            black_percentile = min(
                black_percentile,
                white_percentile - DISPLAY_MIN_GAP,
                )
        elif changed == 'white':
            white_percentile = max(
                white_percentile,
                black_percentile + DISPLAY_MIN_GAP,
                )
        else:
            black_percentile = min(
                black_percentile,
                100 - DISPLAY_MIN_GAP,
                )
            white_percentile = max(
                white_percentile,
                black_percentile + DISPLAY_MIN_GAP,
                )

        black_percentile = float(np.clip(
            black_percentile,
            0,
            100 - DISPLAY_MIN_GAP,
            ))
        white_percentile = float(np.clip(
            white_percentile,
            DISPLAY_MIN_GAP,
            100,
            ))
        if white_percentile - black_percentile < DISPLAY_MIN_GAP:
            if changed == 'black':
                black_percentile = max(
                    0,
                    white_percentile - DISPLAY_MIN_GAP,
                    )
            else:
                white_percentile = min(
                    100,
                    black_percentile + DISPLAY_MIN_GAP,
                    )
                black_percentile = min(
                    black_percentile,
                    white_percentile - DISPLAY_MIN_GAP,
                    )

        self.display_black = round(black_percentile, 1)
        self.display_white = round(white_percentile, 1)
        blockers = [
            QSignalBlocker(self.black_slider),
            QSignalBlocker(self.white_slider),
            QSignalBlocker(self.black_value),
            QSignalBlocker(self.white_value),
            ]
        self.black_slider.setValue(int(round(self.display_black * 10)))
        self.white_slider.setValue(int(round(self.display_white * 10)))
        self.black_value.setValue(self.display_black)
        self.white_value.setValue(self.display_white)
        del blockers

        if schedule_redraw and self.ref_image is not None:
            self.display_redraw_timer.start(DISPLAY_REDRAW_MS)

    def reset_display_range(self, redraw=True):
        self.display_redraw_timer.stop()
        self.set_display_range(
            DISPLAY_BLACK_DEFAULT,
            DISPLAY_WHITE_DEFAULT,
            schedule_redraw=False,
            )
        if redraw and self.ref_image is not None:
            self.plot_image(preserve_view=True)

    def roi_colour_map(self):
        roi_ids = sorted(int(roi_id) for roi_id in self.roi_dict)
        colours = generate_distinct_colours(len(roi_ids))
        return dict(zip(roi_ids, colours))

    def refresh_roi_table(self):
        if not hasattr(self, 'roi_table'):
            return

        self._syncing_roi_table = True
        blocker = QSignalBlocker(self.roi_table)
        roi_ids = sorted(int(roi_id) for roi_id in self.roi_dict)
        colour_map = self.roi_colour_map()
        current_ids = [
            (
                int(self.roi_table.item(row, 1).data(Qt.UserRole))
                if self.roi_table.item(row, 1) is not None else None
                )
            for row in range(self.roi_table.rowCount())
            ]
        current_row = self.roi_table.currentRow()
        current_id = (
            current_ids[current_row]
            if 0 <= current_row < len(current_ids) else None
            )
        rebuild = current_ids != roi_ids
        if rebuild:
            self.roi_table.setRowCount(len(roi_ids))

        for row, roi_id in enumerate(roi_ids):
            roi = self.roi_dict[roi_id]
            colour = QColor.fromRgbF(*colour_map[roi_id])
            if rebuild:
                colour_item = QTableWidgetItem()
                id_item = QTableWidgetItem(str(roi_id))
                pixel_item = QTableWidgetItem()
                fixed_item = QTableWidgetItem()
                self.roi_table.setItem(row, 0, colour_item)
                self.roi_table.setItem(row, 1, id_item)
                self.roi_table.setItem(row, 2, pixel_item)
                self.roi_table.setItem(row, 3, fixed_item)
                swatch = QLabel('■')
                swatch.setObjectName('roiColourSwatch')
                swatch.setAlignment(Qt.AlignCenter)
                swatch.setAttribute(Qt.WA_TransparentForMouseEvents)
                self.roi_table.setCellWidget(row, 0, swatch)
            else:
                colour_item = self.roi_table.item(row, 0)
                id_item = self.roi_table.item(row, 1)
                pixel_item = self.roi_table.item(row, 2)
                fixed_item = self.roi_table.item(row, 3)
                swatch = self.roi_table.cellWidget(row, 0)

            colour_item.setBackground(QBrush(colour))
            colour_item.setForeground(QBrush(colour))
            colour_item.setToolTip(f'ROI {roi_id} colour')
            swatch.setStyleSheet(
                f'color: {colour.name()}; background: transparent; border: none;'
                )
            swatch.setToolTip(f'ROI {roi_id} colour')
            id_item.setText(str(roi_id))
            id_item.setData(Qt.UserRole, roi_id)
            id_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            pixel_item.setText(str(len(np.asarray(roi['xpix']).ravel())))
            pixel_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            fixed_item.setText('yes' if roi_id in self.fixed_ids else '')
            fixed_item.setTextAlignment(Qt.AlignCenter)

        self.selected.intersection_update(roi_ids)
        table_selected = {
            int(item.data(Qt.UserRole))
            for item in self.roi_table.selectedItems()
            if item.column() == 1 and item.data(Qt.UserRole) is not None
            }
        if rebuild or table_selected != self.selected:
            # Qt keeps a private Shift-selection anchor beyond the current index
            self.roi_table.setCurrentItem(None)
            self.roi_table.clearSelection()
            for row, roi_id in enumerate(roi_ids):
                if roi_id not in self.selected:
                    continue
                for column in range(self.roi_table.columnCount()):
                    self.roi_table.item(row, column).setSelected(True)
            if self.selected:
                anchor_id = (
                    current_id
                    if current_id in self.selected else
                    min(self.selected)
                    )
                anchor_row = roi_ids.index(anchor_id)
                self.roi_table.setCurrentCell(
                    anchor_row,
                    1,
                    QItemSelectionModel.NoUpdate,
                    )
            else:
                self.roi_table.setCurrentItem(None)

        self.roi_empty_label.setVisible(not roi_ids)
        del blocker
        self._syncing_roi_table = False

    def roi_table_selection_changed(self):
        if self._syncing_roi_table:
            return
        selected = {
            int(item.data(Qt.UserRole))
            for item in self.roi_table.selectedItems()
            if item.column() == 1 and item.data(Qt.UserRole) is not None
            }
        self.selected = selected
        self.plot_image(preserve_view=True)
        self.refresh_status()

    def centre_roi_from_table(self, row, _column):
        item = self.roi_table.item(row, 1)
        if item is None:
            return
        roi_id = int(item.data(Qt.UserRole))
        roi = self.roi_dict.get(roi_id)
        if roi is None:
            return

        xpix = np.asarray(roi['xpix'])
        ypix = np.asarray(roi['ypix'])
        if xpix.size == 0 or ypix.size == 0:
            return
        self.canvas.centre_on(float(np.mean(xpix)), float(np.mean(ypix)))
        self.refresh_status(f'centred ROI {roi_id}')


    #%% browse
    def browse_source_root(self):
        path = QFileDialog.getExistingDirectory(self, 'select training source root', self.source_root_line.text())
        if path:
            self.source_root_line.setText(path)

    def browse_manifest_out(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            'select dataset table output',
            self.manifest_line.text(),
            'CSV files (*.csv)',
            )
        if path:
            self.manifest_line.setText(path)

    def browse_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            'select training recipe',
            self.config_line.text(),
            'YAML files (*.yaml *.yml)',
            )
        if path:
            self.config_line.setText(path)

    def browse_image(self):
        start_path = self.image_line.text().strip() or str(WORKSPACE_ROOT)
        path, _ = QFileDialog.getOpenFileName(
            self,
            'select channel-2 reference image',
            start_path,
            'NumPy images (*.npy)',
            )
        if path:
            self.image_line.setText(path)
        return path

    def browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            'select trained model',
            str(default_output_root() / 'runs'),
            'Trained models (*.pt)',
            )
        if path:
            self.checkpoint_line.setText(path)

    #%% training processes
    def build_manifest(self):
        source_root = Path(self.source_root_line.text().strip())
        if not source_root.exists():
            self.print_log(f'labelled sessions folder does not exist: {source_root}')
            self.refresh_status('labelled sessions missing')
            return

        args = [
            '--source-root', str(source_root),
            '--out', self.manifest_line.text().strip(),
            '--val-fraction', str(self.val_fraction_spin.value()),
            '--test-fraction', str(self.test_fraction_spin.value()),
            ]
        self.start_process('manifest', 'build_manifest', args)

    def train_model(self):
        manifest = Path(self.manifest_line.text().strip())
        config = Path(self.config_line.text().strip())
        if not manifest.exists():
            source_root = Path(self.source_root_line.text().strip())
            if not source_root.exists():
                self.print_log(f'labelled sessions folder does not exist: {source_root}')
                self.refresh_status('labelled sessions missing')
                return
            self.print_log('dataset table not found; scanning labelled sessions first')
            self.build_manifest()
            self.print_log('scan started; click train model again after it finishes')
            return
        if not config.exists():
            self.print_log(f'training recipe does not exist: {config}')
            self.refresh_status('training recipe missing')
            return

        try:
            config_path = self.write_training_config()
        except Exception as exc:
            self.print_log(f'failed to write training recipe: {exc}')
            return

        run_name = self.run_name_line.text().strip()
        checkpoint_path = default_output_root() / 'runs' / run_name / 'best.pt'
        args = ['--config', str(config_path)]
        if self.start_process('train', 'train_unet', args):
            self.pending_checkpoint_path = checkpoint_path

    def evaluate_model(self):
        model_path = self.current_model_path()
        if not model_path:
            self.print_log('please train or load a model first')
            return
        if not Path(model_path).exists():
            self.print_log(f'trained model does not exist: {model_path}')
            self.refresh_status('model file missing')
            return
        manifest = Path(self.manifest_line.text().strip())
        if not manifest.exists():
            self.print_log(f'dataset table does not exist yet: {manifest}')
            self.refresh_status('dataset table missing')
            return

        args = [
            '--manifest', str(manifest),
            '--checkpoint', model_path,
            '--split', 'test',
            '--threshold', str(self.threshold_spin.value()),
            '--min-size', str(self.min_size_spin.value()),
            '--tta',
            '--device', self.selected_device(),
            ]
        self.start_process('evaluate', 'evaluate', args)

    def preview_training_labels(self):
        manifest = Path(self.manifest_line.text().strip())
        if not manifest.exists():
            self.print_log(f'dataset table does not exist yet: {manifest}')
            self.refresh_status('dataset table missing')
            return

        args = [
            '--manifest', str(manifest),
            '--split', 'train',
            '--n', '6',
            ]
        self.start_process('preview labels', 'plot_training_data', args)

    def preview_model_predictions(self):
        manifest = Path(self.manifest_line.text().strip())
        model_path_text = self.current_model_path()
        if not manifest.exists():
            self.print_log(f'dataset table does not exist yet: {manifest}')
            self.refresh_status('dataset table missing')
            return
        if not model_path_text:
            self.print_log('please train or load a model first')
            self.refresh_status('model missing')
            return
        model_path = Path(model_path_text)
        if not model_path.exists():
            self.print_log(f'trained model does not exist: {model_path}')
            self.refresh_status('model file missing')
            return

        args = [
            '--manifest', str(manifest),
            '--split', 'test',
            '--checkpoint', str(model_path),
            '--threshold', str(self.threshold_spin.value()),
            '--min-size', str(self.min_size_spin.value()),
            '--n', '4',
            '--device', self.selected_device(),
            ]
        self.start_process('preview predictions', 'plot_model_diagnostics', args)

    def inspect_manifest(self):
        path = Path(self.manifest_line.text().strip())
        if not path.exists():
            self.print_log(f'dataset table not found: {path}')
            self.refresh_status('dataset table missing')
            return

        try:
            import csv

            with open(path, 'r', newline='') as f:
                rows = list(csv.DictReader(f))
        except Exception as exc:
            self.print_log(f'failed to read dataset table: {exc}')
            self.refresh_status('dataset table read failed')
            return

        included = [
            row for row in rows
            if str(row.get('included', '')).lower() in {'true', '1', 'yes', 'y'}
            ]
        split_counts = {}
        for row in included:
            split = row.get('split', '') or 'unsplit'
            split_counts[split] = split_counts.get(split, 0) + 1

        excluded_reasons = {}
        for row in rows:
            if str(row.get('included', '')).lower() in {'true', '1', 'yes', 'y'}:
                continue
            reason = row.get('exclusion_reason', '') or 'not included'
            excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1

        roi_total = sum(int(float(row.get('roi_count') or 0)) for row in included)
        self.print_log(f'\ndataset table: {path}')
        self.print_log(f'total sessions: {len(rows)}')
        self.print_log(f'included sessions: {len(included)} | total ROIs: {roi_total}')
        self.print_log(f'splits: {split_counts if split_counts else {}}')
        if excluded_reasons:
            self.print_log(f'excluded: {excluded_reasons}')
        self.refresh_status('dataset summary ready')

    def write_training_config(self):
        config = load_config(self.config_line.text())
        run_name = self.run_name_line.text().strip()
        if not run_name:
            raise ValueError('run name cannot be empty')

        # leave the baseline YAML alone whilst trying controls here; save this recipe beside the run
        config.setdefault('data', {})
        config.setdefault('train', {})
        config.setdefault('postprocess', {})

        config['data']['manifest'] = self.path_for_config(self.manifest_line.text())
        config['train']['run_name'] = run_name
        config['train']['epochs'] = int(self.epochs_spin.value())
        config['train']['device'] = self.selected_device()
        config['postprocess']['threshold'] = float(self.threshold_spin.value())
        config['postprocess']['min_size'] = int(self.min_size_spin.value())

        out_path = default_output_root() / 'gui_configs' / f'{run_name}.yaml'
        save_config(config, out_path)
        self.print_log(f'training recipe written to {out_path}')
        return out_path

    def start_process(self, process_name, module_name, args):
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            self.print_log('another process is already running')
            return False

        self.current_process_name = process_name
        self._process_stdout_buffer = ''
        self._process_stderr_buffer = ''
        self._process_was_stopped = False
        self.process = QProcess(self)
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        self.process.setWorkingDirectory(str(WORKSPACE_ROOT))
        self.process.setProgram(sys.executable)
        command_args = ['-u', '-m', f'fibre_sight.{module_name}'] + args
        self.process.setArguments(command_args)
        self.process.readyReadStandardOutput.connect(self.read_process_stdout)
        self.process.readyReadStandardError.connect(self.read_process_stderr)
        self.process.finished.connect(self.process_finished)

        command_text = shlex.join(['python'] + command_args)
        self.print_log(f'\n$ {command_text}')
        self.process.start()
        # the log keeps the command and output; the status bar only confirms that it started
        self.refresh_status(f'{process_name} started')
        return True

    def stop_process(self):
        if self.process is None or self.process.state() == QProcess.NotRunning:
            self.print_log('no process is running')
            return

        self._process_was_stopped = True
        self.process.kill()
        self.print_log('process stopped')

    def read_process_stdout(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(errors='replace')
        self.buffer_process_text('_process_stdout_buffer', text)

    def read_process_stderr(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode(errors='replace')
        self.buffer_process_text('_process_stderr_buffer', text)

    def buffer_process_text(self, buffer_name, text):
        buffered = getattr(self, buffer_name) + text
        lines = buffered.split('\n')
        setattr(self, buffer_name, lines.pop())
        for line in lines:
            self.print_log(line[:-1] if line.endswith('\r') else line)

    def flush_process_buffers(self):
        for buffer_name in (
                '_process_stdout_buffer',
                '_process_stderr_buffer',
                ):
            text = getattr(self, buffer_name)
            if text:
                self.print_log(text)
            setattr(self, buffer_name, '')

    def process_finished(self, exit_code, exit_status):
        self.read_process_stdout()
        self.read_process_stderr()
        self.flush_process_buffers()
        process_name = self.current_process_name or 'process'
        outcome = 'stopped' if self._process_was_stopped else f'exit code {exit_code}'
        self.print_log(f'\n{process_name} finished: {outcome}')
        training_succeeded = (
            self.current_process_name == 'train' and
            not self._process_was_stopped and
            exit_code == 0 and
            exit_status == QProcess.NormalExit and
            self.pending_checkpoint_path is not None and
            self.pending_checkpoint_path.exists()
            )
        if training_succeeded:
            self.checkpoint_line.setText(str(self.pending_checkpoint_path))
            self.last_saved_model_path = self.pending_checkpoint_path
            self.predictor = None
            self.invalidate_probability()
            self.print_log(f'trained model ready: {self.pending_checkpoint_path}')

        self.current_process_name = None
        self.pending_checkpoint_path = None
        self._process_was_stopped = False
        self.process = None
        self.refresh_status(f'{process_name} finished')

    #%% model prediction
    def choose_channel_image(self):
        if self.browse_image():
            self.load_channel_image()

    def load_edited_channel_image(self):
        image_text = self.image_line.text().strip()
        if image_text and not self.image_matches(image_text):
            self.load_channel_image()

    def image_matches(self, path):
        if self.ref_image is None or self.image_path is None:
            return False
        return self.image_path.resolve() == Path(path).resolve()

    def image_selection_matches_loaded(self):
        image_text = self.image_line.text().strip()
        return bool(image_text) and self.image_matches(image_text)

    def load_channel_image(self):
        path = self.image_line.text().strip()
        if not path:
            self.browse_image()
            path = self.image_line.text().strip()
        if not path:
            return

        try:
            image = squeeze_image(np.load(path))
        except Exception as exc:
            self.print_log(f'failed to load image: {exc}')
            return

        self.ref_image = image
        self.image_path = Path(path)
        self.recname = self.get_recname(self.image_path)
        self.roi_dict = {}
        self.labelled = np.zeros_like(self.ref_image, dtype=np.int32)
        self.selected.clear()
        self.fixed_ids.clear()
        self.undo_stack.clear()
        self.invalidate_probability(redraw=False)
        self.reset_display_range(redraw=False)

        default_roi = self.default_roi_path()
        if default_roi.exists():
            self.print_log(f'found existing ROI dict: {default_roi}')
            self.print_log('import ROIs to review or continue from them')
        self.update_export_tooltip()
        self.plot_image()
        self.canvas.reset_view()
        self.refresh_status('channel-2 image loaded')

    def load_model(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage('loading trained model')
        QApplication.processEvents()
        try:
            self.predictor = self.make_predictor()
            self.predictor.load()
            self.checkpoint_line.setText(str(self.predictor.checkpoint_path))
        except Exception as exc:
            self.print_log(f'failed to load model: {exc}')
            self.predictor = None
            self.refresh_status('model load failed')
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.last_saved_model_path = self.predictor.checkpoint_path
        self.invalidate_probability()
        self.print_log(f'model loaded from {self.predictor.checkpoint_path}')
        self.refresh_status('model loaded')

    def predict_rois(self):
        image_text = self.image_line.text().strip()
        if not image_text:
            self.browse_image()
            image_text = self.image_line.text().strip()
        if not image_text:
            return

        selected_image_path = Path(image_text)
        if not selected_image_path.exists():
            self.print_log(f'channel-2 image does not exist: {selected_image_path}')
            self.refresh_status('channel-2 image missing')
            return

        if not self.image_matches(selected_image_path):
            self.load_channel_image()
        if not self.image_matches(selected_image_path):
            return

        checkpoint_text = self.checkpoint_line.text().strip()
        checkpoint_path = Path(checkpoint_text) if checkpoint_text else None
        predictor_path = self.predictor.checkpoint_path if self.predictor is not None else None
        predictor_matches = (
            predictor_path is not None and
            checkpoint_path is not None and
            predictor_path.resolve() == checkpoint_path.resolve()
            )
        if not predictor_matches:
            self.load_model()
        if self.predictor is None:
            return

        self.predictor.threshold = float(self.threshold_spin.value())
        self.predictor.min_size = int(self.min_size_spin.value())
        self.predictor.tta = True

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage('running prediction')
        QApplication.processEvents()
        try:
            prediction = self.predictor.predict_image(self.ref_image)
        except Exception as exc:
            self.print_log(f'prediction failed: {exc}')
            self.refresh_status('prediction failed')
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.push_undo_state()
        # prediction returns to the same editable state as an MSER or loaded ROI dict
        self.roi_dict = prediction.roi_dict
        self.labelled = prediction.labelled
        self.probability = prediction.probability
        self.selected.clear()
        self.fixed_ids.clear()

        self.set_display_mode('image')
        self.canvas.reset_view()
        self.print_log(
            f'predicted {len(self.roi_dict)} ROIs '
            f'(threshold {prediction.threshold:.2f}, '
            f'min area {prediction.min_size} px)'
            )
        self.refresh_status()
        self.statusBar().clearMessage()

    def rebuild_rois_from_probability(self):
        if self.probability is None:
            return

        # reuse the confidence map here; changing the threshold should not rerun
        # the model whilst I am deciding which faint fibres to keep
        self.push_undo_state()
        self.roi_dict, self.labelled = probability_to_roi_dict(
            self.probability,
            threshold=float(self.threshold_spin.value()),
            min_size=int(self.min_size_spin.value()),
            )
        self.selected.clear()
        self.fixed_ids.clear()
        self.plot_image(preserve_view=True)
        self.print_log(f'rebuilt {len(self.roi_dict)} ROIs from the confidence map')
        self.refresh_status('ROIs rebuilt from the confidence map')

    def make_predictor(self):
        checkpoint = self.checkpoint_line.text().strip()
        checkpoint_path = Path(checkpoint) if checkpoint else None
        return AxonROIPredictor(
            checkpoint_path=checkpoint_path,
            device=self.selected_device(),
            threshold=float(self.threshold_spin.value()),
            min_size=int(self.min_size_spin.value()),
            tta=True,
            )

    def current_model_path(self):
        return self.checkpoint_line.text().strip()

    #%% roi io
    def load_roi_file(self):
        if not self.image_selection_matches_loaded():
            self.print_log('please load the selected channel-2 image first')
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            'import ROIs',
            str(self.image_path.parent if self.image_path else WORKSPACE_ROOT),
            'NumPy dict (*.npy)',
            )
        if not path:
            return

        try:
            roi_dict = load_roi_dict(path)
            labelled, _, _ = roi_dict_to_label(roi_dict, self.ref_image.shape)
        except Exception as exc:
            self.print_log(f'failed to import ROIs: {exc}')
            return

        self.push_undo_state()
        self.roi_dict = labels_to_roi_dict(labelled)
        self.labelled = labelled
        self.selected.clear()
        self.fixed_ids.clear()
        self.invalidate_probability(redraw=False)
        self.plot_image()
        self.print_log(f'imported ROIs from {path}')
        self.refresh_status('ROIs imported')

    def save_roi_file(self):
        if not self.image_selection_matches_loaded():
            self.print_log('please load the selected channel-2 image first')
            return

        self.update_roi_dict()
        out_path = self.default_roi_path()
        save_roi_dict(self.roi_dict, out_path)
        self.print_log(f'exported ROIs to {out_path}')
        self.update_export_tooltip()
        self.refresh_status('ROIs exported')

    def default_roi_path(self):
        if self.image_path is None:
            return default_output_root() / 'predicted_ROI_dict.npy'

        example_root = package_path().parents[1] / 'examples'
        if (
            example_root.is_dir() and
            self.image_path.resolve().is_relative_to(example_root.resolve())
        ):
            # shipped examples stay unchanged; their edited output belongs in workspace
            return default_output_root() / 'demo_predicted_ROI_dict.npy'

        return self.image_path.parent / f'{self.recname}_ROI_dict.npy'

    def update_export_tooltip(self):
        if not hasattr(self, 'curate_buttons'):
            return
        self.curate_buttons['save_roi'].setToolTip(
            f'export immediately to {self.default_roi_path()}'
            )


    #%% mser segmentation
    def reset_segment_parameters(self):
        for spec in PARAMETER_SPECS:
            self.segment_param_widgets[spec['name']].setValue(spec['default'])
        self.refresh_status('parameters reset')

    def get_segment_params(self):
        params = {}
        for spec in PARAMETER_SPECS:
            value = self.segment_param_widgets[spec['name']].value()
            params[spec['name']] = int(value) if spec['kind'] == 'int' else float(value)
        return params

    def get_fixed_roi_dict(self):
        if not self.roi_dict or not self.fixed_ids:
            return {}
        return {
            roi_id: {
                'xpix': self.roi_dict[roi_id]['xpix'].copy(),
                'ypix': self.roi_dict[roi_id]['ypix'].copy(),
            }
            for roi_id in sorted(self.fixed_ids)
            if roi_id in self.roi_dict
        }

    def segment_rois(self):
        if not self.image_selection_matches_loaded():
            self.print_log('please load the selected channel-2 image first')
            return

        if self.labelled is not None:
            self.push_undo_state()

        # the MSER route still helps before a model exists and on images that need hand repair
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage('running MSER segmentation')
        QApplication.processEvents()
        try:
            roi_dict, labelled, fixed_ids, stats = run_mser_segmentation(
                self.ref_image,
                self.get_segment_params(),
                fixed_rois=self.get_fixed_roi_dict(),
            )
        except Exception as exc:
            self.print_log(f'segmentation failed: {exc}')
            self.refresh_status('segmentation failed')
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.roi_dict = roi_dict
        self.labelled = labelled
        self.fixed_ids = fixed_ids
        self.selected.clear()
        self.invalidate_probability(redraw=False)
        self.plot_image()
        self.canvas.reset_view()
        mser_regions = stats['MSER regions']
        kept_rois = stats['kept ROIs']
        fixed_rois = stats['fixed ROIs']
        self.print_log(
            f'MSER regions: {mser_regions} | kept ROIs: {kept_rois} '
            f'(fixed {fixed_rois})'
        )
        self.refresh_status('segmentation complete')

    def toggle_selected_fixed(self):
        selected_ids = self.selected.intersection(self.roi_dict)
        if not selected_ids:
            return
        if selected_ids.issubset(self.fixed_ids):
            self.unfix_selected()
        else:
            self.fix_selected()

    def fix_selected(self):
        if not self.selected:
            return
        self.fixed_ids.update(roi_id for roi_id in self.selected if roi_id in self.roi_dict)
        self.plot_image(preserve_view=True)
        self.refresh_status('selected ROIs fixed')

    def unfix_selected(self):
        if not self.selected:
            return
        self.fixed_ids.difference_update(self.selected)
        self.plot_image(preserve_view=True)
        self.refresh_status('selected ROIs unfixed')

    def clear_fixed(self):
        if not self.fixed_ids:
            return
        self.fixed_ids.clear()
        self.plot_image(preserve_view=True)
        self.refresh_status('fixed marks cleared')

    def clear_unfixed(self):
        if self.labelled is None:
            return
        self.push_undo_state()
        if self.fixed_ids:
            keep_mask = np.isin(self.labelled, list(self.fixed_ids))
            self.labelled[~keep_mask] = 0
        else:
            self.labelled = np.zeros_like(self.labelled, dtype=np.int32)
        self.selected.clear()
        self.compact_labels()
        self.update_roi_dict()
        self.plot_image(preserve_view=True)
        self.refresh_status('unfixed ROIs cleared')

    #%% selection and pruning
    def on_click(self, event):
        if event.button != 1:
            return
        if self.canvas.consume_dragged_release():
            return
        if self.ref_image is None:
            if event.inaxes == self.ax:
                self.load_channel_image()
            return
        if event.inaxes != self.ax or self.labelled is None:
            return
        if event.xdata is None or event.ydata is None:
            return

        xpix = int(round(event.xdata))
        ypix = int(round(event.ydata))
        if not self.in_bounds(ypix, xpix):
            return

        roi_id = int(self.labelled[ypix, xpix])
        if roi_id <= 0:
            return

        modifiers = Qt.NoModifier
        if getattr(event, 'guiEvent', None) is not None:
            modifiers = event.guiEvent.modifiers()

        if modifiers & Qt.ShiftModifier:
            self.selected.add(roi_id)
        elif modifiers & (Qt.ControlModifier | Qt.MetaModifier):
            if roi_id in self.selected:
                self.selected.remove(roi_id)
            else:
                self.selected.add(roi_id)
        else:
            self.selected = {roi_id}

        self.plot_image(preserve_view=True)
        self.refresh_status()

    def select_all(self):
        if self.labelled is None:
            return
        self.selected = {int(label_id) for label_id in np.unique(self.labelled) if label_id > 0}
        self.plot_image(preserve_view=True)
        self.refresh_status('all ROIs selected')

    def invert_selection(self):
        if self.labelled is None:
            return
        ids = {int(label_id) for label_id in np.unique(self.labelled) if label_id > 0}
        self.selected = ids - self.selected
        self.plot_image(preserve_view=True)
        self.refresh_status('selection inverted')

    def clear_selection(self):
        self.selected.clear()
        self.plot_image(preserve_view=True)
        self.refresh_status('selection cleared')

    def delete_selected(self):
        if self.labelled is None or not self.selected:
            return

        self.push_undo_state()
        mask = np.isin(self.labelled, list(self.selected))
        self.labelled[mask] = 0
        self.compact_labels()
        self.selected.clear()
        self.update_roi_dict()
        self.plot_image(preserve_view=True)
        self.refresh_status('selected ROIs deleted')

    def merge_selected(self):
        if self.labelled is None or len(self.selected) < 2:
            return

        self.push_undo_state()
        keep_id = min(self.selected)
        keep_fixed = bool(self.fixed_ids.intersection(self.selected))
        mask = np.isin(self.labelled, list(self.selected))
        self.labelled[mask] = keep_id
        if keep_fixed:
            self.fixed_ids.add(keep_id)
        self.compact_labels()
        self.selected.clear()
        self.update_roi_dict()
        self.plot_image(preserve_view=True)
        self.refresh_status('selected ROIs merged')

    def push_undo_state(self):
        if self.labelled is None:
            return
        self.undo_stack.append((self.labelled.copy(), set(self.selected), set(self.fixed_ids)))
        if len(self.undo_stack) > 30:
            self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack:
            self.print_log('nothing to undo')
            return

        self.labelled, self.selected, self.fixed_ids = self.undo_stack.pop()
        self.update_roi_dict()
        self.plot_image(preserve_view=True)
        self.refresh_status('previous ROI state restored')

    #%% plotting
    def plot_image(self, preserve_view=False):
        xlim, ylim = self.get_current_view()
        theme = self._theme()
        self.refresh_roi_table()
        self.update_display_control_state()
        self.fig.set_facecolor(theme['canvas'])
        self.ax.clear()
        self.ax.set_facecolor(theme['canvas'])
        self.ax.axis('off')

        if self.ref_image is None:
            gui_size = self.interface_font_size
            self.canvas.setCursor(Qt.PointingHandCursor)
            self.canvas.setToolTip('')
            self.ax.text(
                0.5,
                0.53,
                'open a channel-2 image to start',
                transform=self.ax.transAxes,
                ha='center',
                va='center',
                color=theme['muted'],
                fontsize=gui_size + 2,
                fontweight='normal',
                fontfamily=_GUI_FONT_FAMILY,
                )
            self.ax.text(
                0.5,
                0.47,
                'click anywhere on this canvas to browse',
                transform=self.ax.transAxes,
                ha='center',
                va='center',
                color=theme['muted'],
                fontsize=gui_size,
                alpha=0.8,
                fontfamily=_GUI_FONT_FAMILY,
                )
            self.canvas.draw_idle()
            return

        self.canvas.setCursor(Qt.CrossCursor)
        self.canvas.setToolTip('')
        if self.display_mode == 'confidence' and self.probability is not None:
            self.plot_probability()
        else:
            self.display_mode = 'image'
            base = normalise_for_display(
                self.ref_image,
                black_percentile=self.display_black,
                white_percentile=self.display_white,
                )
            self.ax.imshow(
                base,
                cmap='gray',
                vmin=0,
                vmax=1,
                interpolation='nearest',
                )
        if self.roi_overlay_check.isChecked():
            self.plot_roi_overlay()

        if preserve_view and xlim is not None:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        self.canvas.draw_idle()
        self.refresh_status()

    def plot_probability(self):
        probability = np.asarray(self.probability, dtype=np.float32)
        self.ax.imshow(
            probability,
            cmap='gray',
            vmin=0,
            vmax=1,
            interpolation='nearest',
            )

    def plot_roi_overlay(self):
        if self.labelled is None:
            return

        ids = [int(label_id) for label_id in np.unique(self.labelled) if label_id > 0]
        if not ids:
            return

        colour_map = self.roi_colour_map()
        if set(ids) != set(colour_map):
            colours = generate_distinct_colours(len(ids))
            colour_map = dict(zip(ids, colours))

        if self.display_mode == 'confidence':
            for roi_id in ids:
                mask = self.labelled == roi_id
                linewidth = 1.0
                linestyle = 'solid'
                if roi_id in self.fixed_ids and roi_id not in self.selected:
                    linewidth = 1.8
                    linestyle = 'dashed'
                self.plot_mask_outline(
                    mask,
                    colour_map[roi_id],
                    linewidth=linewidth,
                    linestyle=linestyle,
                    )
            self.plot_selection_halo(colour_map)
            return

        overlay = np.zeros((*self.labelled.shape, 4), dtype=np.float32)
        for roi_id in ids:
            mask = self.labelled == roi_id
            overlay[mask, :3] = colour_map[roi_id]
            if roi_id in self.selected:
                overlay[mask, 3] = 0.82
            elif roi_id in self.fixed_ids:
                overlay[mask, 3] = 0.66
            else:
                overlay[mask, 3] = 0.46

        self.ax.imshow(overlay, interpolation='nearest')
        selected_mask = np.isin(self.labelled, list(self.selected))
        fixed_mask = np.isin(self.labelled, list(self.fixed_ids - self.selected))
        if np.any(fixed_mask):
            self.plot_mask_outline(
                fixed_mask,
                '#ffd166',
                linewidth=1.4,
                linestyle='dashed',
                )
        if np.any(selected_mask):
            self.plot_selection_halo(colour_map)

    def plot_selection_halo(self, colour_map):
        selected_ids = [
            roi_id
            for roi_id in colour_map
            if roi_id in self.selected
            ]
        if not selected_ids:
            return

        selected_mask = np.isin(self.labelled, selected_ids)
        self.plot_mask_outline(
            selected_mask,
            SELECTION_OUTER_COLOUR,
            linewidth=SELECTION_OUTER_WIDTH,
            linestyle='solid',
            )
        self.plot_mask_outline(
            selected_mask,
            SELECTION_INNER_COLOUR,
            linewidth=SELECTION_INNER_WIDTH,
            linestyle='solid',
            )
        for roi_id in selected_ids:
            self.plot_mask_outline(
                self.labelled == roi_id,
                colour_map[roi_id],
                linewidth=SELECTION_COLOUR_WIDTH,
                linestyle='solid',
                )

    def plot_mask_outline(self, mask, colour, linewidth, linestyle):
        padded = np.pad(np.asarray(mask, dtype=np.uint8), 1)
        height, width = mask.shape
        self.ax.contour(
            np.arange(-1, width + 1),
            np.arange(-1, height + 1),
            padded,
            levels=[0.5],
            colors=[colour],
            linewidths=linewidth,
            linestyles=linestyle,
            )

    def get_current_view(self):
        if not self.ax.images:
            return None, None
        return self.ax.get_xlim(), self.ax.get_ylim()

    def reset_view(self):
        self.canvas.reset_view()
        self.refresh_status('view fitted')

    def zoom_in(self):
        if self.canvas.zoom_in():
            self.refresh_status('view enlarged')

    def zoom_out(self):
        if self.canvas.zoom_out():
            self.refresh_status('view reduced')

    #%% state helpers
    def update_roi_dict(self):
        if self.labelled is None:
            self.roi_dict = {}
            self.fixed_ids.clear()
            self.refresh_roi_table()
            return
        self.roi_dict = labels_to_roi_dict(self.labelled)
        self.fixed_ids = {roi_id for roi_id in self.fixed_ids if roi_id in self.roi_dict}
        self.refresh_roi_table()

    def compact_labels(self):
        if self.labelled is None:
            return

        out = np.zeros_like(self.labelled, dtype=np.int32)
        remap = {}
        next_id = 1
        for label_id in sorted(np.unique(self.labelled)):
            if label_id == 0:
                continue
            out[self.labelled == label_id] = next_id
            remap[int(label_id)] = next_id
            next_id += 1
        self.labelled = out
        self.fixed_ids = {remap[roi_id] for roi_id in self.fixed_ids if roi_id in remap}

    def in_bounds(self, ypix, xpix):
        return (
            0 <= ypix < self.labelled.shape[0] and
            0 <= xpix < self.labelled.shape[1]
            )

    def refresh_status(self, message=None):
        image_name = self.image_path.name if self.image_path else 'not loaded'
        roi_count = len(self.roi_dict)
        selected_count = len(self.selected)
        fixed_count = len(self.fixed_ids)
        self.state_label.setText(
            image_name if self.image_path else 'image: not loaded'
            )
        self.state_label.setToolTip(
            str(self.image_path) if self.image_path else 'no channel-2 image loaded'
            )
        self.roi_label.setText(
            f'{roi_count} ROIs | {selected_count} selected | {fixed_count} fixed'
            )
        self.model_label.setText(self.model_status_text())
        self.update_export_tooltip()
        self.update_workflow_state()
        if message:
            # keep routine confirmation off the canvas whilst I am selecting and fixing ROIs
            status_bar = self.statusBar()
            status_bar.setFont(load_gui_font(size=self.interface_font_size))
            status_bar.showMessage(message, 4000)

    def model_status_text(self):
        checkpoint = self.checkpoint_line.text().strip()
        if not checkpoint:
            return 'model: not selected'

        checkpoint_path = Path(checkpoint)
        name = checkpoint_path.name
        if not checkpoint_path.exists():
            return f'model: {name} · checkpoint missing'

        next_step = (
            'ready to predict'
            if self.image_selection_matches_loaded()
            else 'awaiting image'
            )
        if self.predictor is None:
            return f'model: {name} · {next_step}'

        try:
            matches = (
                Path(self.predictor.checkpoint_path).resolve() ==
                Path(checkpoint).resolve()
                )
        except (OSError, ValueError):
            matches = False
        if not matches:
            return f'model: {name} · {next_step}'
        return f'model: {name} · {self.predictor.device}'

    def update_workflow_state(self):
        image_text = self.image_line.text().strip()
        image_path_ready = bool(image_text) and Path(image_text).exists()
        image_ready = self.image_selection_matches_loaded()
        checkpoint_text = self.checkpoint_line.text().strip()
        checkpoint_ready = bool(checkpoint_text) and Path(checkpoint_text).exists()
        model_ready = bool(self.current_model_path()) and Path(self.current_model_path()).exists()
        manifest_text = self.manifest_line.text().strip()
        manifest_ready = bool(manifest_text) and Path(manifest_text).exists()
        config_text = self.config_line.text().strip()
        config_ready = bool(config_text) and Path(config_text).exists()
        source_text = self.source_root_line.text().strip()
        source_ready = bool(source_text) and Path(source_text).exists()
        has_rois = self.labelled is not None and bool(np.any(self.labelled > 0))
        process_running = self.process is not None and self.process.state() != QProcess.NotRunning
        selected_ids = self.selected.intersection(self.roi_dict)
        selected_are_fixed = (
            bool(selected_ids) and
            selected_ids.issubset(self.fixed_ids)
            )
        has_fixed = bool(self.fixed_ids)

        self.build_manifest_button.setEnabled(source_ready and not process_running)
        self.inspect_manifest_button.setEnabled(manifest_ready and not process_running)
        self.train_model_button.setEnabled(
            source_ready
            and config_ready
            and bool(self.run_name_line.text().strip())
            and not process_running
            )
        self.evaluate_model_button.setEnabled(manifest_ready and model_ready and not process_running)
        self.preview_training_button.setEnabled(manifest_ready and not process_running)
        self.preview_predictions_button.setEnabled(manifest_ready and model_ready and not process_running)
        self.stop_process_button.setEnabled(process_running)
        self.stop_process_button.setVisible(process_running)

        self.predict_button.setEnabled(
            image_path_ready and checkpoint_ready and not process_running
            )
        probability_ready = self.probability is not None
        self.rebuild_rois_button.setEnabled(
            probability_ready and not process_running
            )
        self.rebuild_rois_button.setVisible(probability_ready)
        self.confidence_view_button.setEnabled(probability_ready)
        self.confidence_view_button.setToolTip(
            'show per-pixel model confidence from 0 to 1'
            if probability_ready else
            'available after prediction produces a confidence map'
            )
        self.update_display_control_state()

        self.segment_button.setEnabled(image_ready)
        self.segment_load_roi_button.setEnabled(image_ready)
        self.fix_selected_button.setText(
            'unfix' if selected_are_fixed else 'fix'
            )
        self.fix_selected_button.setToolTip(
            'allow selected ROIs to change during segmentation'
            if selected_are_fixed else
            'keep selected ROIs when segmentation is run again'
            )
        self.fix_selected_button.setEnabled(bool(selected_ids))
        self.clear_fixed_button.setEnabled(has_fixed)
        self.clear_fixed_button.setVisible(has_fixed)
        self.clear_unfixed_button.setEnabled(has_rois and has_fixed)
        self.clear_unfixed_button.setVisible(has_rois and has_fixed)

        self.curate_buttons['select_all'].setEnabled(has_rois)
        self.curate_buttons['delete'].setEnabled(bool(self.selected))
        self.curate_buttons['merge'].setEnabled(len(self.selected) >= 2)
        self.curate_buttons['undo'].setEnabled(bool(self.undo_stack))
        self.curate_buttons['zoom_out'].setEnabled(image_ready)
        self.curate_buttons['fit_view'].setEnabled(image_ready)
        self.curate_buttons['zoom_in'].setEnabled(image_ready)
        self.curate_buttons['save_roi'].setEnabled(image_ready and has_rois)
        if hasattr(self, 'controls_split_timer'):
            self.controls_split_timer.start(0)

    def print_log(self, text, end='\n'):
        text = format_log_text(text)
        if end:
            text = f'{text}{end}'
        self.output_box.moveCursor(QTextCursor.End)
        self.output_box.insertPlainText(text)
        self.output_box.moveCursor(QTextCursor.End)
        # the log pane starts collapsed; open it far enough to read the newest lines
        if self.activity_splitter.sizes()[1] < 72:
            total = max(sum(self.activity_splitter.sizes()), self.activity_splitter.height(), 400)
            self.activity_splitter.setSizes([total - 105, 105])

    @staticmethod
    def get_recname(path):
        name = Path(path).name
        if '_ref_mat' in name:
            return name.split('_ref_mat')[0]
        return Path(path).stem

    @staticmethod
    def path_for_config(path):
        path = Path(path)
        try:
            return str(path.resolve().relative_to(WORKSPACE_ROOT.resolve()))
        except ValueError:
            return str(path)


#%% entry point
def main():
    app = QApplication.instance()
    if app is None:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(load_gui_font())
    app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    window = FibreSightWorkbench()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
