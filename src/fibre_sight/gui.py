'''
Created on 12 May 2026

Modified on 13 May 2026 whilst wiring the MSER controls into the workbench
Modified on 21 May 2026 to separate labelling, training, prediction, and editing
Modified on 2 June 2026 during the first command-line and diagnostics pass
Modified on 23 June 2026 to bring the MSER editor and model workflow together
Modified on 24 July 2026 to load packaged assets and the bundled model
Modified on 25 July 2026 to separate the workbench tasks and reduce the control density
Modified on 14 August 2026 to simplify the GUI workflow and tests
Modified on 19 August 2026 to curate named NWB proposal runs
Modified on 21 August 2026 to add the automatic recording workflow
Modified on 22 August 2026 to expose piecewise extraction in AUTO

automatic recording analysis, segmentation, training, and ROI curation

@author: Dinghao Luo
'''


#%% imports
from pathlib import Path
import shlex
import sys

import matplotlib

matplotlib.use('Qt5Agg')

import numpy as np
from matplotlib import font_manager
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from pynwb import NWBHDF5IO
from PyQt5.QtCore import (
    QByteArray,
    QItemSelectionModel,
    QProcess,
    QSettings,
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
    QComboBox,
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
    QMessageBox,
    QProgressBar,
    QPushButton,
    QPlainTextEdit,
    QShortcut,
    QSizePolicy,
    QScrollArea,
    QSlider,
    QSplitter,
    QSpinBox,
    QStackedWidget,
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
    OUTPUT_ROOT,
    PACKAGE_ROOT,
    SOURCE_ROOT,
    WORKSPACE_ROOT,
    )
from . import __version__
from .api import (
    BUNDLED_CHECKPOINT,
    BUNDLED_MIN_SIZE,
    BUNDLED_THRESHOLD,
    ROIPredictor,
    list_dff_runs,
    list_fluorescence_runs,
    list_roi_runs,
    load_dff_run,
    load_fluorescence_run,
    load_roi_run,
    save_curated_rois,
    )
from .config import load_recipe, resolve_path, save_recipe
from .gui_canvas import (
    ZoomableCanvas,
    generate_distinct_colours,
    normalise_for_display,
    squeeze_image,
    )
from .gui_worker import (
    SESSION_SCHEMA_VERSION,
    append_session_event,
    fingerprint_paths,
    latest_session_config,
    natural_tiff_paths,
    partial_paths,
    session_stage_states,
    )
from .mser_segmenter import (
    PARAMETER_SPECS,
    PARAMETER_TOOLTIPS,
    segment_mser,
    )
from .manifest import read_manifest
from .postprocess import probability_to_roi_dict
from .roi_io import labels_to_roi_dict, load_roi_dict, roi_dict_to_label, save_roi_dict


APP_ICON_PATH = PACKAGE_ROOT / 'assets' / 'fibresight_icon.ico'
MONONOKI_FONT_DIR = PACKAGE_ROOT / 'assets' / 'fonts' / 'mononoki'
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


#%% main window
class FibreSightGUI(QMainWindow):
    def __init__(self, settings=None):
        app = QApplication.instance()
        if app is not None:
            app.setStyle('Fusion')
        if settings is None:
            settings = QSettings('FibreSight', 'FibreSight')
        interface_font_size = settings.value(
            'interface/font_size',
            GUI_FONT_SIZE,
            type=float,
            )
        gui_font = (
            load_gui_font(size=interface_font_size)
            if app is not None else
            None
            )
        if app is not None and gui_font is not None:
            app.setFont(gui_font)
            QToolTip.setFont(gui_font)
        super().__init__()
        self.settings = settings
        self.setWindowTitle('FibreSight')
        self.setWindowIcon(QIcon(str(APP_ICON_PATH)))
        self.resize(1280, 820)
        self.setMinimumSize(1000, 680)

        self.ref_image = None
        self.image_path = None
        self.nwb_path = None
        self.source_proposal_run = None
        self.recname = None
        self.roi_dict = {}
        self.labelled = None
        self.selected = set()
        self.fixed_ids = set()
        self.undo_stack = []
        self.probability = None
        self.probability_max_size = None
        self.predictor = None
        self.process = None
        self.current_process_name = None
        self.pending_checkpoint_path = None
        self.last_saved_model_path = None
        self.dark_mode = True
        self.display_black = DISPLAY_BLACK_DEFAULT
        self.display_white = DISPLAY_WHITE_DEFAULT
        self.display_mode = 'image'
        self.interface_font_size = interface_font_size
        self._syncing_roi_table = False
        self._process_was_stopped = False
        self.auto_log_path = None
        self.auto_loaded_config = None
        self.trace_cache = None
        self.trace_xlim = None
        self.auto_output_buffer = ''
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
        self.auto_tab = QWidget()
        self.auto_tab.setObjectName('autoTab')
        self.auto_tab_scroll, self.auto_tab_content = self._make_scroll_tab()
        self.predict_tab, self.predict_tab_content = self._make_scroll_tab()
        # predict fits inside the rounded Qt tab at its minimum width
        self.predict_tab.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.training_tab, self.training_tab_content = self._make_scroll_tab()
        self.tabs.addTab(self.auto_tab, 'AUTO')
        self.tabs.addTab(self.training_tab, 'train')
        self.tabs.addTab(self.predict_tab, 'segment')
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.tabs.currentChanged.connect(self.controls_tab_changed)
        self._build_auto_widgets()
        self._build_training_widgets()
        self._build_prediction_widgets()
        self._build_segment_widgets()
        self._build_editing_widgets()
        self._build_persistent_widgets()
        self.output_box = self.make_log_box()

        self.roi_overlay_check = QCheckBox('ROI on')
        self.roi_overlay_check.setAccessibleName('ROI overlay')
        self.roi_overlay_check.setAccessibleDescription(
            'show or hide the ROI outlines on the image')
        self.roi_overlay_check.setChecked(True)
        self.roi_overlay_check.stateChanged.connect(self.set_roi_overlay_visible)
        self.dark_mode_check = QCheckBox('dark mode')
        self.dark_mode_check.setAccessibleName('dark mode')
        self.dark_mode_check.setAccessibleDescription(
            'switch between dark and light interface colours')
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
            action.setChecked(size == self.interface_font_size)
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

    def _build_auto_widgets(self):
        self.auto_tiff_dir_line = QLineEdit()
        self.auto_control_dir_line = QLineEdit()
        self.auto_output_dir_line = QLineEdit(str(OUTPUT_ROOT))
        self.auto_session_line = QLineEdit('session')
        self.auto_signal_label_line = QLineEdit('signal')
        self.auto_control_label_line = QLineEdit('control')

        self.auto_acquisition_combo = QComboBox()
        self.auto_acquisition_combo.addItem('interleaved pages', True)
        self.auto_acquisition_combo.addItem('separate TIFF folders', False)
        self.auto_signal_channel_combo = QComboBox()
        self.auto_control_channel_combo = QComboBox()
        for combo in (
                self.auto_signal_channel_combo,
                self.auto_control_channel_combo,
                ):
            combo.addItems(['1', '2'])
        self.auto_control_channel_combo.setCurrentText('2')

        self.auto_sampling_spin = QDoubleSpinBox()
        self.auto_sampling_spin.setRange(0.01, 10000)
        self.auto_sampling_spin.setDecimals(1)
        self.auto_sampling_spin.setValue(30)
        self.auto_sampling_spin.setSuffix(' Hz')
        self.prepare_value_control(self.auto_sampling_spin)

        self.auto_registration_model_combo = QComboBox()
        self.auto_registration_model_combo.addItems(['rigid', 'piecewise', 'auto'])
        self.auto_registration_model_combo.setCurrentText('auto')
        self.auto_registration_model_combo.setToolTip(
            'piecewise stores local spline fields and uses exact valid pixels '
            'during segmentation and extraction; auto selects it when the '
            'registration benchmarks pass')
        self.auto_registration_channel_combo = QComboBox()
        self.auto_registration_channel_combo.addItems(['control', 'signal'])

        self.auto_reference_channel_combo = QComboBox()
        self.auto_reference_channel_combo.addItems(['control', 'signal'])
        self.auto_checkpoint_line = QLineEdit(str(BUNDLED_CHECKPOINT))
        self.auto_threshold_spin = QDoubleSpinBox()
        self.auto_threshold_spin.setRange(0.01, 0.99)
        self.auto_threshold_spin.setSingleStep(0.01)
        self.auto_threshold_spin.setDecimals(2)
        self.auto_threshold_spin.setValue(BUNDLED_THRESHOLD)
        self.prepare_value_control(self.auto_threshold_spin)
        self.auto_min_size_spin = QSpinBox()
        self.auto_min_size_spin.setRange(1, 100000)
        self.auto_min_size_spin.setValue(BUNDLED_MIN_SIZE)
        self.prepare_value_control(self.auto_min_size_spin)
        self.auto_reference_high_spin = QDoubleSpinBox()
        self.auto_reference_high_spin.setRange(1, 100)
        self.auto_reference_high_spin.setDecimals(1)
        self.auto_reference_high_spin.setValue(97)
        self.auto_reference_high_spin.setSuffix('%')
        self.prepare_value_control(self.auto_reference_high_spin)

        self.auto_surround_method_combo = QComboBox()
        self.auto_surround_method_combo.addItems(['adaptive', 'fixed'])
        self.auto_surround_inner_spin = QSpinBox()
        self.auto_surround_inner_spin.setRange(0, 1000)
        self.auto_surround_inner_spin.setValue(5)
        self.auto_surround_outer_spin = QSpinBox()
        self.auto_surround_outer_spin.setRange(1, 1000)
        self.auto_surround_outer_spin.setValue(8)
        self.auto_surround_min_spin = QSpinBox()
        self.auto_surround_min_spin.setRange(0, 100000)
        self.auto_surround_min_spin.setValue(350)
        for spin in (
                self.auto_surround_inner_spin,
                self.auto_surround_outer_spin,
                self.auto_surround_min_spin,
                ):
            self.prepare_value_control(spin)

        self.auto_statistic_combo = QComboBox()
        self.auto_statistic_combo.addItems(['mean', 'median'])
        self.auto_baseline_percentile_spin = QDoubleSpinBox()
        self.auto_baseline_percentile_spin.setRange(0, 100)
        self.auto_baseline_percentile_spin.setDecimals(1)
        self.auto_baseline_percentile_spin.setValue(20)
        self.auto_baseline_percentile_spin.setSuffix('%')
        self.auto_baseline_window_spin = QDoubleSpinBox()
        self.auto_baseline_window_spin.setRange(0.01, 100000)
        self.auto_baseline_window_spin.setDecimals(1)
        self.auto_baseline_window_spin.setValue(300)
        self.auto_baseline_window_spin.setSuffix(' s')
        self.auto_surround_coefficient_spin = QDoubleSpinBox()
        self.auto_surround_coefficient_spin.setRange(0, 10)
        self.auto_surround_coefficient_spin.setDecimals(2)
        self.auto_surround_coefficient_spin.setValue(0.7)
        for spin in (
                self.auto_baseline_percentile_spin,
                self.auto_baseline_window_spin,
                self.auto_surround_coefficient_spin,
                ):
            self.prepare_value_control(spin)
        self.auto_control_correction_combo = QComboBox()
        self.auto_control_correction_combo.addItem('none', 'none')
        self.auto_control_correction_combo.addItem(
            'signal dF/F minus control dF/F',
            'subtract_dff',
            )

        self.auto_advanced_button = QPushButton('advanced')
        self.auto_advanced_button.setCheckable(True)
        self.set_button_role(self.auto_advanced_button, 'quiet')
        self.auto_advanced_widget = QWidget()
        self.auto_advanced_widget.hide()
        self.auto_reference_low_spin = QDoubleSpinBox()
        self.auto_reference_low_spin.setRange(0, 99)
        self.auto_reference_low_spin.setDecimals(1)
        self.auto_reference_low_spin.setValue(1)
        self.auto_reference_low_spin.setSuffix('%')
        self.auto_tta_check = QCheckBox('four-view TTA')
        self.auto_tta_check.setChecked(True)
        self.auto_device_combo = QComboBox()
        self.auto_device_combo.addItems(['auto', 'cpu', 'mps', 'cuda'])
        self.auto_pixel_size_spin = QDoubleSpinBox()
        self.auto_pixel_size_spin.setRange(0, 10000)
        self.auto_pixel_size_spin.setDecimals(4)
        self.auto_pixel_size_spin.setSpecialValueText('not supplied')
        self.auto_pixel_size_spin.setSuffix(' um')
        self.auto_proposal_run_line = QLineEdit('proposal_auto')
        self.auto_roi_source_combo = QComboBox()
        self.auto_roi_source_combo.setEditable(True)
        self.auto_roi_source_combo.addItem('proposal_auto')
        self.auto_fluorescence_run_line = QLineEdit('fluorescence_auto')
        self.auto_dff_run_line = QLineEdit('dff_auto')

        for widget, name, description in (
                (self.auto_tiff_dir_line, 'TIFF folder',
                 'folder containing the recording TIFFs'),
                (self.auto_control_dir_line, 'control TIFF folder',
                 'second folder when signal and control are stored separately'),
                (self.auto_output_dir_line, 'destination folder',
                 'folder for the NWB and session log'),
                (self.auto_session_line, 'session name',
                 'name used for the NWB output and session log'),
                (self.auto_signal_label_line, 'signal label',
                 'label stored for the signal channel'),
                (self.auto_control_label_line, 'control label',
                 'label stored for the control channel'),
                (self.auto_acquisition_combo, 'recording layout',
                 'whether the two channels share each TIFF or use separate folders'),
                (self.auto_signal_channel_combo, 'signal channel',
                 'channel number for the signal in interleaved TIFFs'),
                (self.auto_control_channel_combo, 'control channel',
                 'channel number for the control in interleaved TIFFs'),
                (self.auto_sampling_spin, 'sampling frequency',
                 'paired observations per second'),
                (self.auto_registration_model_combo, 'registration model',
                 'rigid, piecewise, or automatic registration selection'),
                (self.auto_registration_channel_combo, 'registration channel',
                 'channel used to estimate movement'),
                (self.auto_reference_channel_combo, 'reference channel',
                 'channel used to build the segmentation reference'),
                (self.auto_checkpoint_line, 'segmentation model',
                 'checkpoint used for automatic ROI proposals'),
                (self.auto_threshold_spin, 'prediction threshold',
                 'minimum model confidence for an ROI'),
                (self.auto_min_size_spin, 'minimum ROI area',
                 'smallest connected component kept as an ROI'),
                (self.auto_reference_high_spin, 'reference upper percentile',
                 'upper contrast limit for the full-session reference'),
                (self.auto_surround_method_combo, 'surround method',
                 'method used to choose pixels around each ROI'),
                (self.auto_surround_inner_spin, 'surround inner distance',
                 'inner distance around each ROI'),
                (self.auto_surround_outer_spin, 'surround outer distance',
                 'outer distance for a fixed surround'),
                (self.auto_surround_min_spin, 'minimum surround pixels',
                 'minimum pixels in an adaptive surround'),
                (self.auto_statistic_combo, 'fluorescence statistic',
                 'pixel statistic used for fluorescence traces'),
                (self.auto_baseline_percentile_spin, 'baseline percentile',
                 'percentile used for the rolling baseline'),
                (self.auto_baseline_window_spin, 'baseline window',
                 'rolling baseline window in seconds'),
                (self.auto_surround_coefficient_spin, 'surround coefficient',
                 'weight subtracted from the ROI surround trace'),
                (self.auto_control_correction_combo, 'control correction',
                 'whether control dF/F is subtracted from signal dF/F'),
                (self.auto_reference_low_spin, 'reference lower percentile',
                 'lower contrast limit for the full-session reference'),
                (self.auto_tta_check, 'four-view TTA',
                 'average predictions from four image views'),
                (self.auto_device_combo, 'inference device',
                 'device used for automatic ROI inference'),
                (self.auto_pixel_size_spin, 'pixel size',
                 'optional pixel size in micrometres'),
                (self.auto_proposal_run_line, 'proposal run',
                 'name for the immutable automatic ROI proposal'),
                (self.auto_roi_source_combo, 'extraction ROI run',
                 'ROI run measured during extraction'),
                (self.auto_fluorescence_run_line, 'fluorescence run',
                 'name for the immutable fluorescence run'),
                (self.auto_dff_run_line, 'dF/F run',
                 'name for the immutable dF/F run'),
                ):
            widget.setAccessibleName(name)
            widget.setAccessibleDescription(description)

        self.auto_state_label = ElidedLabel('automatic session not started')
        self.auto_state_label.setObjectName('panelValue')
        self.auto_progress = QProgressBar()
        self.auto_progress.setAccessibleName('pipeline progress')
        self.auto_progress.setAccessibleDescription(
            'completed automatic recording stages out of four')
        self.auto_progress.setRange(0, 4)
        self.auto_progress.setValue(0)
        self.auto_progress.setTextVisible(True)
        self.auto_progress.setFormat('%v / 4 stages')
        self.auto_run_button = QPushButton('RUN PIPELINE')
        self.auto_resume_button = QPushButton('resume session')
        self.auto_stop_button = QPushButton('stop')
        self.auto_stop_button.hide()
        self.set_button_role(self.auto_run_button, 'primary')
        self.set_button_role(self.auto_resume_button, 'secondary')
        self.set_button_role(self.auto_stop_button, 'danger')

        self.auto_acquisition_combo.currentIndexChanged.connect(
            self.auto_acquisition_changed)
        self.auto_surround_method_combo.currentTextChanged.connect(
            self.update_auto_surround_controls)
        self.auto_advanced_button.toggled.connect(
            self.auto_advanced_widget.setVisible)
        self.auto_run_button.clicked.connect(self.start_auto_session)
        self.auto_resume_button.clicked.connect(self.choose_auto_session_log)
        self.auto_stop_button.clicked.connect(self.stop_process)

    def _build_training_widgets(self):
        self.source_root_line = QLineEdit(str(SOURCE_ROOT))
        self.manifest_line = QLineEdit(str(WORKSPACE_ROOT / 'manifests' / 'ch2_manifest.csv'))
        self.config_line = QLineEdit(
            str(PACKAGE_ROOT / 'configs' / 'ch2_unet.yaml')
            )
        self.train_output_dir_line = QLineEdit(str(OUTPUT_ROOT / 'runs'))
        self.run_name_line = QLineEdit('ch2_unet')
        self.source_root_line.textChanged.connect(lambda _: self.refresh_status())
        self.manifest_line.textChanged.connect(lambda _: self.refresh_status())
        self.config_line.textChanged.connect(lambda _: self.refresh_status())
        self.train_output_dir_line.textChanged.connect(
            lambda _: self.refresh_status())
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
        self.set_button_role(self.train_model_button, 'primary')
        self.set_button_role(self.stop_process_button, 'danger')
        self.set_button_role(self.inspect_manifest_button, 'quiet')
        self.set_button_role(self.evaluate_model_button, 'quiet')
        self.evaluate_model_button.setToolTip('score the current trained model on held-out labelled sessions')
        self.stop_process_button.hide()

        self.build_manifest_button.clicked.connect(self.build_manifest)
        self.train_model_button.clicked.connect(self.train_model)
        self.evaluate_model_button.clicked.connect(self.evaluate_model)
        self.inspect_manifest_button.clicked.connect(self.inspect_manifest)
        self.stop_process_button.clicked.connect(self.stop_process)

    def _build_prediction_widgets(self):
        self.image_line = QLineEdit()
        self.checkpoint_line = QLineEdit(str(BUNDLED_CHECKPOINT))
        self.proposal_run_combo = QComboBox()
        self.curated_run_line = QLineEdit()
        self.proposal_run_combo.setAccessibleName('proposal run')
        self.curated_run_line.setAccessibleName('curated run name')
        self.curated_run_line.setPlaceholderText('new immutable run name')
        self.image_line.textChanged.connect(self.prediction_changed)
        self.image_line.editingFinished.connect(self.load_edited_channel_image)
        self.checkpoint_line.textChanged.connect(self.prediction_changed)
        self.proposal_run_combo.currentIndexChanged.connect(
            self.load_selected_proposal_run
            )
        self.curated_run_line.textChanged.connect(lambda _: self.refresh_status())

        # these stayed visible because they were the useful controls during tuning
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.01, 0.99)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(BUNDLED_THRESHOLD)
        self.prepare_value_control(self.threshold_spin)

        self.min_size_spin = QSpinBox()
        self.min_size_spin.setRange(1, 100000)
        self.min_size_spin.setValue(BUNDLED_MIN_SIZE)
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
        self.image_view_button.setAccessibleName('image display')
        self.confidence_view_button.setAccessibleName('confidence display')
        self.image_view_button.setAccessibleDescription(
            'show the loaded reference image')
        self.confidence_view_button.setAccessibleDescription(
            'show the model confidence map')
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

        self.black_slider.setAccessibleName('black point')
        self.black_slider.setAccessibleDescription(
            'lower display percentile for the image')
        self.white_slider.setAccessibleName('white point')
        self.white_slider.setAccessibleDescription(
            'upper display percentile for the image')
        self.black_value.setAccessibleName('black point value')
        self.white_value.setAccessibleName('white point value')

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

        self.activity_frame = QFrame()
        self.activity_frame.setObjectName('activityFrame')
        activity_layout = QVBoxLayout(self.activity_frame)
        activity_layout.setContentsMargins(8, 8, 8, 8)
        self.activity_tabs = QTabWidget()
        self.activity_tabs.setObjectName('activityTabs')
        self.activity_tabs.setAccessibleName('activity view')
        self.activity_tabs.currentChanged.connect(self.set_activity_view)
        activity_layout.addWidget(self.activity_tabs, 1)

        self.trace_figure = Figure(dpi=100, facecolor=self._theme()['canvas'])
        self.trace_canvas = FigureCanvasQTAgg(self.trace_figure)
        self.trace_canvas.setMinimumSize(300, 300)
        self.trace_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.trace_canvas.setAccessibleName('fluorescence trace inspector')
        self.trace_canvas.mpl_connect('scroll_event', self.trace_scroll)
        self.trace_run_combo = QComboBox()
        self.trace_run_combo.setAccessibleName('dF/F run')
        self.trace_roi_combo = QComboBox()
        self.trace_roi_combo.setAccessibleName('ROI trace')
        self.trace_signal_check = QCheckBox('signal')
        self.trace_signal_check.setAccessibleName('show signal trace')
        self.trace_signal_check.setChecked(True)
        self.trace_control_check = QCheckBox('control')
        self.trace_control_check.setAccessibleName('show control trace')
        self.trace_control_check.setChecked(True)
        self.trace_zoom_out_button = QPushButton('zoom out')
        self.trace_fit_button = QPushButton('fit')
        self.trace_zoom_in_button = QPushButton('zoom in')
        for button in (
                self.trace_zoom_out_button,
                self.trace_fit_button,
                self.trace_zoom_in_button,
                ):
            self.set_button_role(button, 'small')
        self.trace_run_combo.currentIndexChanged.connect(self.load_trace_run)
        self.trace_roi_combo.currentIndexChanged.connect(self.draw_trace_inspector)
        self.trace_signal_check.toggled.connect(self.draw_trace_inspector)
        self.trace_control_check.toggled.connect(self.draw_trace_inspector)
        self.trace_zoom_out_button.clicked.connect(
            lambda: self.zoom_trace(0.5))
        self.trace_fit_button.clicked.connect(self.fit_trace)
        self.trace_zoom_in_button.clicked.connect(
            lambda: self.zoom_trace(2.0))

        trace_header = QFrame()
        trace_header.setObjectName('curationBar')
        trace_header_layout = QHBoxLayout(trace_header)
        trace_header_layout.setContentsMargins(7, 5, 7, 5)
        trace_header_layout.setSpacing(6)
        trace_header_layout.addWidget(QLabel('dF/F run'))
        trace_header_layout.addWidget(self.trace_run_combo, 1)
        trace_header_layout.addWidget(QLabel('ROI'))
        trace_header_layout.addWidget(self.trace_roi_combo)
        trace_header_layout.addWidget(self.trace_signal_check)
        trace_header_layout.addWidget(self.trace_control_check)
        trace_header_layout.addWidget(self.trace_zoom_out_button)
        trace_header_layout.addWidget(self.trace_fit_button)
        trace_header_layout.addWidget(self.trace_zoom_in_button)

        trace_frame = QFrame()
        trace_frame.setObjectName('canvasFrame')
        trace_frame_layout = QVBoxLayout(trace_frame)
        trace_frame_layout.setContentsMargins(3, 3, 3, 3)
        trace_frame_layout.addWidget(self.trace_canvas)
        trace_stage = QWidget()
        trace_stage_layout = QVBoxLayout(trace_stage)
        trace_stage_layout.setContentsMargins(0, 0, 0, 0)
        trace_stage_layout.setSpacing(8)
        trace_stage_layout.addWidget(trace_header)
        trace_stage_layout.addWidget(trace_frame, 1)

        self.activity_tabs.addTab(trace_stage, 'trace inspector')
        self.activity_tabs.addTab(self.output_box, 'console')

        self.right_stack = QStackedWidget()
        self.right_stack.addWidget(canvas_stage)

        self.activity_splitter = QSplitter(Qt.Vertical)
        self.activity_splitter.setObjectName('activitySplitter')
        self.activity_splitter.addWidget(self.right_stack)
        self.activity_splitter.addWidget(self.activity_frame)
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

        self._layout_auto_tab()
        self._layout_prediction_tab()
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
        self.proposal_run_label = self.make_form_label(
            'proposal run',
            'immutable NWB proposal to inspect and curate',
            self.proposal_run_combo,
            )
        self.curated_run_label = self.make_form_label(
            'curated run',
            'new immutable run name for edited ROIs',
            self.curated_run_line,
            )
        resources_form.addRow(
            self.proposal_run_label,
            self.proposal_run_combo,
            )
        resources_form.addRow(
            self.curated_run_label,
            self.curated_run_line,
            )
        self.set_nwb_controls_visible(False)
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

        self.stage_header = QFrame()
        self.stage_header.setObjectName('stageHeader')
        stage_layout = QHBoxLayout(self.stage_header)
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
        main_layout.setContentsMargins(10, 8, 10, 8)
        main_layout.setSpacing(6)
        main_layout.addWidget(self.stage_header)
        main_layout.addWidget(self.main_splitter, 1)

        container = QWidget()
        container.setObjectName('centralWidget')
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        self.controls_tab_changed(self.tabs.currentIndex())
        self.controls_split_timer.start(0)

    def controls_tab_changed(self, _index):
        auto_selected = self.tabs.currentWidget() is self.auto_tab
        segment_selected = self.tabs.currentWidget() is self.predict_tab
        auto_trace_ready = (
            auto_selected
            and self.auto_loaded_config is not None
            and self.trace_cache is not None
            and self.nwb_path == Path(self.auto_loaded_config['output_path'])
            )
        self.resources_panel.setVisible(segment_selected)
        self.persistent_panel.setVisible(segment_selected or auto_trace_ready)
        self.curation_bar.setVisible(segment_selected)
        self.stage_header.setVisible(segment_selected)
        self.right_stack.setVisible(segment_selected or auto_trace_ready)
        if segment_selected:
            self.right_stack.setCurrentIndex(0)
            self.activity_tabs.setCurrentIndex(1)
        elif auto_trace_ready:
            self.right_stack.setCurrentIndex(0)
            self.activity_tabs.setCurrentIndex(0)
        else:
            self.activity_tabs.setCurrentIndex(1)
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
                self.auto_tab_scroll
                if self.tabs.currentWidget() is self.auto_tab else
                self.tabs.currentWidget()
                )
            tab_chrome = max(
                0,
                self.tabs.height() - current_scroll.viewport().height(),
                )
            stored_height = (
                (
                    self.resources_panel.sizeHint().height() +
                    self.upper_controls_layout.spacing()
                    if self.resources_panel.isVisible() else 0
                    ) +
                tab_chrome +
                max(
                    current_scroll.widget().sizeHint().height(),
                    current_scroll.widget().minimumSizeHint().height(),
                    )
                )

        if (
                self.tabs.currentWidget() is self.auto_tab
                and self.trace_cache is not None
                ):
            stored_height = int(total * 2 / 3)

        min_upper = self.upper_controls.minimumSizeHint().height()
        persistent_height = (
            self.persistent_panel.minimumSizeHint().height()
            if self.persistent_panel.isVisible() else 0
            )
        max_upper = total - persistent_height
        target = int(np.clip(stored_height, min_upper, max_upper))
        self._sizing_controls_splitter = True
        self.controls_splitter.setSizes([target, total - target])
        self._sizing_controls_splitter = False

    def _layout_auto_tab(self):
        tab_layout = QVBoxLayout(self.auto_tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(0)

        layout = QVBoxLayout(self.auto_tab_content)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)

        layout.addWidget(self.make_section_label('recording'))
        source_form = QFormLayout()
        source_form.setVerticalSpacing(5)
        source_form.addRow(
            self.make_form_label('TIFF folder', 'folder containing the recording TIFFs'),
            self._path_row(self.auto_tiff_dir_line, self.browse_auto_tiff_dir),
            )
        self.auto_control_dir_row = self._path_row(
            self.auto_control_dir_line,
            self.browse_auto_control_dir,
            )
        self.auto_control_dir_label = self.make_form_label(
            'control TIFF folder',
            'second folder when signal and control are stored separately',
            )
        source_form.addRow(
            self.auto_control_dir_label,
            self.auto_control_dir_row,
            )
        source_form.addRow('layout', self.auto_acquisition_combo)
        source_form.addRow('sampling frequency', self.auto_sampling_spin)
        source_form.addRow('signal channel', self.auto_signal_channel_combo)
        source_form.addRow('control channel', self.auto_control_channel_combo)
        source_form.addRow('signal label', self.auto_signal_label_line)
        source_form.addRow('control label', self.auto_control_label_line)
        source_form.addRow(
            self.make_form_label('destination', 'folder for the NWB and session log'),
            self._path_row(self.auto_output_dir_line, self.browse_auto_output_dir),
            )
        source_form.addRow('session name', self.auto_session_line)
        layout.addLayout(source_form)

        layout.addWidget(self.make_section_label('registration'))
        registration_form = QFormLayout()
        registration_form.setVerticalSpacing(5)
        registration_form.addRow(
            self.make_form_label(
                'model',
                self.auto_registration_model_combo.toolTip(),
                self.auto_registration_model_combo,
                ),
            self.auto_registration_model_combo,
            )
        registration_form.addRow('registration channel', self.auto_registration_channel_combo)
        layout.addLayout(registration_form)

        layout.addWidget(self.make_section_label('segmentation'))
        segmentation_form = QFormLayout()
        segmentation_form.setVerticalSpacing(5)
        segmentation_form.addRow('reference channel', self.auto_reference_channel_combo)
        segmentation_form.addRow(
            self.make_form_label(
                'reference upper percentile',
                'upper contrast limit for the full-session reference',
                ),
            self.auto_reference_high_spin,
            )
        segmentation_form.addRow(
            self.make_form_label('model', 'checkpoint used for ROI proposals'),
            self._path_row(self.auto_checkpoint_line, self.browse_auto_checkpoint),
            )
        segmentation_form.addRow('prediction threshold', self.auto_threshold_spin)
        segmentation_form.addRow('minimum ROI area', self.auto_min_size_spin)
        layout.addLayout(segmentation_form)

        layout.addWidget(self.make_section_label('extraction'))
        extraction_form = QFormLayout()
        extraction_form.setVerticalSpacing(5)
        extraction_form.addRow('surround method', self.auto_surround_method_combo)
        extraction_form.addRow('inner distance', self.auto_surround_inner_spin)
        self.auto_surround_outer_label = QLabel('outer distance')
        extraction_form.addRow(
            self.auto_surround_outer_label,
            self.auto_surround_outer_spin,
            )
        self.auto_surround_min_label = QLabel('minimum surround pixels')
        extraction_form.addRow(
            self.auto_surround_min_label,
            self.auto_surround_min_spin,
            )
        layout.addLayout(extraction_form)

        layout.addWidget(self.make_section_label('dF/F'))
        dff_form = QFormLayout()
        dff_form.setVerticalSpacing(5)
        dff_form.addRow('statistic', self.auto_statistic_combo)
        dff_form.addRow('baseline percentile', self.auto_baseline_percentile_spin)
        dff_form.addRow('baseline window', self.auto_baseline_window_spin)
        dff_form.addRow('surround coefficient', self.auto_surround_coefficient_spin)
        dff_form.addRow('control correction', self.auto_control_correction_combo)
        layout.addLayout(dff_form)

        layout.addWidget(self.auto_advanced_button)
        advanced_form = QFormLayout(self.auto_advanced_widget)
        advanced_form.setContentsMargins(0, 0, 0, 0)
        advanced_form.setVerticalSpacing(5)
        advanced_form.addRow('reference lower percentile', self.auto_reference_low_spin)
        advanced_form.addRow('inference', self.auto_tta_check)
        advanced_form.addRow('device', self.auto_device_combo)
        advanced_form.addRow('pixel size', self.auto_pixel_size_spin)
        advanced_form.addRow('proposal run', self.auto_proposal_run_line)
        advanced_form.addRow('extraction ROI run', self.auto_roi_source_combo)
        advanced_form.addRow('fluorescence run', self.auto_fluorescence_run_line)
        advanced_form.addRow('dF/F run', self.auto_dff_run_line)
        layout.addWidget(self.auto_advanced_widget)
        layout.addStretch(1)

        self.auto_footer = QFrame()
        self.auto_footer.setObjectName('autoFooter')
        footer_layout = QVBoxLayout(self.auto_footer)
        footer_layout.setContentsMargins(10, 7, 10, 9)
        footer_layout.setSpacing(5)
        footer_layout.addWidget(self.auto_state_label)
        footer_layout.addWidget(self.auto_progress)
        actions = QHBoxLayout()
        actions.setSpacing(6)
        actions.addWidget(self.auto_run_button, 1)
        actions.addWidget(self.auto_resume_button)
        actions.addWidget(self.auto_stop_button)
        footer_layout.addLayout(actions)
        tab_layout.addWidget(self.auto_tab_scroll, 1)
        tab_layout.addWidget(self.auto_footer)
        self.auto_acquisition_changed()
        self.update_auto_surround_controls()

    def _layout_prediction_tab(self):
        layout = QVBoxLayout(self.predict_tab_content)
        layout.setSpacing(8)
        layout.setContentsMargins(10, 10, 10, 10)
        self._layout_prediction_section(layout)
        self.mser_advanced_button = QPushButton('MSER proposal controls')
        self.mser_advanced_button.setCheckable(True)
        self.set_button_role(self.mser_advanced_button, 'quiet')
        self.mser_panel = QWidget()
        mser_layout = QVBoxLayout(self.mser_panel)
        mser_layout.setContentsMargins(0, 0, 0, 0)
        mser_layout.setSpacing(6)
        self._layout_mser_section(mser_layout)
        mser_actions = QHBoxLayout()
        mser_actions.addWidget(self.segment_button, 1)
        mser_actions.addWidget(self.reset_segment_button)
        mser_layout.addLayout(mser_actions)
        self.mser_panel.hide()
        self.mser_advanced_button.toggled.connect(self.mser_panel.setVisible)
        layout.addWidget(self.mser_advanced_button)
        layout.addWidget(self.mser_panel)
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
        data_form = QFormLayout()
        data_form.setVerticalSpacing(5)
        data_form.addRow(
            self.make_form_label(
                'labelled sessions',
                'folder containing processed sessions for training',
                self.source_root_line,
                ),
            self._path_row(self.source_root_line, self.browse_source_root),
            )
        data_form.addRow(
            self.make_form_label(
                'dataset table',
                'CSV index of labelled images and ROI dicts',
                self.manifest_line,
                ),
            self._path_row(self.manifest_line, self.browse_manifest_out),
            )
        data_form.addRow(
            self.make_form_label(
                'validation split',
                'fraction used for tuning during training',
                self.val_fraction_spin,
            ),
            self.val_fraction_spin,
            )
        data_form.addRow(
            self.make_form_label(
                'test split',
                'held-out fraction used for scoring',
                self.test_fraction_spin,
            ),
            self.test_fraction_spin,
            )
        layout.addLayout(data_form)
        data_actions = QGridLayout()
        data_actions.setHorizontalSpacing(6)
        data_actions.setVerticalSpacing(6)
        data_actions.addWidget(self.build_manifest_button, 0, 0)
        data_actions.addWidget(self.inspect_manifest_button, 0, 1)
        layout.addLayout(data_actions)

        layout.addWidget(self.make_section_label('training'))
        form = QFormLayout()
        form.setVerticalSpacing(5)
        form.addRow(
            self.make_form_label(
                'training recipe',
                'YAML settings used for model training',
                self.config_line,
                ),
            self._path_row(self.config_line, self.browse_config),
            )
        form.addRow(
            self.make_form_label(
                'output folder',
                'best.pt and latest.pt are saved under output folder/run name',
                self.train_output_dir_line,
                ),
            self._path_row(
                self.train_output_dir_line,
                self.browse_training_output_dir,
                ),
            )
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
        box.setPlaceholderText('pipeline messages appear here')
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
        display_text = 'dF/F' if text == 'dF/F' else text.upper()
        label = QLabel(display_text)
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
            QFrame#autoFooter {{
                border: none;
                border-top: 1px solid {theme['border']};
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
            QScrollArea, QWidget#tabContent, QWidget#autoTab {{
                border: none;
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
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
                border: 1px solid {theme['border']};
                border-radius: 2px;
                padding: 3px 6px;
                background: {theme['surface_alt']};
                color: {theme['text']};
                selection-background-color: {theme['selection']};
                min-height: 23px;
            }}
            QLineEdit:hover, QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {{
                border-color: {theme['border_strong']};
                background: {theme['surface']};
            }}
            QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus {{
                border-color: {theme['text']};
                background: {theme['surface']};
            }}
            QLineEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled,
            QSpinBox:disabled {{
                border-color: {theme['border']};
                background: {theme['surface_alt']};
                color: {theme['disabled']};
            }}
            QProgressBar {{
                border: 1px solid {theme['border']};
                border-radius: 2px;
                background: {theme['surface_alt']};
                min-height: 18px;
                max-height: 18px;
                text-align: center;
                color: {theme['text']};
            }}
            QProgressBar::chunk {{
                background: {theme['border_strong']};
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
        self.draw_trace_inspector()

    def set_interface_font_size(self, point_size):
        text_widget_types = (
            QAbstractButton,
            QAbstractSpinBox,
            QComboBox,
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
        self.settings.setValue('interface/font_size', point_size)
        self.settings.sync()
        self.plot_image(preserve_view=True)
        self.draw_trace_inspector()
        self._controls_split_positions.clear()
        self.controls_split_timer.start(0)
        self.schedule_curation_layout()

    def prediction_changed(self, _text=None):
        self.clear_probability()
        self.refresh_status()

    def clear_probability(self, redraw=True):
        needs_redraw = self.probability is not None or self.display_mode == 'confidence'
        self.probability = None
        self.probability_max_size = None
        self.display_mode = 'image'
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
            # qt keeps a private shift-selection anchor beyond the current index
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
        self.sync_trace_roi_selection()
        self.plot_image(preserve_view=True)
        self.refresh_status()

    def sync_trace_roi_selection(self):
        if (
                self.tabs.currentWidget() is self.auto_tab
                and self.trace_cache is not None
                and self.selected
                ):
            trace_index = self.trace_roi_combo.findData(min(self.selected))
            if trace_index >= 0:
                self.trace_roi_combo.setCurrentIndex(trace_index)
                self.activity_tabs.setCurrentIndex(0)

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


    #%% automatic recording workflow
    def browse_auto_tiff_dir(self):
        self._browse_auto_directory(self.auto_tiff_dir_line, 'select TIFF folder')

    def browse_auto_control_dir(self):
        self._browse_auto_directory(
            self.auto_control_dir_line,
            'select control TIFF folder',
            )

    def browse_auto_output_dir(self):
        self._browse_auto_directory(
            self.auto_output_dir_line,
            'select destination folder',
            )

    def _browse_auto_directory(self, line_edit, title):
        selected = QFileDialog.getExistingDirectory(
            self,
            title,
            line_edit.text().strip() or str(Path.home()),
            )
        if selected:
            line_edit.setText(selected)

    def browse_auto_checkpoint(self):
        selected, _ = QFileDialog.getOpenFileName(
            self,
            'select segmentation model',
            self.auto_checkpoint_line.text().strip() or str(PACKAGE_ROOT),
            'PyTorch checkpoints (*.pt *.pth);;All files (*)',
            )
        if selected:
            self.auto_checkpoint_line.setText(selected)

    def auto_acquisition_changed(self, _index=None):
        separate = not bool(self.auto_acquisition_combo.currentData())
        self.auto_control_dir_label.setVisible(separate)
        self.auto_control_dir_row.setVisible(separate)
        self.auto_signal_channel_combo.setEnabled(not separate)
        self.auto_control_channel_combo.setEnabled(not separate)

    def update_auto_surround_controls(self, _method=None):
        fixed = self.auto_surround_method_combo.currentText() == 'fixed'
        self.auto_surround_outer_label.setVisible(fixed)
        self.auto_surround_outer_spin.setVisible(fixed)
        self.auto_surround_min_label.setVisible(not fixed)
        self.auto_surround_min_spin.setVisible(not fixed)

    def auto_session_config(self):
        signal_text = self.auto_tiff_dir_line.text().strip()
        signal_dir = Path(signal_text)
        if not signal_text or not signal_dir.is_dir():
            raise ValueError('select a TIFF folder')
        signal_paths = natural_tiff_paths(signal_dir)
        if not signal_paths:
            raise ValueError(f'no TIFF files found in {signal_dir}')

        multiplexed = bool(self.auto_acquisition_combo.currentData())
        signal_channel = int(self.auto_signal_channel_combo.currentText())
        control_channel = int(self.auto_control_channel_combo.currentText())
        if multiplexed and signal_channel == control_channel:
            raise ValueError('signal and control channels must be different')
        control_paths = []
        if not multiplexed:
            control_text = self.auto_control_dir_line.text().strip()
            control_dir = Path(control_text)
            if not control_text or not control_dir.is_dir():
                raise ValueError('select the control TIFF folder')
            control_paths = natural_tiff_paths(control_dir)
            if len(control_paths) != len(signal_paths):
                raise ValueError(
                    'signal and control TIFF folders contain different file counts')

        destination_text = self.auto_output_dir_line.text().strip()
        destination = Path(destination_text)
        if not destination_text or not destination.is_dir():
            raise ValueError('select an existing destination folder')
        session_name = self.auto_session_line.text().strip()
        if not session_name or Path(session_name).name != session_name:
            raise ValueError('session name must be one file name')
        checkpoint_text = self.auto_checkpoint_line.text().strip()
        checkpoint = Path(checkpoint_text)
        if not checkpoint_text or not checkpoint.is_file():
            raise ValueError('select a segmentation model')
        low = float(self.auto_reference_low_spin.value())
        high = float(self.auto_reference_high_spin.value())
        if low >= high:
            raise ValueError('reference percentiles must satisfy lower < upper')

        signal_records = fingerprint_paths(signal_paths)
        control_records = fingerprint_paths(control_paths)
        checkpoint_record = fingerprint_paths([checkpoint])[0]
        proposal_run = self.auto_proposal_run_line.text().strip()
        roi_run = self.auto_roi_source_combo.currentText().strip()
        fluorescence_run = self.auto_fluorescence_run_line.text().strip()
        dff_run = self.auto_dff_run_line.text().strip()
        if not all((proposal_run, roi_run, fluorescence_run, dff_run)):
            raise ValueError('automatic run names cannot be empty')

        return {
            'schema_version': SESSION_SCHEMA_VERSION,
            'fibre_sight_version': __version__,
            'output_path': str((destination / f'{session_name}.nwb').resolve()),
            'source_files': signal_records + control_records,
            'signal_files': signal_records,
            'control_files': control_records,
            'source': {
                'multiplexed': multiplexed,
                'sampling_frequency_hz': float(self.auto_sampling_spin.value()),
                'signal_channel': signal_channel,
                'control_channel': control_channel,
                'signal_label': self.auto_signal_label_line.text().strip(),
                'control_label': self.auto_control_label_line.text().strip(),
                'pixel_size_um': (
                    float(self.auto_pixel_size_spin.value())
                    if self.auto_pixel_size_spin.value() else
                    None
                    ),
                },
            'registration': {
                'model': self.auto_registration_model_combo.currentText(),
                'channel': self.auto_registration_channel_combo.currentText(),
                },
            'segmentation': {
                'run_name': proposal_run,
                'reference_channel': self.auto_reference_channel_combo.currentText(),
                'reference_low_percentile': low,
                'reference_high_percentile': high,
                'checkpoint_path': str(checkpoint.resolve()),
                'checkpoint_file': checkpoint_record,
                'threshold': float(self.auto_threshold_spin.value()),
                'min_size': int(self.auto_min_size_spin.value()),
                'tta': self.auto_tta_check.isChecked(),
                'device': self.auto_device_combo.currentText(),
                },
            'extraction': {
                'run_name': fluorescence_run,
                'roi_run': roi_run,
                'surround_method': self.auto_surround_method_combo.currentText(),
                'surround_inner_px': int(self.auto_surround_inner_spin.value()),
                'surround_outer_px': int(self.auto_surround_outer_spin.value()),
                'surround_min_pixels': int(self.auto_surround_min_spin.value()),
                },
            'dff': {
                'run_name': dff_run,
                'fluorescence_run': fluorescence_run,
                'statistic': self.auto_statistic_combo.currentText(),
                'baseline_percentile': float(
                    self.auto_baseline_percentile_spin.value()),
                'baseline_window_s': float(self.auto_baseline_window_spin.value()),
                'surround_coefficient': float(
                    self.auto_surround_coefficient_spin.value()),
                'control_correction': self.auto_control_correction_combo.currentData(),
                },
            }

    def _remove_auto_partials(self, config):
        existing = [path for path in partial_paths(config) if path.exists()]
        if not existing:
            return True
        names = '\n'.join(path.name for path in existing)
        answer = QMessageBox.question(
            self,
            'Interrupted automatic stage',
            f'Remove the incomplete stage file before restarting?\n\n{names}',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
            )
        if answer != QMessageBox.Yes:
            return False
        for path in existing:
            path.unlink()
            self.print_log(f'removed incomplete stage file: {path}')
        return True

    def start_auto_session(self):
        try:
            config = self.auto_session_config()
        except Exception as exc:
            self.print_log(str(exc))
            self.refresh_status('automatic session configuration failed')
            return

        output_path = Path(config['output_path'])
        log_path = output_path.with_suffix('.fibresight.jsonl')
        resuming = (
            self.auto_log_path is not None
            and self.auto_log_path.resolve() == log_path.resolve()
            )
        if not resuming:
            if log_path.exists():
                self.print_log(f'session log already exists: {log_path}')
                self.refresh_status('open the existing session log to resume')
                return
            if output_path.exists():
                self.print_log(f'NWB output already exists: {output_path}')
                self.refresh_status('automatic output already exists')
                return
        else:
            previous = latest_session_config(log_path)
            if previous != config:
                self.print_log(
                    'automatic settings changed; reload the session log to restore them')
                self.refresh_status('automatic settings changed')
                return

        if not self._remove_auto_partials(config):
            return
        append_session_event(log_path, {'event': 'configured', 'config': config})
        self.auto_log_path = log_path
        self.auto_loaded_config = config
        started = self.start_process(
            'automatic session',
            'gui_worker',
            [str(log_path)],
            )
        if started:
            self.trace_cache = None
            self.trace_xlim = None
            self.auto_output_buffer = ''
            self.trace_run_combo.clear()
            self.trace_roi_combo.clear()
            self.auto_state_label.setText('automatic session running')
            self.auto_progress.setRange(0, 4)
            self.auto_progress.setValue(0)
            self.auto_stop_button.show()
            self.right_stack.hide()
            self.activity_tabs.setCurrentIndex(1)

    def choose_auto_session_log(self):
        selected, _ = QFileDialog.getOpenFileName(
            self,
            'open automatic session log',
            self.auto_output_dir_line.text().strip() or str(OUTPUT_ROOT),
            'FibreSight session logs (*.fibresight.jsonl);;JSON Lines (*.jsonl)',
            )
        if selected:
            self.load_auto_session_log(selected)

    def load_auto_session_log(self, log_path):
        log_path = Path(log_path)
        try:
            config = latest_session_config(log_path)
        except Exception as exc:
            self.print_log(f'failed to load automatic session: {exc}')
            return
        self.auto_log_path = log_path
        self.auto_loaded_config = config
        self.populate_auto_config(config)
        states = session_stage_states(log_path)
        completed = [
            stage for stage in ('preprocessing', 'segmentation', 'extraction', 'dff')
            if states.get(stage) in {'stage_completed', 'stage_skipped'}
            ]
        state_text = 'ready to resume'
        if completed:
            state_text = f'completed: {", ".join(completed)}'
        self.auto_state_label.setText(state_text)
        self.trace_cache = None
        self.trace_xlim = None
        self.trace_run_combo.clear()
        self.trace_roi_combo.clear()
        self.right_stack.hide()
        self.activity_tabs.setCurrentIndex(1)
        output_path = Path(config['output_path'])
        if output_path.exists():
            self.load_nwb_recording(output_path)
            if self.trace_cache is not None:
                self.right_stack.setCurrentIndex(0)
                self.right_stack.show()
                self.activity_tabs.setCurrentIndex(0)
                self.controls_tab_changed(self.tabs.currentIndex())
        self.print_log(f'loaded automatic session: {log_path}')
        if self.trace_cache is not None:
            total = max(
                sum(self.activity_splitter.sizes()),
                self.activity_splitter.height(),
                400,
                )
            self.activity_splitter.setSizes([
                int(total * 2 / 3),
                int(total / 3),
                ])
        self.refresh_status('automatic session ready to resume')

    def populate_auto_config(self, config):
        source = config['source']
        signal_files = config['signal_files']
        control_files = config['control_files']
        if signal_files:
            self.auto_tiff_dir_line.setText(str(Path(signal_files[0]['path']).parent))
        if control_files:
            self.auto_control_dir_line.setText(str(Path(control_files[0]['path']).parent))
        self.auto_acquisition_combo.setCurrentIndex(0 if source['multiplexed'] else 1)
        self.auto_output_dir_line.setText(str(Path(config['output_path']).parent))
        self.auto_session_line.setText(Path(config['output_path']).stem)
        self.auto_sampling_spin.setValue(source['sampling_frequency_hz'])
        self.auto_signal_channel_combo.setCurrentText(str(source['signal_channel']))
        self.auto_control_channel_combo.setCurrentText(str(source['control_channel']))
        self.auto_signal_label_line.setText(source['signal_label'])
        self.auto_control_label_line.setText(source['control_label'])
        self.auto_pixel_size_spin.setValue(source['pixel_size_um'] or 0)

        registration = config['registration']
        self.auto_registration_model_combo.setCurrentText(registration['model'])
        self.auto_registration_channel_combo.setCurrentText(registration['channel'])
        segmentation = config['segmentation']
        self.auto_reference_channel_combo.setCurrentText(
            segmentation['reference_channel'])
        self.auto_reference_low_spin.setValue(
            segmentation['reference_low_percentile'])
        self.auto_reference_high_spin.setValue(
            segmentation['reference_high_percentile'])
        self.auto_checkpoint_line.setText(segmentation['checkpoint_path'])
        self.auto_threshold_spin.setValue(segmentation['threshold'])
        self.auto_min_size_spin.setValue(segmentation['min_size'])
        self.auto_tta_check.setChecked(segmentation['tta'])
        self.auto_device_combo.setCurrentText(segmentation['device'])
        self.auto_proposal_run_line.setText(segmentation['run_name'])

        extraction = config['extraction']
        self.auto_surround_method_combo.setCurrentText(
            extraction['surround_method'])
        self.auto_surround_inner_spin.setValue(extraction['surround_inner_px'])
        self.auto_surround_outer_spin.setValue(extraction['surround_outer_px'])
        self.auto_surround_min_spin.setValue(extraction['surround_min_pixels'])
        self.auto_roi_source_combo.setCurrentText(extraction['roi_run'])
        self.auto_fluorescence_run_line.setText(extraction['run_name'])

        dff = config['dff']
        self.auto_dff_run_line.setText(dff['run_name'])
        self.auto_statistic_combo.setCurrentText(dff['statistic'])
        self.auto_baseline_percentile_spin.setValue(dff['baseline_percentile'])
        self.auto_baseline_window_spin.setValue(dff['baseline_window_s'])
        self.auto_surround_coefficient_spin.setValue(dff['surround_coefficient'])
        correction_index = self.auto_control_correction_combo.findData(
            dff['control_correction'])
        self.auto_control_correction_combo.setCurrentIndex(correction_index)

    def load_auto_result(self):
        if self.auto_loaded_config is None:
            return
        nwb_path = Path(self.auto_loaded_config['output_path'])
        if not nwb_path.exists():
            return
        self.load_nwb_recording(nwb_path)
        target_run = self.auto_loaded_config['dff']['run_name']
        target_index = self.trace_run_combo.findData(target_run)
        if target_index >= 0:
            self.trace_run_combo.setCurrentIndex(target_index)
        self.right_stack.setCurrentIndex(0)
        if self.tabs.currentWidget() is self.auto_tab:
            self.right_stack.show()
            self.activity_tabs.setCurrentIndex(0)
            self.controls_tab_changed(self.tabs.currentIndex())
        total = max(
            sum(self.activity_splitter.sizes()),
            self.activity_splitter.height(),
            400,
            )
        self.activity_splitter.setSizes([int(total * 2 / 3), int(total / 3)])

    def refresh_trace_runs(self):
        blocker = QSignalBlocker(self.trace_run_combo)
        self.trace_run_combo.clear()
        if self.nwb_path is not None:
            for run in list_dff_runs(self.nwb_path):
                self.trace_run_combo.addItem(run['run_name'], run['run_name'])
        del blocker
        if self.trace_run_combo.count():
            self.trace_run_combo.setCurrentIndex(0)
            self.load_trace_run()
        else:
            self.trace_cache = None
            self.trace_roi_combo.clear()
            self.draw_trace_inspector()

    def load_trace_run(self, _index=None):
        run_name = self.trace_run_combo.currentData()
        if self.nwb_path is None or not run_name:
            self.trace_cache = None
            self.draw_trace_inspector()
            return
        dff = load_dff_run(self.nwb_path, run_name)
        fluorescence = load_fluorescence_run(
            self.nwb_path,
            dff['fluorescence_run'],
            )
        with NWBHDF5IO(self.nwb_path, 'r') as io:
            nwbfile = io.read()
            container = nwbfile.processing['dff'][run_name]
            timestamps = np.asarray(
                container.roi_response_series['signal_roi_dff'].timestamps,
                dtype=float,
                )
            analysis_valid = np.asarray(
                nwbfile.processing['quality_control']['registration_qc'][
                    'analysis_valid'],
                dtype=bool,
                )
            metadata = nwbfile.processing['preprocessing']['recording_metadata']
            def metadata_text(name, fallback):
                if name not in metadata.colnames:
                    return fallback
                value = metadata[name][0]
                return value.decode() if isinstance(value, bytes) else str(value)

            labels = {
                'signal': metadata_text('signal_label', 'signal'),
                'control': metadata_text('control_label', 'control'),
                }
        self.trace_cache = {
            'run_name': run_name,
            'dff': dff,
            'fluorescence': fluorescence,
            'timestamps': timestamps,
            'analysis_valid': analysis_valid,
            'labels': labels,
            }
        self.trace_signal_check.setText(labels['signal'])
        self.trace_control_check.setText(labels['control'])
        self.trace_xlim = None
        blocker = QSignalBlocker(self.trace_roi_combo)
        self.trace_roi_combo.clear()
        for roi_id in dff['roi_ids']:
            self.trace_roi_combo.addItem(str(int(roi_id)), int(roi_id))
        del blocker
        if self.trace_roi_combo.count():
            self.trace_roi_combo.setCurrentIndex(0)
        self.draw_trace_inspector()

    def set_activity_view(self, index):
        self.activity_tabs.setCurrentIndex(index)

    def fit_trace(self):
        self.trace_xlim = None
        self.draw_trace_inspector()

    def zoom_trace(self, factor):
        if self.trace_cache is None or not self.trace_figure.axes:
            return
        axis = self.trace_figure.axes[0]
        x0, x1 = axis.get_xlim()
        centre = (x0 + x1) / 2
        half_width = (x1 - x0) / (2 * factor)
        timestamps = self.trace_cache['timestamps']
        lower = float(timestamps[0])
        upper = float(timestamps[-1])
        if upper <= lower:
            return
        half_width = min(half_width, (upper - lower) / 2)
        centre = float(np.clip(centre, lower + half_width, upper - half_width))
        self.trace_xlim = (centre - half_width, centre + half_width)
        axis.set_xlim(*self.trace_xlim)
        self.trace_canvas.draw_idle()

    def trace_scroll(self, event):
        if event.inaxes is None or event.button not in ('up', 'down'):
            return
        self.zoom_trace(1.5 if event.button == 'up' else 1 / 1.5)

    def draw_trace_inspector(self, _index=None):
        self.trace_figure.clear()
        if self.trace_cache is None or self.trace_roi_combo.currentData() is None:
            axis = self.trace_figure.add_subplot(111)
            axis.axis('off')
            axis.text(
                0.5,
                0.5,
                'no dF/F run',
                ha='center',
                va='center',
                color=self._theme()['muted'],
                )
            self.trace_canvas.draw_idle()
            return

        cache = self.trace_cache
        roi_id = int(self.trace_roi_combo.currentData())
        roi_ids = np.asarray(cache['dff']['roi_ids'], dtype=int)
        roi_index = int(np.flatnonzero(roi_ids == roi_id)[0])
        timestamps = cache['timestamps']
        analysis_valid = cache['analysis_valid']
        dff = cache['dff']['traces']
        colours = self._theme()

        axis = self.trace_figure.add_subplot(111)
        labels = cache.get('labels', {'signal': 'signal', 'control': 'control'})
        if self.trace_signal_check.isChecked():
            signal = dff['signal_surround_corrected_dff'][:, roi_index].copy()
            signal[~analysis_valid] = np.nan
            axis.plot(
                timestamps,
                signal,
                color='#D55E00',
                linewidth=0.9,
                label=labels['signal'],
                )
        if self.trace_control_check.isChecked():
            control = dff['control_surround_corrected_dff'][:, roi_index].copy()
            control[~analysis_valid] = np.nan
            axis.plot(
                timestamps,
                control,
                color='#0072B2',
                linewidth=0.9,
                label=labels['control'],
                )
        axis.set_ylabel('dF/F')
        axis.set_xlabel('time (s)')
        axis.set_title(f'ROI {roi_id} dF/F')
        axis.set_facecolor(colours['canvas'])
        axis.tick_params(colors=colours['text'])
        axis.xaxis.label.set_color(colours['text'])
        axis.yaxis.label.set_color(colours['text'])
        axis.title.set_color(colours['text'])
        axis.grid(axis='y', color=colours['border'], linewidth=0.6)
        axis.spines[['top', 'right']].set_visible(False)
        axis.spines['left'].set_color(colours['border'])
        axis.spines['bottom'].set_color(colours['border'])
        if axis.lines:
            axis.legend(frameon=False, fontsize=8, labelcolor=colours['text'])
        if self.trace_xlim is None:
            axis.set_xlim(float(timestamps[0]), float(timestamps[-1]))
        else:
            axis.set_xlim(*self.trace_xlim)
        self.trace_figure.patch.set_facecolor(colours['canvas'])
        self.trace_figure.tight_layout(pad=1.3)
        self.trace_canvas.draw_idle()

    #%% browse
    def browse_source_root(self):
        path = QFileDialog.getExistingDirectory(
            self,
            'select training source root',
            self.source_root_line.text(),
            )
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

    def browse_training_output_dir(self):
        path = QFileDialog.getExistingDirectory(
            self,
            'select training output folder',
            self.train_output_dir_line.text().strip() or str(OUTPUT_ROOT),
            )
        if path:
            self.train_output_dir_line.setText(path)

    def browse_image(self):
        start_path = self.image_line.text().strip() or str(WORKSPACE_ROOT)
        path, _ = QFileDialog.getOpenFileName(
            self,
            'select channel-2 reference or NWB recording',
            start_path,
            'Reference images and NWB recordings (*.npy *.nwb)',
            )
        if path:
            self.image_line.setText(path)
        return path

    def browse_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            'select trained model',
            str(OUTPUT_ROOT / 'runs'),
            'Trained models (*.pt)',
            )
        if path:
            self.checkpoint_line.setText(path)

    #%% training processes
    def training_checkpoint_path(self):
        output_dir = resolve_path(
            self.train_output_dir_line.text().strip(),
            WORKSPACE_ROOT,
            )
        return output_dir / self.run_name_line.text().strip() / 'best.pt'

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

        checkpoint_path = self.training_checkpoint_path()
        args = ['--config', str(config_path)]
        if self.start_process('train', 'train_unet', args):
            self.pending_checkpoint_path = checkpoint_path

    def evaluate_model(self):
        model_path = self.training_checkpoint_path()
        if not model_path.exists():
            self.print_log(f'trained model does not exist: {model_path}')
            self.refresh_status('model file missing')
            return
        run_recipe = model_path.parent / 'config.yaml'
        if not run_recipe.exists():
            self.print_log(f'training recipe does not exist: {run_recipe}')
            self.refresh_status('training recipe missing')
            return
        manifest = Path(self.manifest_line.text().strip())
        if not manifest.exists():
            self.print_log(f'dataset table does not exist yet: {manifest}')
            self.refresh_status('dataset table missing')
            return
        config = load_recipe(run_recipe)

        args = [
            '--manifest', str(manifest),
            '--checkpoint', str(model_path),
            '--split', 'test',
            '--threshold', str(config['postprocess']['threshold']),
            '--min-size', str(config['postprocess']['min_size']),
            '--tta',
            '--device', 'auto',
            ]
        self.start_process('evaluate', 'evaluate', args)

    def inspect_manifest(self):
        path = Path(self.manifest_line.text().strip())
        if not path.exists():
            self.print_log(f'dataset table not found: {path}')
            self.refresh_status('dataset table missing')
            return

        sessions = read_manifest(path)
        included = [session for session in sessions if session['included']]
        split_counts = {}
        for session in included:
            split = session['split'] or 'unsplit'
            split_counts[split] = split_counts.get(split, 0) + 1

        excluded_reasons = {}
        for session in sessions:
            if session['included']:
                continue
            reason = session['exclusion_reason'] or 'not included'
            excluded_reasons[reason] = excluded_reasons.get(reason, 0) + 1

        roi_total = sum(session['roi_count'] for session in included)
        self.print_log(f'\ndataset table: {path}')
        self.print_log(f'total sessions: {len(sessions)}')
        self.print_log(f'included sessions: {len(included)} | total ROIs: {roi_total}')
        self.print_log(f'splits: {split_counts if split_counts else {}}')
        if excluded_reasons:
            self.print_log(f'excluded: {excluded_reasons}')
        self.refresh_status('dataset summary ready')

    def write_training_config(self):
        config = load_recipe(self.config_line.text())
        run_name = self.run_name_line.text().strip()
        if not run_name:
            raise ValueError('run name cannot be empty')

        # leave the baseline YAML alone whilst trying controls here; save this recipe beside the run
        config['data']['manifest'] = self.path_for_config(self.manifest_line.text())
        config['train']['run_name'] = run_name
        output_dir = self.train_output_dir_line.text().strip()
        if not output_dir:
            raise ValueError('output folder cannot be empty')
        config['train']['out_dir'] = self.path_for_config(output_dir)
        config['train']['epochs'] = int(self.epochs_spin.value())
        config['train']['device'] = 'auto'

        out_path = OUTPUT_ROOT / 'gui_configs' / f'{run_name}.yaml'
        save_recipe(config, out_path)
        self.print_log(f'training recipe written to {out_path}')
        return out_path

    def start_process(self, process_name, module_name, args):
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            self.print_log('another process is already running')
            return False

        self.current_process_name = process_name
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
        self.refresh_status(f'{process_name} started')
        return True

    def stop_process(self):
        if self.process is None or self.process.state() == QProcess.NotRunning:
            self.print_log('no process is running')
            return

        self._process_was_stopped = True
        if self.current_process_name == 'automatic session' and self.auto_log_path:
            append_session_event(
                self.auto_log_path,
                {'event': 'session_cancelled'},
                )
        self.process.kill()
        self.print_log('process stopped')

    def read_process_stdout(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardOutput()).decode(errors='replace')
        self.print_log(text, end='')
        if self.current_process_name == 'automatic session':
            self.auto_output_buffer += text
            lines = self.auto_output_buffer.split('\n')
            self.auto_output_buffer = lines.pop()
            stages = {
                'preprocessing': 1,
                'segmentation': 2,
                'extraction': 3,
                'dF/F': 4,
            }
            for line in lines:
                line = line.strip()
                stage = next(
                    (name for name in stages if line.startswith(f'{name}:')),
                    None,
                    )
                if stage is None:
                    continue
                self.auto_state_label.setText(line)
                if (
                        'completed' in line
                        or 'existing result retained' in line
                        ):
                    self.auto_progress.setValue(stages[stage])
                else:
                    self.auto_progress.setValue(stages[stage] - 1)

    def read_process_stderr(self):
        if self.process is None:
            return
        text = bytes(self.process.readAllStandardError()).decode(errors='replace')
        self.print_log(text, end='')

    def process_finished(self, exit_code, exit_status):
        self.read_process_stdout()
        self.read_process_stderr()
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
            self.clear_probability()
            self.print_log(f'trained model ready: {self.pending_checkpoint_path}')

        automatic_succeeded = (
            self.current_process_name == 'automatic session'
            and not self._process_was_stopped
            and exit_code == 0
            and exit_status == QProcess.NormalExit
            )
        if self.current_process_name == 'automatic session':
            self.auto_progress.setRange(0, 4)
            self.auto_progress.setValue(4 if automatic_succeeded else 0)
            self.auto_stop_button.hide()
            self.auto_state_label.setText(
                'automatic session complete'
                if automatic_succeeded else
                'automatic session interrupted'
                )
            if not automatic_succeeded:
                self.right_stack.hide()
        if automatic_succeeded:
            self.load_auto_result()

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

        image_path = Path(path)
        if image_path.suffix.lower() == '.nwb':
            self.load_nwb_recording(image_path)
            return

        try:
            image = squeeze_image(np.load(image_path))
        except Exception as exc:
            self.print_log(f'failed to load image: {exc}')
            return

        self.nwb_path = None
        self.source_proposal_run = None
        self.set_nwb_controls_visible(False)
        self.ref_image = image
        self.image_path = image_path
        self.recname = self.recording_name(self.image_path)
        self.roi_dict = {}
        self.labelled = np.zeros_like(self.ref_image, dtype=np.int32)
        self.selected.clear()
        self.fixed_ids.clear()
        self.undo_stack.clear()
        self.clear_probability(redraw=False)
        self.reset_display_range(redraw=False)

        default_roi = self.default_roi_path()
        if default_roi.exists():
            self.print_log(f'found existing ROI dict: {default_roi}')
            self.print_log('import ROIs to review or continue from them')
        self.update_export_tooltip()
        self.plot_image()
        self.canvas.fit_to_image()
        self.refresh_status('channel-2 image loaded')

    def set_nwb_controls_visible(self, visible):
        for widget in (
            self.proposal_run_label,
            self.proposal_run_combo,
            self.curated_run_label,
            self.curated_run_line,
            ):
            widget.setVisible(visible)

    def load_nwb_recording(self, nwb_path):
        try:
            proposal_runs = list_roi_runs(nwb_path, run_type='proposed')
        except Exception as exc:
            self.print_log(f'failed to read NWB proposal runs: {exc}')
            self.refresh_status('NWB load failed')
            return

        self.nwb_path = Path(nwb_path)
        self.source_proposal_run = None
        self.image_path = self.nwb_path
        self.recname = self.recording_name(self.nwb_path)
        self.ref_image = None
        self.roi_dict = {}
        self.labelled = None
        self.selected.clear()
        self.fixed_ids.clear()
        self.undo_stack.clear()
        self.clear_probability(redraw=False)
        self.set_nwb_controls_visible(True)

        blocker = QSignalBlocker(self.proposal_run_combo)
        self.proposal_run_combo.clear()
        self.proposal_run_combo.addItem('select proposal run', None)
        for proposal_run in proposal_runs:
            run_name = proposal_run['run_name']
            self.proposal_run_combo.addItem(run_name, run_name)
        del blocker

        self.curated_run_line.clear()
        if len(proposal_runs) == 1:
            self.proposal_run_combo.setCurrentIndex(1)
        else:
            self.plot_image()
        self.refresh_trace_runs()
        roi_runs = list_roi_runs(nwb_path)
        roi_blocker = QSignalBlocker(self.auto_roi_source_combo)
        current_roi_run = self.auto_roi_source_combo.currentText()
        self.auto_roi_source_combo.clear()
        for run in roi_runs:
            self.auto_roi_source_combo.addItem(run['run_name'])
        if current_roi_run:
            self.auto_roi_source_combo.setCurrentText(current_roi_run)
        del roi_blocker
        if proposal_runs:
            self.refresh_status('select an NWB proposal run')
        else:
            self.print_log(f'no proposal runs found in {self.nwb_path}')
            self.refresh_status('no NWB proposal runs found')

    def load_selected_proposal_run(self, _index=None):
        run_name = self.proposal_run_combo.currentData()
        if self.nwb_path is None:
            return
        if run_name is None:
            self.source_proposal_run = None
            self.ref_image = None
            self.roi_dict = {}
            self.labelled = None
            self.selected.clear()
            self.fixed_ids.clear()
            self.undo_stack.clear()
            self.clear_probability(redraw=False)
            self.plot_image()
            self.refresh_status('select an NWB proposal run')
            return

        try:
            loaded_run = load_roi_run(self.nwb_path, run_name)
            labelled, _ = roi_dict_to_label(
                loaded_run['roi_dict'],
                loaded_run['reference'].shape,
                )
        except Exception as exc:
            self.print_log(f'failed to load proposal run {run_name}: {exc}')
            self.refresh_status('proposal run load failed')
            return

        self.source_proposal_run = run_name
        self.ref_image = loaded_run['reference']
        self.labelled = labelled
        self.roi_dict = labels_to_roi_dict(labelled)
        self.probability = loaded_run['probability']
        recorded_max_size = int(loaded_run['provenance']['max_size'])
        self.probability_max_size = (
            None if recorded_max_size == -1 else recorded_max_size
            )
        self.threshold_spin.setValue(float(loaded_run['provenance']['threshold']))
        self.min_size_spin.setValue(int(loaded_run['provenance']['min_size']))
        self.selected.clear()
        self.fixed_ids.clear()
        self.undo_stack.clear()
        self.set_display_mode('image')
        self.reset_display_range(redraw=False)
        self.canvas.fit_to_image()
        self.print_log(
            f'loaded proposal run {run_name} from {self.nwb_path}'
            )
        self.refresh_status('NWB proposal loaded')

    def load_model(self):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage('loading trained model')
        QApplication.processEvents()
        try:
            checkpoint = self.checkpoint_line.text().strip()
            self.predictor = ROIPredictor(
                checkpoint_path=Path(checkpoint) if checkpoint else None,
                device='auto',
                threshold=float(self.threshold_spin.value()),
                min_size=int(self.min_size_spin.value()),
                tta=True,
                )
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
        self.clear_probability()
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
            roi_dict, labelled, probability = self.predictor.predict_image(
                self.ref_image
                )
        except Exception as exc:
            self.print_log(f'prediction failed: {exc}')
            self.refresh_status('prediction failed')
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.push_undo_state()
        # prediction returns to the same editable state as an MSER or loaded ROI dict
        self.roi_dict = roi_dict
        self.labelled = labelled
        self.probability = probability
        self.probability_max_size = self.predictor.max_size
        self.selected.clear()
        self.fixed_ids.clear()

        self.set_display_mode('image')
        self.canvas.fit_to_image()
        self.print_log(
            f'predicted {len(self.roi_dict)} ROIs '
            f'(threshold {self.predictor.threshold:.2f}, '
            f'min area {self.predictor.min_size} px)'
            )
        self.refresh_status()
        self.statusBar().clearMessage()

    def rebuild_rois_from_probability(self):
        if self.probability is None:
            return

        # threshold changes reuse the confidence map whilst I decide which faint fibres to keep
        self.push_undo_state()
        self.roi_dict, self.labelled = probability_to_roi_dict(
            self.probability,
            threshold=float(self.threshold_spin.value()),
            min_size=int(self.min_size_spin.value()),
            max_size=self.probability_max_size,
            )
        self.selected.clear()
        self.fixed_ids.clear()
        self.plot_image(preserve_view=True)
        self.print_log(f'rebuilt {len(self.roi_dict)} ROIs from the confidence map')
        self.refresh_status('ROIs rebuilt from the confidence map')

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
            labelled, _ = roi_dict_to_label(roi_dict, self.ref_image.shape)
        except Exception as exc:
            self.print_log(f'failed to import ROIs: {exc}')
            return

        self.push_undo_state()
        self.roi_dict = labels_to_roi_dict(labelled)
        self.labelled = labelled
        self.selected.clear()
        self.fixed_ids.clear()
        self.clear_probability(redraw=False)
        self.plot_image()
        self.print_log(f'imported ROIs from {path}')
        self.refresh_status('ROIs imported')

    def save_roi_file(self):
        if not self.image_selection_matches_loaded():
            self.print_log('please load the selected channel-2 image first')
            return

        self.update_roi_dict()
        if self.nwb_path is not None:
            run_name = self.curated_run_line.text().strip()
            if not run_name:
                self.print_log('enter a name for the curated NWB run')
                self.refresh_status('curated run name required')
                return

            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.statusBar().showMessage('saving curated NWB run')
            QApplication.processEvents()
            try:
                result = save_curated_rois(
                    self.nwb_path,
                    run_name,
                    self.roi_dict,
                    self.source_proposal_run,
                    )
            except Exception as exc:
                self.print_log(f'failed to save curated run: {exc}')
                self.refresh_status('curated run save failed')
                return
            finally:
                QApplication.restoreOverrideCursor()

            self.print_log(
                f'saved {result["roi_count"]} ROIs as curated run {run_name}'
                )
            self.refresh_status('curated NWB run saved')
            return

        out_path = self.default_roi_path()
        save_roi_dict(self.roi_dict, out_path)
        self.print_log(f'exported ROIs to {out_path}')
        self.update_export_tooltip()
        self.refresh_status('ROIs exported')

    def default_roi_path(self):
        if self.image_path is None:
            return OUTPUT_ROOT / 'predicted_ROI_dict.npy'

        example_root = PACKAGE_ROOT.parents[1] / 'examples'
        if (
            example_root.is_dir() and
            self.image_path.resolve().is_relative_to(example_root.resolve())
        ):
            # shipped examples stay unchanged; their edited output belongs in workspace
            return OUTPUT_ROOT / 'demo_predicted_ROI_dict.npy'

        return self.image_path.parent / f'{self.recname}_ROI_dict.npy'

    def update_export_tooltip(self):
        if self.nwb_path is not None:
            self.curate_buttons['save_roi'].setText('save curated run')
            self.curate_buttons['save_roi'].setToolTip(
                f'append a new immutable ROI run to {self.nwb_path}'
                )
            return

        self.curate_buttons['save_roi'].setText('export ROIs')
        self.curate_buttons['save_roi'].setToolTip(
            f'export immediately to {self.default_roi_path()}'
            )


    #%% mser segmentation
    def reset_segment_parameters(self):
        for spec in PARAMETER_SPECS:
            self.segment_param_widgets[spec['name']].setValue(spec['default'])
        self.refresh_status('parameters reset')

    def segment_params(self):
        params = {}
        for spec in PARAMETER_SPECS:
            value = self.segment_param_widgets[spec['name']].value()
            params[spec['name']] = int(value) if spec['kind'] == 'int' else float(value)
        return params

    def fixed_rois(self):
        if not self.roi_dict or not self.fixed_ids:
            return {}
        return {
            roi_id: {
                'xpix': self.roi_dict[roi_id]['xpix'].copy(),
                'ypix': self.roi_dict[roi_id]['ypix'].copy(),
            }
            for roi_id in sorted(self.fixed_ids)
        }

    def segment_rois(self):
        if not self.image_selection_matches_loaded():
            self.print_log('please load the selected channel-2 image first')
            return

        # the MSER route still helps before a model exists and on images that need hand repair
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.statusBar().showMessage('running MSER segmentation')
        QApplication.processEvents()
        try:
            roi_dict, labelled, fixed_ids, stats = segment_mser(
                self.ref_image,
                self.segment_params(),
                fixed_rois=self.fixed_rois(),
            )
        except Exception as exc:
            self.print_log(f'segmentation failed: {exc}')
            self.refresh_status('segmentation failed')
            return
        finally:
            QApplication.restoreOverrideCursor()

        if self.labelled is not None:
            self.push_undo_state()
        self.roi_dict = roi_dict
        self.labelled = labelled
        self.fixed_ids = fixed_ids
        self.selected.clear()
        self.clear_probability(redraw=False)
        self.plot_image()
        self.canvas.fit_to_image()
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
            self.fixed_ids.difference_update(selected_ids)
            status = 'selected ROIs unfixed'
        else:
            self.fixed_ids.update(selected_ids)
            status = 'selected ROIs fixed'
        self.plot_image(preserve_view=True)
        self.refresh_status(status)

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

        self.refresh_roi_table()
        self.sync_trace_roi_selection()
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
        xlim, ylim = self.current_view()
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

    def current_view(self):
        if not self.ax.images:
            return None, None
        return self.ax.get_xlim(), self.ax.get_ylim()

    def reset_view(self):
        self.canvas.fit_to_image()
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
        if self.tabs.currentWidget() is self.auto_tab:
            session_name = self.auto_session_line.text().strip() or 'not named'
            self.state_label.setText(f'session: {session_name}')
            self.state_label.setToolTip(
                str(self.auto_log_path) if self.auto_log_path else 'automatic session'
                )
            self.roi_label.setText('')
            self.model_separator.setText('')
            self.model_label.setText('')
            self.update_controls()
            if message:
                self.statusBar().showMessage(message, 4000)
            return
        image_name = self.image_path.name if self.image_path else 'not loaded'
        if self.source_proposal_run:
            image_name = f'{image_name} · {self.source_proposal_run}'
        roi_count = len(self.roi_dict)
        selected_count = len(self.selected)
        fixed_count = len(self.fixed_ids)
        self.state_label.setText(
            image_name if self.image_path else 'image: not loaded'
            )
        self.state_label.setToolTip(
            str(self.image_path) if self.image_path else 'no channel-2 image loaded'
            )
        self.model_separator.setText('·')
        self.roi_label.setText(
            f'{roi_count} ROIs | {selected_count} selected | {fixed_count} fixed'
            )
        self.model_label.setText(self.model_status_text())
        self.update_export_tooltip()
        self.update_controls()
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

        matches = (
            Path(self.predictor.checkpoint_path).resolve() ==
            Path(checkpoint).resolve()
            )
        if not matches:
            return f'model: {name} · {next_step}'
        return f'model: {name} · {self.predictor.device}'

    def update_controls(self):
        image_ready = self.image_selection_matches_loaded()
        checkpoint_text = self.checkpoint_line.text().strip()
        checkpoint_ready = bool(checkpoint_text) and Path(checkpoint_text).exists()
        training_run_dir = self.training_checkpoint_path().parent
        training_model_ready = (
            (training_run_dir / 'best.pt').exists()
            and (training_run_dir / 'config.yaml').exists()
            )
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

        self.auto_run_button.setEnabled(not process_running)
        self.auto_resume_button.setEnabled(not process_running)
        self.auto_stop_button.setEnabled(process_running)

        self.build_manifest_button.setEnabled(source_ready and not process_running)
        self.inspect_manifest_button.setEnabled(manifest_ready and not process_running)
        self.train_model_button.setEnabled(
            source_ready
            and config_ready
            and bool(self.train_output_dir_line.text().strip())
            and bool(self.run_name_line.text().strip())
            and not process_running
            )
        self.evaluate_model_button.setEnabled(
            manifest_ready and training_model_ready and not process_running)
        self.stop_process_button.setEnabled(process_running)
        self.stop_process_button.setVisible(process_running)

        self.predict_button.setEnabled(
            image_ready and checkpoint_ready and not process_running
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
        if self.nwb_path is not None:
            self.curate_buttons['save_roi'].setEnabled(
                image_ready
                and bool(self.curated_run_line.text().strip())
                )
        if hasattr(self, 'controls_split_timer'):
            self.controls_split_timer.start(0)

    def print_log(self, text, end='\n'):
        self.output_box.moveCursor(QTextCursor.End)
        self.output_box.insertPlainText(f'{text}{end}')
        self.output_box.moveCursor(QTextCursor.End)
        # the log pane starts collapsed; open it far enough to read the newest lines
        if self.activity_splitter.sizes()[1] < 72:
            total = max(sum(self.activity_splitter.sizes()), self.activity_splitter.height(), 400)
            self.activity_splitter.setSizes([total - 105, 105])

    @staticmethod
    def recording_name(path):
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
    window = FibreSightGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
