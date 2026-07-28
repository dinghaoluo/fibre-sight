'''
Created on 12 May 2026

Modified on 13 May 2026 while wiring the MSER controls into the workbench
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
import sys

import matplotlib

matplotlib.use('Qt5Agg')

import numpy as np
from matplotlib import font_manager
from matplotlib.figure import Figure
from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtGui import QColor, QFont, QFontDatabase, QIcon, QKeySequence, QPalette, QTextCursor
from PyQt5.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QPlainTextEdit,
    QShortcut,
    QSizePolicy,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTabWidget,
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
CORE_SEGMENT_PARAMETERS = (
    'MSER threshold',
    'MSER min area',
    'MSER max area',
    'area min',
    )


#%% helpers
def load_gui_font(size=9):
    for font_path in sorted(MONONOKI_FONT_DIR.glob('*.ttf')):
        QFontDatabase.addApplicationFont(str(font_path))
        font_manager.fontManager.addfont(str(font_path))

    matplotlib.rcParams['font.family'] = MONONOKI_FONT_FAMILY
    matplotlib.rcParams['font.monospace'] = [MONONOKI_FONT_FAMILY, 'Consolas', 'Courier New']
    return QFont(MONONOKI_FONT_FAMILY, size)


#%% main window
class FibreSightWorkbench(QMainWindow):
    def __init__(self):
        super().__init__()
        app = QApplication.instance()
        if app is not None:
            app.setFont(load_gui_font())
        self.setWindowTitle('fibre-sight')
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
        self.loaded_device_choice = None
        self.dark_mode = False

        self._build_widgets()
        self._build_layout()
        self._apply_palette_and_style()
        self._connect_shortcuts()
        self.plot_image()
        self.refresh_status()

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
            'channel-2 image and editable ROI overlay; '
            'click an ROI to select; Shift-click adds; right-drag pans; scroll zooms'
            )
        self.canvas.mpl_connect('button_press_event', self.on_click)

        self.state_label = QLabel('image: not loaded')
        self.state_label.setObjectName('stateSummary')
        self.roi_label = QLabel('0 ROIs | 0 selected | 0 fixed')
        self.roi_label.setObjectName('panelValue')
        self.model_label = QLabel('model: not loaded')
        self.model_label.setObjectName('panelValue')

        self.tabs = QTabWidget()
        self.predict_tab, self.predict_tab_content = self._make_scroll_tab()
        self.mser_tab, self.mser_tab_content = self._make_scroll_tab()
        self.training_tab, self.training_tab_content = self._make_scroll_tab()
        self.tabs.addTab(self.predict_tab, 'Predict')
        self.tabs.addTab(self.mser_tab, 'Label')
        self.tabs.addTab(self.training_tab, 'Train')
        self.tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.tabs.currentChanged.connect(lambda _: self.refresh_status())
        self._build_training_widgets()
        self._build_prediction_widgets()
        self._build_segment_widgets()
        self._build_editing_widgets()
        self.output_box = self.make_log_box()

        self.roi_overlay_check = QCheckBox('ROI on')
        self.roi_overlay_check.setChecked(True)
        self.roi_overlay_check.stateChanged.connect(self.set_roi_overlay_visible)
        self.dark_mode_check = QCheckBox('dark mode')
        self.dark_mode_check.stateChanged.connect(self.set_dark_mode)

    def _make_scroll_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        content = QWidget()
        content.setObjectName('tabContent')
        scroll.setWidget(content)
        return scroll, content

    def _build_training_widgets(self):
        self.source_root_line = QLineEdit(str(default_source_root()))
        self.manifest_line = QLineEdit(str(WORKSPACE_ROOT / 'manifests' / 'ch2_manifest.csv'))
        self.config_line = QLineEdit(
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

        self.device_combo = QComboBox()
        self.device_combo.addItem('automatic', 'auto')
        self.device_combo.addItem('Apple GPU (MPS)', 'mps')
        self.device_combo.addItem('NVIDIA GPU (CUDA)', 'cuda')
        self.device_combo.addItem('CPU', 'cpu')
        self.device_combo.setToolTip(
            'automatic uses CUDA when available, then Apple MPS, then CPU; '
            'the same setting applies to prediction and scoring'
            )
        self.device_combo.currentIndexChanged.connect(self.compute_device_changed)

        self.build_manifest_button = QPushButton('scan labelled sessions')
        self.train_model_button = QPushButton('train model')
        self.evaluate_model_button = QPushButton('score model')
        self.stop_process_button = QPushButton('stop process')
        self.inspect_manifest_button = QPushButton('dataset summary')
        self.preview_training_button = QPushButton('preview labels')
        self.preview_predictions_button = QPushButton('preview predictions')
        self.training_diagnostics_button = QPushButton('diagnostics')
        self.training_diagnostics_popup = None
        self.training_advanced_button = QPushButton('advanced options')
        self.training_advanced_popup = None
        self.set_button_role(self.train_model_button, 'primary')
        self.set_button_role(self.stop_process_button, 'danger')
        self.set_button_role(self.training_diagnostics_button, 'quiet')
        self.set_button_role(self.training_advanced_button, 'quiet')
        self.set_button_role(self.inspect_manifest_button, 'quiet')
        self.set_button_role(self.evaluate_model_button, 'quiet')
        self.set_button_role(self.preview_training_button, 'quiet')
        self.set_button_role(self.preview_predictions_button, 'quiet')
        self.evaluate_model_button.setToolTip('score the current trained model on held-out labelled sessions')
        self.preview_predictions_button.setToolTip('save example overlays comparing model ROIs with held-out labels')
        self.training_diagnostics_button.setToolTip('open model scoring and preview tools')
        self.training_advanced_button.setToolTip(
            'open the dataset table, training recipe, split fractions, and device setting'
            )
        self.stop_process_button.hide()

        self.build_manifest_button.clicked.connect(self.build_manifest)
        self.train_model_button.clicked.connect(self.train_model)
        self.evaluate_model_button.clicked.connect(self.evaluate_model)
        self.inspect_manifest_button.clicked.connect(self.inspect_manifest)
        self.preview_training_button.clicked.connect(self.preview_training_labels)
        self.preview_predictions_button.clicked.connect(self.preview_model_predictions)
        self.training_diagnostics_button.clicked.connect(self.show_training_diagnostics_popup)
        self.training_advanced_button.clicked.connect(self.show_training_advanced_popup)
        self.stop_process_button.clicked.connect(self.stop_process)

    def _build_prediction_widgets(self):
        self.image_line = QLineEdit()
        self.checkpoint_line = QLineEdit(str(get_default_checkpoint()))
        self.image_line.textChanged.connect(self.prediction_inputs_changed)
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

        self.show_probability_check = QCheckBox('show model confidence')
        self.show_probability_check.stateChanged.connect(lambda _: self.plot_image(preserve_view=True))

        self.predict_button = QPushButton('run prediction')
        self.apply_settings_button = QPushButton('apply settings')
        self.set_button_role(self.predict_button, 'primary')
        self.set_button_role(self.apply_settings_button, 'secondary')
        self.predict_button.setToolTip(
            'load the selected image and model, then predict ROIs'
            )
        self.apply_settings_button.setToolTip(
            'rebuild ROIs from the retained confidence map; '
            'Undo restores the previous ROI state'
            )
        self.apply_settings_button.hide()
        self.show_probability_check.hide()

        self.threshold_spin.setToolTip('higher values keep only stronger model detections')
        self.min_size_spin.setToolTip('smallest ROI size to keep')

        self.predict_button.clicked.connect(self.predict_rois)
        self.apply_settings_button.clicked.connect(self.apply_probability_threshold)


    def _build_segment_widgets(self):
        self.segment_param_widgets = {}
        for spec in PARAMETER_SPECS:
            self.segment_param_widgets[spec['name']] = self._make_segment_param_widget(spec)

        self.segment_button = QPushButton('segment')
        self.reset_segment_button = QPushButton('reset params')
        self.advanced_segment_button = QPushButton('more parameters')
        self.fix_selected_button = QPushButton('fix selected')
        self.unfix_selected_button = QPushButton('unfix selected')
        self.clear_fixed_button = QPushButton('clear fixed')
        self.clear_unfixed_button = QPushButton('clear unfixed')
        self.segment_load_image_button = QPushButton('open image')
        self.segment_load_roi_button = QPushButton('load ROI dict')
        self.segment_advanced_popup = None

        self.set_button_role(self.segment_button, 'primary')
        self.set_button_role(self.segment_load_image_button, 'secondary')
        self.set_button_role(self.segment_load_roi_button, 'secondary')
        self.set_button_role(self.reset_segment_button, 'quiet')
        self.set_button_role(self.advanced_segment_button, 'quiet')
        self.set_button_role(self.fix_selected_button, 'secondary')
        self.set_button_role(self.unfix_selected_button, 'secondary')
        self.set_button_role(self.clear_fixed_button, 'quiet')
        self.set_button_role(self.clear_unfixed_button, 'dangerQuiet')

        self.fix_selected_button.setToolTip('keep selected ROIs when segmentation is run again')
        self.unfix_selected_button.setToolTip('allow selected ROIs to change during segmentation')
        self.clear_fixed_button.setToolTip('remove all fixed marks without deleting ROIs')
        self.clear_unfixed_button.setToolTip('remove all ROIs except fixed ROIs')

        self.segment_button.clicked.connect(self.segment_rois)
        self.reset_segment_button.clicked.connect(self.reset_segment_parameters)
        self.fix_selected_button.clicked.connect(self.fix_selected)
        self.unfix_selected_button.clicked.connect(self.unfix_selected)
        self.clear_fixed_button.clicked.connect(self.clear_fixed)
        self.clear_unfixed_button.clicked.connect(self.clear_unfixed)
        self.advanced_segment_button.clicked.connect(self.show_segment_advanced_popup)
        self.segment_load_image_button.clicked.connect(self.choose_channel_image)
        self.segment_load_roi_button.clicked.connect(self.load_roi_file)

    def _build_editing_widgets(self):
        self.curate_buttons = {
            'select_all': QPushButton('select all'),
            'delete': QPushButton('delete'),
            'merge': QPushButton('merge'),
            'undo': QPushButton('undo'),
            'reset_view': QPushButton('reset view'),
            'save_roi': QPushButton('save ROI dict'),
        }
        self.set_button_role(self.curate_buttons['delete'], 'danger')
        self.set_button_role(self.curate_buttons['select_all'], 'quiet')
        self.set_button_role(self.curate_buttons['merge'], 'quiet')
        self.set_button_role(self.curate_buttons['undo'], 'quiet')
        self.set_button_role(self.curate_buttons['reset_view'], 'quiet')
        self.set_button_role(self.curate_buttons['save_roi'], 'primary')

        self.curate_buttons['select_all'].clicked.connect(self.select_all)
        self.curate_buttons['delete'].clicked.connect(self.delete_selected)
        self.curate_buttons['merge'].clicked.connect(self.merge_selected)
        self.curate_buttons['undo'].clicked.connect(self.undo)
        self.curate_buttons['reset_view'].clicked.connect(self.reset_view)
        self.curate_buttons['save_roi'].clicked.connect(self.save_roi_file)

    def _build_layout(self):
        canvas_layout = QVBoxLayout()
        canvas_layout.setContentsMargins(8, 8, 8, 8)
        canvas_layout.addWidget(self.canvas)
        canvas_frame = QFrame()
        canvas_frame.setObjectName('canvasFrame')
        canvas_frame.setLayout(canvas_layout)

        curation_bar = QFrame()
        curation_bar.setObjectName('curationBar')
        curation_layout = QVBoxLayout(curation_bar)
        curation_layout.setContentsMargins(8, 6, 8, 6)

        curation_buttons = QHBoxLayout()
        curation_buttons.setSpacing(4)
        curation_buttons.addWidget(self.curate_buttons['select_all'])
        curation_buttons.addWidget(self.curate_buttons['merge'])
        curation_buttons.addWidget(self.curate_buttons['delete'])
        curation_buttons.addWidget(self.curate_buttons['undo'])
        curation_buttons.addStretch(1)
        curation_buttons.addWidget(self.curate_buttons['reset_view'])
        curation_buttons.addWidget(self.curate_buttons['save_roi'])
        curation_layout.addLayout(curation_buttons)

        canvas_stage = QWidget()
        canvas_stage.setObjectName('canvasStage')
        canvas_stage_layout = QVBoxLayout(canvas_stage)
        canvas_stage_layout.setContentsMargins(0, 0, 0, 0)
        canvas_stage_layout.setSpacing(8)
        canvas_stage_layout.addWidget(curation_bar)
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

        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(0)
        controls_layout.addWidget(self.tabs, 1)

        controls_widget = QWidget()
        controls_widget.setObjectName('controlsPanel')
        controls_widget.setLayout(controls_layout)
        controls_widget.setMinimumWidth(380)
        controls_widget.setMaximumWidth(450)

        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setObjectName('mainSplitter')
        self.main_splitter.addWidget(controls_widget)
        self.main_splitter.addWidget(right_pane)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setSizes([400, 860])

        stage_header = QFrame()
        stage_header.setObjectName('stageHeader')
        stage_layout = QHBoxLayout(stage_header)
        stage_layout.setContentsMargins(12, 8, 12, 8)
        stage_layout.setSpacing(14)

        stage_layout.addWidget(self.state_label, 1)
        stage_layout.addWidget(self.roi_label)
        stage_layout.addWidget(self.model_label)
        stage_layout.addSpacing(8)
        stage_layout.addWidget(self.roi_overlay_check)
        stage_layout.addWidget(self.dark_mode_check)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(10)
        main_layout.addWidget(stage_header)
        main_layout.addWidget(self.main_splitter, 1)

        container = QWidget()
        container.setObjectName('centralWidget')
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def _layout_mser_tab(self):
        layout = QVBoxLayout(self.mser_tab_content)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)
        self._layout_mser_section(layout)
        layout.addStretch(1)

    def _layout_prediction_tab(self):
        layout = QVBoxLayout(self.predict_tab_content)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)
        self._layout_prediction_section(layout)
        layout.addStretch(1)

    def _layout_training_tab(self):
        layout = QVBoxLayout(self.training_tab_content)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 12, 12, 12)
        self._layout_training_section(layout)
        layout.addStretch(1)

    def _layout_mser_section(self, layout):
        io_buttons = QGridLayout()
        io_buttons.setHorizontalSpacing(6)
        io_buttons.setVerticalSpacing(6)
        io_buttons.addWidget(self.segment_load_image_button, 0, 0)
        io_buttons.addWidget(self.segment_load_roi_button, 0, 1)
        layout.addLayout(io_buttons)

        specs_by_name = {spec['name']: spec for spec in PARAMETER_SPECS}
        main_specs = [specs_by_name[name] for name in CORE_SEGMENT_PARAMETERS]
        param_form = QFormLayout()
        param_form.setHorizontalSpacing(12)
        param_form.setVerticalSpacing(6)
        for spec in main_specs:
            tooltip = PARAMETER_TOOLTIPS.get(spec['name'], '')
            widget = self.segment_param_widgets[spec['name']]
            widget.setToolTip(tooltip)
            param_form.addRow(self.make_form_label(spec['name'], tooltip, widget), widget)
        layout.addLayout(param_form)

        buttons = QGridLayout()
        buttons.setHorizontalSpacing(6)
        buttons.setVerticalSpacing(6)
        buttons.addWidget(self.segment_button, 0, 0)
        buttons.addWidget(self.reset_segment_button, 0, 1)
        buttons.addWidget(self.advanced_segment_button, 1, 0, 1, 2)
        buttons.addWidget(self.fix_selected_button, 2, 0)
        buttons.addWidget(self.unfix_selected_button, 2, 1)
        buttons.addWidget(self.clear_fixed_button, 3, 0)
        buttons.addWidget(self.clear_unfixed_button, 3, 1)
        layout.addLayout(buttons)

    def _layout_training_section(self, layout):
        layout.addWidget(self.make_form_label(
            'labelled sessions',
            'folder containing processed sessions for training',
            self.source_root_line,
            ))
        layout.addWidget(self._path_row(self.source_root_line, self.browse_source_root))
        form = QFormLayout()
        form.setVerticalSpacing(8)
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

        buttons = QGridLayout()
        buttons.setHorizontalSpacing(6)
        buttons.setVerticalSpacing(6)
        buttons.addWidget(self.build_manifest_button, 0, 0)
        buttons.addWidget(self.train_model_button, 0, 1)
        buttons.addWidget(self.training_diagnostics_button, 1, 0)
        buttons.addWidget(self.training_advanced_button, 1, 1)
        buttons.addWidget(self.stop_process_button, 2, 0, 1, 2)
        layout.addLayout(buttons)

    def _layout_prediction_section(self, layout):
        layout.addWidget(self.make_form_label(
            'channel-2 image',
            'channel-2 reference image used for ROI prediction',
            self.image_line,
            ))
        layout.addWidget(self._path_row(self.image_line, self.browse_image))
        layout.addWidget(self.make_form_label(
            'trained model',
            'saved trained model file',
            self.checkpoint_line,
            ))
        layout.addWidget(self._path_row(self.checkpoint_line, self.browse_checkpoint))
        form = QFormLayout()
        form.setVerticalSpacing(8)
        form.addRow(
            self.make_form_label(
                'strictness',
                'higher values keep only stronger model detections',
                self.threshold_spin,
                ),
            self.threshold_spin,
            )
        form.addRow(
            self.make_form_label(
                'minimum ROI size',
                'smallest ROI size to keep',
                self.min_size_spin,
                ),
            self.min_size_spin,
            )
        layout.addLayout(form)

        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        buttons.addWidget(self.predict_button, 1)
        buttons.addWidget(self.apply_settings_button)
        layout.addLayout(buttons)

        options = QHBoxLayout()
        options.addWidget(self.show_probability_check)
        options.addStretch(1)
        layout.addLayout(options)


    def show_segment_advanced_popup(self):
        if self.segment_advanced_popup is None:
            self.segment_advanced_popup = QFrame(self, Qt.Popup)
            self.segment_advanced_popup.setObjectName('advancedPopup')
            layout = QFormLayout(self.segment_advanced_popup)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setHorizontalSpacing(10)
            layout.setVerticalSpacing(7)
            for spec in PARAMETER_SPECS:
                if spec['name'] in CORE_SEGMENT_PARAMETERS:
                    continue
                tooltip = PARAMETER_TOOLTIPS.get(spec['name'], '')
                widget = self.segment_param_widgets[spec['name']]
                widget.setToolTip(tooltip)
                layout.addRow(self.make_form_label(spec['name'], tooltip, widget), widget)

        pos = self.advanced_segment_button.mapToGlobal(self.advanced_segment_button.rect().bottomLeft())
        self.segment_advanced_popup.move(pos)
        self.segment_advanced_popup.show()
        self.segment_advanced_popup.raise_()

    def show_training_diagnostics_popup(self):
        if self.training_diagnostics_popup is None:
            self.training_diagnostics_popup = QFrame(self, Qt.Popup)
            self.training_diagnostics_popup.setObjectName('advancedPopup')
            layout = QGridLayout(self.training_diagnostics_popup)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setHorizontalSpacing(6)
            layout.setVerticalSpacing(6)
            layout.addWidget(self.inspect_manifest_button, 0, 0)
            layout.addWidget(self.evaluate_model_button, 0, 1)
            layout.addWidget(self.preview_training_button, 1, 0)
            layout.addWidget(self.preview_predictions_button, 1, 1)

        pos = self.training_diagnostics_button.mapToGlobal(self.training_diagnostics_button.rect().bottomLeft())
        self.training_diagnostics_popup.move(pos)
        self.training_diagnostics_popup.show()
        self.training_diagnostics_popup.raise_()

    def show_training_advanced_popup(self):
        if self.training_advanced_popup is None:
            self.training_advanced_popup = QFrame(self, Qt.Popup)
            self.training_advanced_popup.setObjectName('advancedPopup')
            layout = QFormLayout(self.training_advanced_popup)
            layout.setContentsMargins(12, 10, 12, 10)
            layout.setHorizontalSpacing(10)
            layout.setVerticalSpacing(7)
            layout.addRow(
                self.make_form_label(
                    'dataset table',
                    'CSV index of labelled images and ROI dicts',
                    self.manifest_line,
                    ),
                self._path_row(self.manifest_line, self.browse_manifest_out),
                )
            layout.addRow(
                self.make_form_label(
                    'training recipe',
                    'YAML settings used for model training',
                    self.config_line,
                    ),
                self._path_row(self.config_line, self.browse_config),
                )
            layout.addRow(
                self.make_form_label(
                    'validation split',
                    'fraction used for tuning during training',
                    self.val_fraction_spin,
                    ),
                self.val_fraction_spin,
                )
            layout.addRow(
                self.make_form_label(
                    'test split',
                    'held-out fraction used for scoring',
                    self.test_fraction_spin,
                    ),
                self.test_fraction_spin,
                )
            layout.addRow(
                self.make_form_label(
                    'compute device',
                    'this setting also applies to prediction and scoring',
                    self.device_combo,
                    ),
                self.device_combo,
                )

        pos = self.training_advanced_button.mapToGlobal(self.training_advanced_button.rect().bottomLeft())
        self.training_advanced_popup.move(pos)
        self.training_advanced_popup.show()
        self.training_advanced_popup.raise_()

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
            ('Ctrl+F', self.fix_selected),
            ('Ctrl+Shift+F', self.clear_fixed),
            ('Ctrl+A', self.select_all),
            ('Ctrl+I', self.invert_selection),
            ('Escape', self.clear_selection),
            ('Delete', self.delete_selected),
            ('Ctrl+Z', self.undo),
            ('Backspace', self.undo),
            ]
        for sequence, slot in shortcuts:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(slot)

    def _theme(self):
        if self.dark_mode:
            return {
                'window': '#151b17',
                'surface': '#1e2721',
                'surface_alt': '#242f28',
                'surface_strong': '#34433a',
                'surface_hover': '#2c3931',
                'border': '#3c4b42',
                'border_strong': '#66786d',
                'text': '#edf3ef',
                'muted': '#b7c4bb',
                'primary': '#79b28b',
                'primary_hover': '#92c5a1',
                'primary_text': '#102017',
                'danger_bg': '#3b2528',
                'danger_hover': '#503034',
                'danger_border': '#83545b',
                'danger_text': '#f0c6cb',
                'selection': '#365c42',
                'canvas': '#182019',
                }

        return {
            'window': '#f3f6f4',
            'surface': '#ffffff',
            'surface_alt': '#f7faf8',
            'surface_strong': '#dbe7df',
            'surface_hover': '#edf5f0',
            'border': '#cbd8d0',
            'border_strong': '#8aa093',
            'text': '#17211b',
            'muted': '#516157',
            'primary': '#2f6f4e',
            'primary_hover': '#25593f',
            'primary_text': '#ffffff',
            'danger_bg': '#f6e9eb',
            'danger_hover': '#efd8dc',
            'danger_border': '#c88e98',
            'danger_text': '#8f3544',
            'selection': '#c9ead2',
            'canvas': '#f7f9f8',
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
        self.setPalette(palette)

        self.setStyleSheet(
            f'''
            QMainWindow, QWidget#centralWidget {{
                background: {theme['window']};
            }}
            QFrame#canvasFrame {{
                border: 1px solid {theme['border']};
                border-radius: 7px;
                background: {theme['canvas']};
            }}
            QFrame#curationBar, QFrame#activityFrame, QFrame#advancedPopup {{
                border: 1px solid {theme['border']};
                border-radius: 7px;
                background: {theme['surface']};
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
                border-radius: 4px;
            }}
            QSplitter::handle:horizontal {{
                width: 8px;
            }}
            QSplitter::handle:vertical {{
                height: 8px;
            }}
            QWidget#controlsPanel, QWidget#mainPane, QWidget#canvasStage {{
                background: transparent;
            }}
            QScrollArea, QWidget#tabContent {{
                border: none;
                background: {theme['surface']};
            }}
            QTabWidget::pane {{
                border: 1px solid {theme['border']};
                border-radius: 7px;
                background: {theme['surface']};
            }}
            QTabBar::tab {{
                background: {theme['surface_alt']};
                border: 1px solid {theme['border']};
                border-bottom: none;
                padding: 6px 9px;
                margin-right: 3px;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                color: {theme['muted']};
                font-weight: 600;
                min-width: 58px;
            }}
            QTabBar::tab:hover {{
                background: {theme['surface_hover']};
                color: {theme['text']};
                border-color: {theme['border_strong']};
            }}
            QTabBar::tab:selected {{
                background: {theme['surface']};
                color: {theme['primary']};
                border-color: {theme['border_strong']};
                border-top: 2px solid {theme['primary']};
            }}
            QLabel {{
                color: {theme['text']};
            }}
            QLabel#stateSummary {{
                color: {theme['text']};
                font-weight: 700;
                font-size: 10pt;
                padding: 0;
            }}
            QLabel#panelValue {{
                color: {theme['muted']};
                font-weight: 600;
                font-family: 'mononoki', Consolas, 'Courier New', monospace;
                padding: 0;
            }}
            QLineEdit, QDoubleSpinBox, QSpinBox, QComboBox {{
                border: 1px solid {theme['border']};
                border-radius: 5px;
                padding: 4px 7px;
                background: {theme['surface_alt']};
                color: {theme['text']};
                selection-background-color: {theme['selection']};
                min-height: 24px;
            }}
            QLineEdit[field='path'], QDoubleSpinBox, QSpinBox {{
                font-family: 'mononoki', Consolas, 'Courier New', monospace;
            }}
            QLineEdit:hover, QDoubleSpinBox:hover, QSpinBox:hover, QComboBox:hover {{
                border-color: {theme['border_strong']};
                background: {theme['surface']};
            }}
            QLineEdit:focus, QDoubleSpinBox:focus, QSpinBox:focus, QComboBox:focus {{
                border-color: {theme['primary']};
                background: {theme['surface']};
            }}
            QCheckBox {{
                spacing: 6px;
                color: {theme['text']};
                font-weight: 500;
            }}
            QCheckBox:hover {{
                color: {theme['primary']};
            }}
            QCheckBox:focus {{
                color: {theme['primary']};
            }}
            QPushButton {{
                border: 1px solid {theme['border']};
                border-radius: 5px;
                padding: 4px 8px;
                background: {theme['surface']};
                color: {theme['text']};
                min-height: 24px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: {theme['surface_hover']};
                border-color: {theme['border_strong']};
            }}
            QPushButton:pressed {{
                background: {theme['surface_strong']};
            }}
            QPushButton:focus {{
                border: 2px solid {theme['primary']};
                padding: 3px 8px;
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
            QPushButton[role='danger'] {{
                background: {theme['danger_bg']};
                border-color: {theme['danger_border']};
                color: {theme['danger_text']};
            }}
            QPushButton[role='danger']:hover {{
                background: {theme['danger_hover']};
                border-color: {theme['danger_text']};
            }}
            QPushButton[role='dangerQuiet'] {{
                background: transparent;
                border-color: transparent;
                color: {theme['danger_text']};
            }}
            QPushButton[role='dangerQuiet']:hover {{
                background: {theme['danger_bg']};
                border-color: {theme['danger_border']};
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
            QPushButton:disabled {{
                background: {theme['surface_alt']};
                border-color: {theme['border']};
                color: {theme['muted']};
            }}
            QPlainTextEdit#logBox {{
                border: none;
                border-radius: 4px;
                background: {theme['surface_alt']};
                color: {theme['text']};
                font-family: 'mononoki', Consolas, 'Courier New', monospace;
                font-size: 9pt;
                padding: 7px;
                selection-background-color: {theme['selection']};
            }}
            QToolTip {{
                border: 1px solid {theme['border_strong']};
                border-radius: 6px;
                padding: 5px 7px;
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

    def selected_device(self):
        return self.device_combo.currentData()

    def compute_device_changed(self):
        device_name = self.selected_device()
        if self.predictor is not None and device_name != self.loaded_device_choice:
            self.predictor = None
            self.loaded_device_choice = None
            self.model_label.setText('model: not loaded (compute device changed)')
        self.invalidate_probability()
        self.refresh_status('compute device changed')

    def prediction_inputs_changed(self, _text=None):
        self.invalidate_probability()
        self.refresh_status()

    def invalidate_probability(self):
        if self.probability is None:
            return
        self.probability = None
        if self.show_probability_check.isChecked():
            self.plot_image(preserve_view=True)


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
        self.pending_checkpoint_path = default_output_root() / 'runs' / run_name / 'best.pt'
        args = ['--config', str(config_path)]
        self.start_process('train', 'train_unet', args)

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

        # Leave the baseline YAML alone while trying controls here; save this recipe beside the run.
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
            return

        self.current_process_name = process_name
        self.process = QProcess(self)
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        self.process.setWorkingDirectory(str(WORKSPACE_ROOT))
        self.process.setProgram(sys.executable)
        command_args = ['-m', f'fibre_sight.{module_name}'] + args
        self.process.setArguments(command_args)
        self.process.readyReadStandardOutput.connect(self.read_process_stdout)
        self.process.readyReadStandardError.connect(self.read_process_stderr)
        self.process.finished.connect(self.process_finished)

        command_text = ' '.join([sys.executable] + command_args)
        self.print_log(f'\n$ {command_text}')
        self.process.start()
        # The log keeps the command and output; the status bar only confirms that it started.
        self.refresh_status(f'{process_name} started')

    def stop_process(self):
        if self.process is None or self.process.state() == QProcess.NotRunning:
            self.print_log('no process is running')
            return

        self.process.kill()
        self.print_log('process stopped')

    def read_process_stdout(self):
        text = bytes(self.process.readAllStandardOutput()).decode(errors='replace')
        self.print_log(text, end='')

    def read_process_stderr(self):
        text = bytes(self.process.readAllStandardError()).decode(errors='replace')
        self.print_log(text, end='')

    def process_finished(self, exit_code, exit_status):
        process_name = self.current_process_name or 'process'
        self.print_log(f'\n{process_name} finished: exit code {exit_code}')
        if (
            self.current_process_name == 'train' and
            exit_code == 0 and
            exit_status == QProcess.NormalExit and
            self.pending_checkpoint_path is not None and
            self.pending_checkpoint_path.exists()
        ):
            self.checkpoint_line.setText(str(self.pending_checkpoint_path))
            self.last_saved_model_path = self.pending_checkpoint_path
            self.print_log(f'trained model ready: {self.pending_checkpoint_path}')

        self.current_process_name = None
        self.pending_checkpoint_path = None
        self.process = None
        self.refresh_status(f'{process_name} finished')

    #%% model prediction
    def choose_channel_image(self):
        if self.browse_image():
            self.load_channel_image()

    def image_matches(self, path):
        if self.ref_image is None or self.image_path is None:
            return False
        return self.image_path.resolve() == Path(path).resolve()

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
        self.probability = None

        default_roi = self.default_roi_path()
        if default_roi.exists():
            self.print_log(f'found existing ROI dict: {default_roi}')
            self.print_log('load an ROI dict to review or continue from it')
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
            self.loaded_device_choice = None
            self.refresh_status('model load failed')
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.loaded_device_choice = self.selected_device()
        self.model_label.setText(
            f'model: {self.predictor.checkpoint_path.name} | {self.predictor.device}'
            )
        self.last_saved_model_path = self.predictor.checkpoint_path
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
            predictor_path.resolve() == checkpoint_path.resolve() and
            self.loaded_device_choice == self.selected_device()
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

        self.plot_image()
        self.canvas.reset_view()
        self.print_log(
            f'predicted {len(self.roi_dict)} ROIs '
            f'(strictness {prediction.threshold:.2f}, min size {prediction.min_size})'
            )
        self.refresh_status('prediction complete')

    def apply_probability_threshold(self):
        if self.probability is None:
            return

        # reuse the confidence map here; changing strictness should not rerun
        # the model while I am deciding which faint fibres to keep
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
        self.refresh_status('prediction settings applied')

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
        checkpoint = self.checkpoint_line.text().strip()
        if checkpoint:
            return checkpoint
        if self.predictor is not None:
            return str(self.predictor.checkpoint_path)
        if self.last_saved_model_path is not None:
            return str(self.last_saved_model_path)
        return ''

    #%% roi io
    def load_roi_file(self):
        if self.ref_image is None:
            self.print_log('please load a channel-2 image first')
            return

        path, _ = QFileDialog.getOpenFileName(
            self,
            'load ROI dict',
            str(self.image_path.parent if self.image_path else WORKSPACE_ROOT),
            'NumPy dict (*.npy)',
            )
        if not path:
            return

        try:
            roi_dict = load_roi_dict(path)
            labelled, _, _ = roi_dict_to_label(roi_dict, self.ref_image.shape)
        except Exception as exc:
            self.print_log(f'failed to load ROI dict: {exc}')
            return

        self.push_undo_state()
        self.roi_dict = labels_to_roi_dict(labelled)
        self.labelled = labelled
        self.selected.clear()
        self.fixed_ids.clear()
        self.probability = None
        self.plot_image()
        self.refresh_status('ROI dict loaded')

    def save_roi_file(self):
        if self.ref_image is None:
            self.print_log('please load a channel-2 image first')
            return

        self.update_roi_dict()
        out_path = self.default_roi_path()
        save_roi_dict(self.roi_dict, out_path)
        self.print_log(f'saved ROI dict to {out_path}')
        self.refresh_status('ROI dict saved')

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
        if self.ref_image is None:
            self.print_log('please load a channel-2 image first')
            return

        if self.labelled is not None:
            self.push_undo_state()

        # MSER still helps before a model exists and on images that need hand repair
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
        self.probability = None
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

        if modifiers & (Qt.ControlModifier | Qt.ShiftModifier):
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
        self.fig.set_facecolor(theme['canvas'])
        self.ax.clear()
        self.ax.set_facecolor(theme['canvas'])
        self.ax.axis('off')

        if self.ref_image is None:
            self.canvas.setCursor(Qt.PointingHandCursor)
            self.canvas.setToolTip('click to open a channel-2 image')
            self.ax.text(
                0.5,
                0.53,
                'open a channel-2 image to start',
                transform=self.ax.transAxes,
                ha='center',
                va='center',
                color=theme['muted'],
                fontsize=13,
                fontweight='bold',
                fontfamily=MONONOKI_FONT_FAMILY,
                )
            self.ax.text(
                0.5,
                0.47,
                'click anywhere on this canvas to browse',
                transform=self.ax.transAxes,
                ha='center',
                va='center',
                color=theme['muted'],
                fontsize=9,
                alpha=0.8,
                fontfamily=MONONOKI_FONT_FAMILY,
                )
            self.canvas.draw_idle()
            return

        self.canvas.setCursor(Qt.CrossCursor)
        self.canvas.setToolTip(
            'click an ROI to select; Shift-click adds; right-drag pans; scroll zooms'
            )
        base = normalise_for_display(self.ref_image)
        self.ax.imshow(base, cmap='gray', interpolation='nearest')

        if self.show_probability_check.isChecked() and self.probability is not None:
            self.plot_probability()
        if self.roi_overlay_check.isChecked():
            self.plot_roi_overlay()

        if preserve_view and xlim is not None:
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)

        self.canvas.draw_idle()
        self.refresh_status()

    def plot_probability(self):
        probability = np.asarray(self.probability, dtype=np.float32)
        rgba = matplotlib.colormaps['magma'](np.clip(probability, 0, 1))
        rgba[..., 3] = np.clip(probability, 0, 1) * 0.35
        self.ax.imshow(rgba, interpolation='nearest')

    def plot_roi_overlay(self):
        if self.labelled is None:
            return

        ids = [int(label_id) for label_id in np.unique(self.labelled) if label_id > 0]
        if not ids:
            return

        overlay = np.zeros((*self.labelled.shape, 4), dtype=np.float32)
        colours = generate_distinct_colours(len(ids))
        for colour, roi_id in zip(colours, ids):
            mask = self.labelled == roi_id
            overlay[mask, :3] = colour
            if roi_id in self.selected:
                overlay[mask, 3] = 0.82
            elif roi_id in self.fixed_ids:
                overlay[mask, 3] = 0.66
            else:
                overlay[mask, 3] = 0.46

        self.ax.imshow(overlay, interpolation='nearest')
        selected_mask = np.isin(self.labelled, list(self.selected))
        fixed_mask = np.isin(self.labelled, list(self.fixed_ids - self.selected))
        if np.any(selected_mask) and np.any(~selected_mask):
            self.ax.contour(
                selected_mask,
                levels=[0.5],
                colors=['#79ddff'],
                linewidths=1.5,
                linestyles='solid',
                )
        if np.any(fixed_mask) and np.any(~fixed_mask):
            self.ax.contour(
                fixed_mask,
                levels=[0.5],
                colors=['#ffd166'],
                linewidths=1.4,
                linestyles='dashed',
                )

    def get_current_view(self):
        if not self.ax.images:
            return None, None
        return self.ax.get_xlim(), self.ax.get_ylim()

    def reset_view(self):
        self.canvas.reset_view()
        self.refresh_status('view reset')

    #%% state helpers
    def update_roi_dict(self):
        if self.labelled is None:
            self.roi_dict = {}
            self.fixed_ids.clear()
            return
        self.roi_dict = labels_to_roi_dict(self.labelled)
        self.fixed_ids = {roi_id for roi_id in self.fixed_ids if roi_id in self.roi_dict}

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
        self.state_label.setText(f'image: {image_name}')
        self.roi_label.setText(
            f'{roi_count} ROIs | {selected_count} selected | {fixed_count} fixed'
            )
        self.update_workflow_state()
        if message:
            # Keep routine confirmation off the canvas while I am selecting and fixing ROIs.
            self.statusBar().showMessage(message, 4000)

    def update_workflow_state(self):
        image_ready = self.ref_image is not None
        image_text = self.image_line.text().strip()
        image_path_ready = bool(image_text) and Path(image_text).exists()
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
        selected_fixed = bool(self.selected.intersection(self.fixed_ids))
        selected_unfixed = bool(self.selected.difference(self.fixed_ids))
        has_fixed = bool(self.fixed_ids)

        self.build_manifest_button.setEnabled(source_ready and not process_running)
        self.inspect_manifest_button.setEnabled(manifest_ready and not process_running)
        self.train_model_button.setEnabled(source_ready and config_ready and bool(self.run_name_line.text().strip()) and not process_running)
        self.evaluate_model_button.setEnabled(manifest_ready and model_ready and not process_running)
        self.preview_training_button.setEnabled(manifest_ready and not process_running)
        self.preview_predictions_button.setEnabled(manifest_ready and model_ready and not process_running)
        self.stop_process_button.setEnabled(process_running)
        self.stop_process_button.setVisible(process_running)
        self.device_combo.setEnabled(not process_running)

        self.predict_button.setEnabled(
            image_path_ready and checkpoint_ready and not process_running
            )
        probability_ready = self.probability is not None
        self.apply_settings_button.setEnabled(probability_ready and not process_running)
        self.apply_settings_button.setVisible(probability_ready)
        self.show_probability_check.setVisible(probability_ready)

        self.segment_button.setEnabled(image_ready)
        self.segment_load_roi_button.setEnabled(image_ready)
        self.fix_selected_button.setEnabled(selected_unfixed)
        self.fix_selected_button.setVisible(selected_unfixed)
        self.unfix_selected_button.setEnabled(selected_fixed)
        self.unfix_selected_button.setVisible(selected_fixed)
        self.clear_fixed_button.setEnabled(has_fixed)
        self.clear_fixed_button.setVisible(has_fixed)
        self.clear_unfixed_button.setEnabled(has_rois and has_fixed)
        self.clear_unfixed_button.setVisible(has_rois and has_fixed)

        self.curate_buttons['select_all'].setEnabled(has_rois)
        self.curate_buttons['delete'].setEnabled(bool(self.selected))
        self.curate_buttons['merge'].setEnabled(len(self.selected) >= 2)
        self.curate_buttons['undo'].setEnabled(bool(self.undo_stack))
        self.curate_buttons['reset_view'].setEnabled(image_ready)
        self.curate_buttons['save_roi'].setEnabled(image_ready and has_rois)

    def print_log(self, text, end='\n'):
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
        app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(load_gui_font())
    app.setWindowIcon(QIcon(str(APP_ICON_PATH)))
    window = FibreSightWorkbench()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
