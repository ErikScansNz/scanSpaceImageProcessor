#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# To install dependencies, run:
#   pip install colour-checker-detection psutil pillow

import os
import time
import webbrowser
from functools import partial

import cv2
import numpy as np
import colour
import psutil
import tempfile
from OpenImageIO import ColorConfig
from PIL import Image
from colour.models import RGB_COLOURSPACES
from colour_checker_detection import (
    detect_colour_checkers_segmentation, detect_colour_checkers_inference)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QListWidgetItem,
    QGraphicsScene, QGraphicsPixmapItem, QLabel, QRubberBand, QWidget, QGraphicsView, QDialog, QProgressDialog
)
from PySide6.QtCore import (
    QRunnable, QThreadPool, Signal, QObject, QTimer, Qt, QSettings, QEvent, QRect, Slot, QPropertyAnimation, QEasingCurve, QThread, QMetaObject, Property
)
from PySide6.QtGui import QColor, QPixmap, QImage, QPainter, QIcon, QCursor, QTransform, QBrush, QKeySequence, QAction

from interface.scanspaceImageProcessor_UI import Ui_MainWindow
from interface.settings_UI import Ui_ImageProcessorSettings
from resource_path import get_main_icon, get_icon_path, list_available_resources
from ImageProcessor.imageProcessorWorker import ImageCorrectionWorker
from ImageProcessor.chartTools import ChartTools
from ImageProcessor.imageLoader import ImageLoader, RawLoadWorker
from ImageProcessor.fileNamingSchema import FileNamingSchema
from ImageProcessor.serverClient import ServerConnectionError, ServerAPIError
from ImageProcessor.editingTools import apply_all_adjustments
from ImageProcessor.themes import theme_manager
from ImageProcessor.function_tools import (
    export_current_project, send_project_to_server, update_server_status_label,
    _show_submission_confirmation_dialog, _build_submission_details, 
    _on_submission_progress, _on_submission_success, _on_submission_error, 
    _on_submission_finished, _build_project_data, _sanitize_project_data,
    _debug_find_unserializable_objects, ProjectSubmissionWorker
)

# Logging system
from enum import IntEnum
import datetime

class LogLevel(IntEnum):
    """Log levels for filtering log output by importance."""
    CRITICAL = 50  # Critical errors that may cause application failure
    ERROR = 40     # Error conditions
    WARNING = 30   # Warning messages
    INFO = 20      # General information messages
    DEBUG = 10     # Debug information for troubleshooting
    TRACE = 5      # Detailed trace information

# Code to get taskbar icon visible (Windows only)
import platform
if platform.system() == 'Windows':
    import ctypes
    scanSpaceImageProcessor = u'mycompany.myproduct.subproduct.version' # arbitrary string to trick windows
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(scanSpaceImageProcessor)

# supported file formats
RAW_EXTENSIONS = ('.nef', '.cr2', '.cr3', '.dng', '.arw', '.raw')
OUTPUT_FORMATS = ('.jpg', '.png', '.tiff', '.exr')

reference = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
illuminant_XYZ = colour.xy_to_XYZ(reference.illuminant)

reference_swatches = colour.XYZ_to_RGB(
    colour.xyY_to_XYZ(list(reference.data.values())),
    RGB_COLOURSPACES['sRGB'],
    illuminant=reference.illuminant,           # a 2-tuple (x, y)
    chromatic_adaptation_transform="CAT02",
    apply_cctf_encoding=False,
)

class WorkerSignals(QObject):
    log = Signal(str)
    preview = Signal(object)
    status = Signal(str, str, float, str)

class ClickableLabel(QLabel):
    clicked = Signal(int)  # Pass the index in the imagesListWidget

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index

    def mouseReleaseEvent(self, event):
        self.clicked.emit(self.index)
        super().mouseReleaseEvent(event)

# Helper classes for UI elements moved to ImageProcessor.chartTools

class SwatchPreviewSignals(QObject):
    """Worker signals for previewing manual swatch correction."""
    finished = Signal(np.ndarray)       # Emits the corrected uint8 array
    error    = Signal(str)              # Emits an error message on failure

class SwatchPreviewWorker(QRunnable):
    """
    Runs colour correction + CCTF encoding in a background thread.
    Emits the final uint8 image array when done.
    """
    def __init__(self, img_fp, swatch_colours, reference_swatches):
        super().__init__()
        self.img_fp            = img_fp
        self.swatch_colours    = swatch_colours
        self.reference_swatches= reference_swatches
        self.signals           = SwatchPreviewSignals()

    def run(self):
        try:

            # 1) Colour‐correct
            corrected = colour.colour_correction(
                self.img_fp,
                self.swatch_colours,
                self.reference_swatches
            )
            
            corrected = np.clip(corrected, 0, 1)

            # 2) Apply CCTF & convert to uint8
            corrected_uint8 = np.uint8(255 * colour.cctf_encoding(corrected))

            # 3) Emit the raw array back to the main thread
            self.signals.finished.emit(corrected_uint8)

        except Exception as e:
            self.signals.error.emit(str(e))


class ImageProcessorSettingsDialog(QDialog):
    """
    Settings dialog for the Image Processor application.
    
    This dialog provides a tabbed interface for configuring various
    application settings including general preferences, processing
    parameters, and advanced options. Settings are stored in an INI file.
    """
    
    def __init__(self, parent=None):
        """
        Initialize the settings dialog.
        
        Args:
            parent: Parent widget (typically the main window)
        """
        super().__init__(parent)
        self.ui = Ui_ImageProcessorSettings()
        self.ui.setupUi(self)
        self.main_window = parent  # Store reference to main window
        
        # Set window properties
        self.setWindowTitle("Image Processor Settings")
        self.setModal(True)  # Make dialog modal
        
        # Connect dialog buttons if they exist
        if hasattr(self.ui, 'buttonBox'):
            self.ui.buttonBox.accepted.connect(self.accept)
            self.ui.buttonBox.rejected.connect(self.reject)
        
        # Setup log level combo box options
        if hasattr(self.ui, 'logDebugDepthCombobox'):
            self.ui.logDebugDepthCombobox.addItems([
                "CRITICAL (50)",
                "ERROR (40)", 
                "WARNING (30)",
                "INFO (20)",
                "DEBUG (10)",
                "TRACE (5)"
            ])
            # Connect to update handler
            self.ui.logDebugDepthCombobox.currentIndexChanged.connect(self._on_log_level_changed)
        
        # Setup combo box options
        self._setup_combo_boxes()
        
        # Connect browse button for default color chart
        self.ui.browseForDefaultColorChartPushButton.clicked.connect(self.browse_for_default_color_chart)
        
        # Setup export schema validation and preview
        self._setup_export_schema_validation()
    
    def _on_log_level_changed(self, index):
        """Handle log level combo box changes."""
        if not hasattr(self.ui, 'logDebugDepthCombobox'):
            return
        
        # Map combo box index to LogLevel values
        log_levels = [
            LogLevel.CRITICAL,  # 0: CRITICAL (50)
            LogLevel.ERROR,     # 1: ERROR (40)
            LogLevel.WARNING,   # 2: WARNING (30) 
            LogLevel.INFO,      # 3: INFO (20)
            LogLevel.DEBUG,     # 4: DEBUG (10)
            LogLevel.TRACE      # 5: TRACE (5)
        ]
        
        if 0 <= index < len(log_levels) and self.main_window:
            new_level = log_levels[index]
            self.main_window.current_log_level = new_level
            self.main_window.log_debug(f"Log level changed to {new_level.name} ({new_level.value})")
    
    def _setup_combo_boxes(self):
        """Setup combo box options with default values."""
        # Setup export format combo box
        self.ui.defaultExportFormatComboBox.addItems(['.jpg', '.png', '.tiff', '.exr'])
        self.ui.defaultExportFormatComboBox.setCurrentText('.jpg')

        config = ColorConfig()
        colour_space_names = config.getColorSpaceNames()  # returns List[str]
        for cs in colour_space_names:
            self.ui.defaultColorspaceComboBox.addItem(cs)
        self.ui.defaultColorspaceComboBox.setCurrentIndex(1)
    
    def load_settings(self):
        """
        Load settings from INI file and populate the UI controls.
        This method should be called before showing the dialog.
        """
        settings = QSettings('ScanSpace', 'ImageProcessor')
        
        # Load general settings
        display_log = settings.value('display_log', True, type=bool)
        thread_count = settings.value('thread_count', 4, type=int)
        export_format = settings.value('export_format', '.jpg', type=str)
        bit_depth_16 = settings.value('bit_depth_16_default', False, type=bool)
        default_colorspace = settings.value('default_colorspace', 'sRGB', type=str)
        use_precalc_charts = settings.value('use_precalculated_charts', False, type=bool)
        chart_folder_path = settings.value('chart_folder_path', '', type=str)
        correct_thumbnails = settings.value('correct_thumbnails', False, type=bool)
        log_level = settings.value('log_level', LogLevel.INFO, type=int)
        
        # Load import/export settings
        look_in_subfolders = settings.value('look_in_subfolders', False, type=bool)
        group_by_subfolder = settings.value('group_by_subfolder', False, type=bool)
        group_by_prefix = settings.value('group_by_prefix', False, type=bool)
        prefix_string = settings.value('prefix_string', '', type=str)
        ignore_formats = settings.value('ignore_formats', False, type=bool)
        ignore_string = settings.value('ignore_string', '', type=str)
        
        # Load export schema settings
        export_schema = settings.value('export_schema', '[r]/[o][n4][e]', type=str)
        use_export_schema = settings.value('use_export_schema', False, type=bool)
        use_import_rules = settings.value('use_import_rules', False, type=bool)
        
        # Load server/client settings
        enable_server = settings.value('enable_server', False, type=bool)
        # Load network settings
        host_server_address = settings.value('host_server_address', '', type=str)

        # Theme Settings
        dark_mode = settings.value('dark_mode', False, type=bool)
        
        # Apply settings to UI controls
        self.ui.enableDarkThemeCheckbox.setChecked(dark_mode)
        self.ui.displayLogCheckBox.setChecked(display_log)
        self.ui.defaultThreadCountSpinbox.setValue(thread_count)
        self.ui.colorCorrectThumbnailsCheckbox.setChecked(correct_thumbnails)
        
        # Set log level combo box
        if hasattr(self.ui, 'logDebugDepthCombobox'):
            # Map LogLevel value to combo box index
            log_level_map = {
                LogLevel.CRITICAL: 0,  # CRITICAL (50)
                LogLevel.ERROR: 1,     # ERROR (40)
                LogLevel.WARNING: 2,   # WARNING (30)
                LogLevel.INFO: 3,      # INFO (20)
                LogLevel.DEBUG: 4,     # DEBUG (10)
                LogLevel.TRACE: 5      # TRACE (5)
            }
            combo_index = log_level_map.get(log_level, 3)  # Default to INFO
            self.ui.logDebugDepthCombobox.setCurrentIndex(combo_index)
        
        # Set export format combo box
        index = self.ui.defaultExportFormatComboBox.findText(export_format)
        if index >= 0:
            self.ui.defaultExportFormatComboBox.setCurrentIndex(index)
        
        self.ui.bitDepth16EnableCheckbox.setChecked(bit_depth_16)
        
        # Set colorspace combo box
        index = self.ui.defaultColorspaceComboBox.findText(default_colorspace)
        if index >= 0:
            self.ui.defaultColorspaceComboBox.setCurrentIndex(index)
        
        # Set default color chart settings
        self.ui.usePrecalculatedChartsCheckBox.setChecked(use_precalc_charts)
        self.ui.chartFolderPathLineEdit.setText(chart_folder_path)

        # Set import/export settings
        self.ui.lookForImagesInSubfolderCheckbox.setChecked(look_in_subfolders)
        self.ui.groupImagesBySubfolderCheckbox.setChecked(group_by_subfolder)
        self.ui.groupImagesByPrefixCheckbox.setChecked(group_by_prefix)
        self.ui.prefixGroupingLineEdit.setText(prefix_string)
        self.ui.ignoreFormatsCheckbox.setChecked(ignore_formats)
        self.ui.ignoreStringLineEdit.setText(ignore_string)
        self.ui.useImportRulesCheckbox.setChecked(use_import_rules)
        
        # Set export schema settings
        self.ui.exportSettingsLineEdit.setText(export_schema)
        self.ui.useExportRulesCheckbox.setChecked(use_export_schema)
        
        # Set server/client settings
        self.ui.enableServerCheckbox.setChecked(enable_server)
        if hasattr(self.ui, 'hostServerAddressLineEdit'):
            self.ui.hostServerAddressLineEdit.setText(host_server_address)
    
    def save_settings(self):
        """
        Save settings from UI controls to INI file.
        This method should be called when the dialog is accepted.
        """
        settings = QSettings('ScanSpace', 'ImageProcessor')
        
        # Save general settings
        settings.setValue('display_log', self.ui.displayLogCheckBox.isChecked())
        settings.setValue('thread_count', self.ui.defaultThreadCountSpinbox.value())
        settings.setValue('export_format', self.ui.defaultExportFormatComboBox.currentText())
        settings.setValue('bit_depth_16_default', self.ui.bitDepth16EnableCheckbox.isChecked())
        settings.setValue('default_colorspace', self.ui.defaultColorspaceComboBox.currentText())
        settings.setValue('use_precalculated_charts', self.ui.usePrecalculatedChartsCheckBox.isChecked())
        settings.setValue('chart_folder_path', self.ui.chartFolderPathLineEdit.text())
        settings.setValue('correct_thumbnails', self.ui.colorCorrectThumbnailsCheckbox.isChecked())
        
        # Save network settings
        if hasattr(self.ui, 'hostServerAddressLineEdit'):
            settings.setValue('host_server_address', self.ui.hostServerAddressLineEdit.text())
        
        # Save log level setting
        if hasattr(self.ui, 'logDebugDepthCombobox'):
            # Map combo box index to LogLevel value
            log_levels = [
                LogLevel.CRITICAL,  # 0: CRITICAL (50)
                LogLevel.ERROR,     # 1: ERROR (40)
                LogLevel.WARNING,   # 2: WARNING (30)
                LogLevel.INFO,      # 3: INFO (20)
                LogLevel.DEBUG,     # 4: DEBUG (10)
                LogLevel.TRACE      # 5: TRACE (5)
            ]
            index = self.ui.logDebugDepthCombobox.currentIndex()
            if 0 <= index < len(log_levels):
                settings.setValue('log_level', log_levels[index].value)
        
        # Save import/export settings
        settings.setValue('look_in_subfolders', self.ui.lookForImagesInSubfolderCheckbox.isChecked())
        settings.setValue('group_by_subfolder', self.ui.groupImagesBySubfolderCheckbox.isChecked())
        settings.setValue('group_by_prefix', self.ui.groupImagesByPrefixCheckbox.isChecked())
        settings.setValue('prefix_string', self.ui.prefixGroupingLineEdit.text())
        settings.setValue('ignore_formats', self.ui.ignoreFormatsCheckbox.isChecked())
        settings.setValue('ignore_string', self.ui.ignoreStringLineEdit.text())
        
        # Save export schema settings
        settings.setValue('export_schema', self.ui.exportSettingsLineEdit.text())
        settings.setValue('use_export_schema', self.ui.useExportRulesCheckbox.isChecked())
        settings.setValue('use_import_rules', self.ui.useImportRulesCheckbox.isChecked())
        
        # Save server/client settings
        settings.setValue('enable_server', self.ui.enableServerCheckbox.isChecked())

        # Save theme settings
        settings.setValue('dark_mode', self.ui.enableDarkThemeCheckbox.isChecked())
        
        # Ensure settings are written to disk
        settings.sync()
    
    def browse_for_default_color_chart(self):
        """Browse for a .npy file containing default color chart swatch data."""
        file_dialog = QFileDialog()
        folder_path, _ = file_dialog.getExistingDirectory(
            self,
            "Select A directory that holds template colour chart calibration files"
        )
        
        if folder_path:
            # Update the path in the UI
            self.ui.chartFolderPathLineEdit.setText(folder_path)
    
    def _setup_export_schema_validation(self):
        """Setup real-time validation and preview for export schema."""
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QLabel

        # Create schema validator
        self.schema_validator = FileNamingSchema()
        
        # Create validation timer to debounce input
        self.validation_timer = QTimer()
        self.validation_timer.setSingleShot(True)
        self.validation_timer.timeout.connect(self._validate_export_schema)
        
        # Connect text change event
        self.ui.exportSettingsLineEdit.textChanged.connect(
            lambda: self.validation_timer.start(500)  # 500ms delay
        )
        
        # Create preview label if it doesn't exist
        if not hasattr(self, 'schema_preview_label'):
            self.schema_preview_label = QLabel()
            self.schema_preview_label.setWordWrap(True)
            self.schema_preview_label.setStyleSheet("QLabel { color: #666; font-size: 10px; margin: 2px; }")
            # Try to find the parent layout and add the label
            try:
                parent_layout = self.ui.exportSettingsLineEdit.parent().layout()
                if parent_layout:
                    # Find the export settings line edit in the layout
                    for i in range(parent_layout.count()):
                        item = parent_layout.itemAt(i)
                        if item and item.widget() == self.ui.exportSettingsLineEdit:
                            parent_layout.insertWidget(i + 1, self.schema_preview_label)
                            break
            except:
                pass  # If layout manipulation fails, continue without preview
        
        # Initial validation
        self._validate_export_schema()
    
    def _validate_export_schema(self):
        """Validate the current export schema and update UI feedback."""
        schema = self.ui.exportSettingsLineEdit.text().strip()
        
        if not schema:
            self._set_schema_feedback("", True)
            return
        
        # Validate schema
        is_valid, errors = self.schema_validator.validate_schema(schema)
        
        # Generate preview
        preview = ""
        if is_valid:
            try:
                # Use a sample path for preview
                sample_path = "C:/Projects/chineseVase05/crossPolarized/IMG_0001.NEF"
                preview_output = self.schema_validator.preview_output(
                    schema=schema,
                    input_path=sample_path,
                    custom_name="CustomName",
                    image_number=1,
                    output_extension=".jpg"
                )
                preview = f"Preview: {preview_output}"
            except Exception as e:
                preview = f"Preview error: {e}"
                is_valid = False
        
        # Set feedback
        if is_valid:
            self._set_schema_feedback(preview, True)
        else:
            error_text = "; ".join(errors) if errors else "Invalid schema"
            self._set_schema_feedback(f"Error: {error_text}", False)
    
    def _set_schema_feedback(self, message, is_valid):
        """Set visual feedback for schema validation."""
        # Update line edit style
        if message == "":
            # No schema - neutral style
            self.ui.exportSettingsLineEdit.setStyleSheet("")
        elif is_valid:
            # Valid schema - green border
            self.ui.exportSettingsLineEdit.setStyleSheet(
                "QLineEdit { border: 2px solid #4CAF50; }"
            )
        else:
            # Invalid schema - red border
            self.ui.exportSettingsLineEdit.setStyleSheet(
                "QLineEdit { border: 2px solid #f44336; }"
            )
        
        # Update preview label if it exists
        if hasattr(self, 'schema_preview_label'):
            self.schema_preview_label.setText(message)
            if is_valid:
                self.schema_preview_label.setStyleSheet(
                    "QLabel { color: #4CAF50; font-size: 10px; margin: 2px; }"
                )
            else:
                self.schema_preview_label.setStyleSheet(
                    "QLabel { color: #f44336; font-size: 10px; margin: 2px; }"
                )


    def accept(self):
        """Override accept to save settings and apply them to main window."""
        self.save_settings()
        
        # Apply settings to the main window immediately
        if self.main_window:
            self.main_window.apply_settings(update_network=True)
        
        super().accept()


class LoadingSpinner(QLabel):
    """
    Animated loading spinner widget using SVG with smooth rotation.
    """
    
    def __init__(self, svg_path: str, size: int = 60, parent=None):
        super().__init__(parent)
        self.spinner_size = size
        self.setFixedSize(size, size)
        
        # Load SVG using QPixmap (simpler and handles transparency correctly)
        self.original_pixmap = QPixmap(svg_path)
        if self.original_pixmap.isNull():
            # Fallback: create a simple circle if SVG fails to load
            self.original_pixmap = QPixmap(size, size)
            self.original_pixmap.fill(Qt.transparent)
            painter = QPainter(self.original_pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(QColor(100, 150, 200, 180))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(5, 5, size-10, size-10)
            painter.end()
        else:
            # Scale the SVG to our desired size
            self.original_pixmap = self.original_pixmap.scaled(
                size, size, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
        
        # Set initial pixmap
        self._render_rotated_pixmap(0)  # Initial render at 0 degrees
        
        # Set up rotation animation
        self.rotation_angle = 0
        self.animation = QPropertyAnimation(self, b"rotation")
        self.animation.setDuration(500)  # 1 second per rotation
        self.animation.setStartValue(0)
        self.animation.setEndValue(360)
        self.animation.setLoopCount(-1)  # Infinite loop
        self.animation.setEasingCurve(QEasingCurve.Linear)
        
        # Connect animation to update the display
        self.animation.valueChanged.connect(self._update_rotation)
        
        # No background styling - let the SVG transparency show through
        self.setStyleSheet("")
    
    def _render_rotated_pixmap(self, angle: float):
        """Render the pixmap with rotation."""
        if self.original_pixmap.isNull():
            return
            
        # Create rotated pixmap
        rotated_pixmap = QPixmap(self.spinner_size, self.spinner_size)
        rotated_pixmap.fill(Qt.transparent)
        
        painter = QPainter(rotated_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Apply rotation transform around center
        center = self.spinner_size / 2
        painter.translate(center, center)
        painter.rotate(angle)
        painter.translate(-center, -center)
        
        # Draw the original pixmap
        painter.drawPixmap(0, 0, self.original_pixmap)
        painter.end()
        
        self.setPixmap(rotated_pixmap)
    
    def _update_rotation(self, angle):
        """Update rotation angle and re-render."""
        self.rotation_angle = angle
        self._render_rotated_pixmap(angle)
    
    # Qt Property for animation
    def get_rotation(self):
        return self.rotation_angle
    
    def set_rotation(self, angle):
        self.rotation_angle = angle
        self._render_rotated_pixmap(angle)
    
    rotation = Property(float, fget=get_rotation, fset=set_rotation)
    
    def start_spinning(self):
        """Start the spinning animation."""
        self.animation.start()
        self.show()
    
    def stop_spinning(self):
        """Stop the spinning animation."""
        self.animation.stop()
        self.hide()


class MainWindow(QMainWindow):
    # Signal for thread-safe UI updates
    update_status_signal = Signal(str, str, float, str)  # image_path, status, processing_time, output_path
    
    def __init__(self):
        super().__init__()

        # ────────────────────────────────────────────────────────────
        # 1) UI Setup
        # ────────────────────────────────────────────────────────────
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        
        # Apply saved theme
        theme_manager.apply_theme(self)

        # Connect signal for thread-safe UI updates
        self.update_status_signal.connect(self.update_image_status)

        # Thread pool for background tasks
        self.threadpool = QThreadPool()

        # Settings Dialog
        self.settings_dialog = None

        # ────────────────────────────────────────────────────────────
        # 2) Application Icon & Settings
        # ────────────────────────────────────────────────────────────
        self.settings = QSettings('scanSpace', 'ImageProcessor')
        # Get the absolute path to the icon file relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path = os.path.join(script_dir, "resources", "imageSpace_logo.ico")
        if os.path.exists(icon_path):
            icon = QIcon(icon_path)
            self.setWindowIcon(icon)
            # Also set the application icon for the dock/taskbar
            QApplication.instance().setWindowIcon(icon)
            print(f"Icon loaded from: {icon_path}")
        else:
            print(f"Icon file not found at: {icon_path}")

        # Restore last-used folders
        rawf = self.settings.value('rawFolder', '')
        outf = self.settings.value('outputFolder', '')
        if rawf:
            self.ui.rawImagesDirectoryLineEdit.setText(rawf)
        if outf:
            self.ui.outputDirectoryLineEdit.setText(outf)

        self.dark_mode = self.settings.value('dark_mode', False, type=bool)

        # Bind imported functions as methods using functools.partial
        self.export_current_project = partial(export_current_project, self)
        self.update_server_status_label = partial(update_server_status_label, self)
        self.send_project_to_server = partial(send_project_to_server, self)
        self._show_submission_confirmation_dialog = partial(_show_submission_confirmation_dialog, self)
        self._build_submission_details = partial(_build_submission_details, self)
        self._on_submission_progress = partial(_on_submission_progress, self)
        self._on_submission_success = partial(_on_submission_success, self)
        self._on_submission_error = partial(_on_submission_error, self)
        self._on_submission_finished = partial(_on_submission_finished, self)
        self._build_project_data = partial(_build_project_data, self)
        self._sanitize_project_data = partial(_sanitize_project_data, self)
        self._debug_find_unserializable_objects = partial(_debug_find_unserializable_objects, self)
        self.ProjectSubmissionWorker = ProjectSubmissionWorker

        # ────────────────────────────────────────────────────────────
        # 3) Preview Scene
        # ────────────────────────────────────────────────────────────
        self.previewScene = QGraphicsScene(self)
        view = self.ui.imagePreviewGraphicsView
        view.setScene(self.previewScene)
        view.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Create loading spinner overlay
        self.loading_spinner = LoadingSpinner(
            svg_path="resources/icons/loader-quarter.png",
            size=64,
            parent=view
        )
        self.loading_spinner.hide()  # Hidden by default

        # ────────────────────────────────────────────────────────────
        # 4) Default UI State
        # ────────────────────────────────────────────────────────────
        # Hide chart tools until needed
        self.ui.detectChartToolshelfFrame.setVisible(False)
        self.ui.colourChartDebugToolsFrame.setVisible(False)  # via setup_debug_views()
        self.ui.processingStatusBarFrame.setVisible(False) # hide our processing status bar
        self.ui.chartInformationLabel.setVisible(False)
        self.cursor = None

        # Disable buttons until prerequisites are met
        for btn in (
            self.ui.detectChartShelfPushbutton,
            self.ui.showOriginalImagePushbutton,
            self.ui.flattenChartImagePushButton,
            self.ui.finalizeChartPushbutton
        ):
            btn.setEnabled(False)

        # Initialize debug‐view toggles
        self.setup_debug_views()

        # Populate the imageFormatComboBox:
        for fmt in OUTPUT_FORMATS:
            self.ui.imageFormatComboBox.addItem(fmt)

        # Hide bit-depth and JPEG-quality frames until the user picks those formats:
        self.ui.bitDepthFrame.setVisible(False)
        self.ui.jpgQualityFrame.setVisible(False)

        # Whenever the format changes, update which controls are visible:
        self.ui.imageFormatComboBox.currentTextChanged.connect(
            self._update_format_controls
        )

        # Populate the combo with every OIIO colourspace
        config = ColorConfig()
        self.colour_space_names = config.getColorSpaceNames()  # returns List[str]
        for cs in self.colour_space_names:
            self.ui.exrColourSpaceComboBox.addItem(cs)

        # Whenever the format changes, update which frames are shown:
        self.ui.imageFormatComboBox.currentTextChanged.connect(self._update_format_controls)
        # Initial call
        self._update_format_controls(self.ui.imageFormatComboBox.currentText())

        # ────────────────────────────────────────────────────────────
        # 5) Internal State Variables
        # ────────────────────────────────────────────────────────────
        # Calibration & chart data - now supports per-group calibrations
        self.calibration_file       = None  # Legacy single calibration
        self.chart_image            = None
        self.chart_swatches         = None  # Legacy single swatches
        self.temp_swatches          = []
        self.flatten_swatch_rects   = None
        self.average_enabled = False
        self.selected_average_source = None
        
        # Group-specific calibrations
        self.group_calibrations = {}  # {group_name: {'file': path, 'swatches': array}}
        self.current_chart_group = None
        self.chart_groups = []  # List of [chart_path, group_name] pairs

        # Mode flags
        self.manual_selection_mode  = False
        self.flatten_mode           = False
        self.showing_chart_preview  = False

        # Image buffers
        self.cropped_fp             = None
        self.original_preview_pixmap= None
        self.fp_image_array         = None

        # Thumbnails & metadata
        self.thumbnail_cache        = {}
        self.image_metadata_map     = {}
        self.correct_thumbnails = False
        
        # Thumbnail view state
        self.ui.thumbnailContainerScrollArea.hide()
        self.is_thumbnail_view_mode = False
        self.thumbnail_zoom_level = 3  # Default number of columns (1-6 range, matches slider default)
        self.thumbnail_grid_widget = None  # Grid widget for thumbnails
        self.thumbnail_grid_layout = None  # Grid layout for thumbnails

        # UI helpers
        self.instruction_label      = None
        self.corner_points          = []

        # Profiling counters
        self.total_images           = 0
        self.finished_images        = 0
        self.global_start           = None
        self.processing_active      = False
        self.active_workers         = []

        # Settings variables
        self.ignore_string = None
        self.ignore_formats = None
        self.prefix_string = None
        self.group_by_prefix = None
        self.group_by_subfolder = None
        self.look_in_subfolders = None
        self.export_schema = None
        self.use_export_schema = None

        self.supported_chart_types = ["Colour Checker Classic", "Colour Checker Passport"]
        self.selected_chart_type = None

        # Initialize server status
        self.update_server_status_label("")
        
        # ────────────────────────────────────────────────────────────
        # 6) Real-time Preview System
        # ────────────────────────────────────────────────────────────
        # Preview state variables
        self.current_raw_path = None          # Currently selected RAW image path
        self.current_raw_array = None         # Full resolution RAW array (float32)
        self.current_preview_array = None     # Resized preview array (max 4MP)
        self.original_white_balance = 5500    # Original image white balance
        self.is_loading_raw = False           # Flag to prevent concurrent loading
        self.sampling_white_balance = False   # Flag for white balance sampling mode
        
        # Preview optimization (removed caching for stability)
        
        # Maximum preview resolution (approximately 4MP)
        self.max_preview_width = 2048
        self.max_preview_height = 2048
        self.use_import_rules = None
        
        # Logging system
        self.current_log_level = LogLevel.INFO  # Default to INFO level

        # ────────────────────────────────────────────────────────────
        # 6) System Usage Meters
        # ────────────────────────────────────────────────────────────
        for bar in (self.ui.cpuUsageProgressBar, self.ui.memoryUsageProgressBar):
            bar.setRange(0, 100)
            bar.setFormat(f"{bar.objectName().replace('ProgressBar','')} Usage: %p%")

        self.ui.cpuUsageProgressBar.setRange(0, 100)
        self.ui.memoryUsageProgressBar.setRange(0, 100)
        self.ui.cpuUsageProgressBar.setFormat("CPU usage: %p%")
        self.ui.memoryUsageProgressBar.setFormat("Memory usage: %p%")
        self.cpuTimer = QTimer(self)
        self.cpuTimer.timeout.connect(self.update_system_usage)
        self.cpuTimer.start(1000)

        # RAW loading delay timer (prevents slowdowns during fast scrolling)
        self.raw_load_timer = QTimer(self)
        self.raw_load_timer.setSingleShot(True)  # Only fire once
        self.raw_load_timer.timeout.connect(self._on_raw_load_timer)
        self.pending_raw_path = None  # Path to load when timer fires

        # Status‐bar warning label
        self.memoryWarningLabel = QLabel('', self)
        self.statusBar().addPermanentWidget(self.memoryWarningLabel)

        # ────────────────────────────────────────────────────────────
        # 7) Manual‐Selection RubberBand
        # ────────────────────────────────────────────────────────────
        self.rubberBand = QRubberBand(QRubberBand.Rectangle,
                                      self.ui.imagePreviewGraphicsView.viewport())

        # ────────────────────────────────────────────────────────────
        # 8) Signal Connections
        # ────────────────────────────────────────────────────────────
        ui = self.ui  # alias for brevity

        ui.browseForChartPushbutton.clicked.connect(self.browse_chart)
        ui.browseForImagesPushbutton.clicked.connect(self.browse_images)
        ui.browseoutputDirectoryPushbutton.clicked.connect(self.browse_output_directory)
        
        # Connect precalculated chart functionality
        if hasattr(ui, 'exportChartConfigPushButton'):
            ui.exportChartConfigPushButton.clicked.connect(self.export_chart_config)
        if hasattr(ui, 'usePrecalcChartCheckbox'):
            ui.usePrecalcChartCheckbox.toggled.connect(self.on_use_precalc_chart_toggled)
        if hasattr(ui, 'precalcChartComboBox'):
            ui.precalcChartComboBox.currentIndexChanged.connect(self.on_precalc_chart_selection_changed)
        
        # Connect project export functionality
        if hasattr(ui, 'exportCurrentProjectPushButton'):
            ui.exportCurrentProjectPushButton.clicked.connect(self.export_current_project)
        
        # Connect send project to server functionality
        if hasattr(ui, 'sendProjectToServerPushButton'):
            ui.sendProjectToServerPushButton.clicked.connect(self.send_project_to_server)

        ui.setSelectedAsChartPushbutton.clicked.connect(self.set_selected_as_chart)
        ui.processImagesPushbutton.clicked.connect(self.process_images_button_clicked)
        
        # Chart management buttons
        ui.copyChartPushbutton.clicked.connect(lambda: ChartTools.copy_chart_assignment(self))
        ui.pasteChartPushbutton.clicked.connect(lambda: ChartTools.paste_chart_assignment(self))
        ui.useSelectedChartForAllGroupsPushbutton.clicked.connect(lambda: ChartTools.use_selected_chart_for_all_groups(self))
        ui.removeSelectedChartPushbutton.clicked.connect(lambda: ChartTools.remove_selected_chart_assignment(self))

        ui.imagesListWidget.itemSelectionChanged.connect(self.preview_selected)
        ui.imagesListWidget.currentRowChanged.connect(self.update_thumbnail_strip)

        ui.manuallySelectChartPushbutton.clicked.connect(self.manually_select_chart)
        ui.detectChartShelfPushbutton.clicked.connect(
            lambda: self.detect_chart(input_source=ui.chartPathLineEdit.text(), is_npy=False)
        )
        ui.revertImagePushbutton.clicked.connect(self.revert_image)
        ui.showOriginalImagePushbutton.clicked.connect(self.toggle_chart_preview)
        ui.flattenChartImagePushButton.clicked.connect(self.flatten_chart_image)
        ui.finalizeChartPushbutton.clicked.connect(self.finalize_manual_chart_selection)

        ui.calculateAverageExposurePushbutton.clicked.connect(self.calculate_average_exposure)
        ui.removeAverageDataPushbutton.clicked.connect(self.remove_average_exposure_data)
        ui.displayDebugExposureDataCheckBox.toggled.connect(
            lambda checked: checked and self.show_exposure_debug_overlay()
        )
        ui.highlightLimitSpinBox.valueChanged.connect(
            lambda: self.show_exposure_debug_overlay() if ui.displayDebugExposureDataCheckBox.isChecked() else None
        )
        ui.shadowLimitSpinBox.valueChanged.connect(
            lambda: self.show_exposure_debug_overlay() if ui.displayDebugExposureDataCheckBox.isChecked() else None
        )

        # Add colour charts to supported charts dropdown
        ui.supportedColourChartsComboBox.addItems(self.supported_chart_types)
        ui.supportedColourChartsComboBox.setCurrentIndex(0)
        ui.supportedColourChartsComboBox.currentIndexChanged.connect(self.set_chart_type)

        ui.nextImagePushbutton.clicked.connect(self.select_next_image)
        ui.previousImagePushbutton.clicked.connect(self.select_previous_image)
        ui.setSelectedImageAsAveragePushbutton.clicked.connect(self.set_selected_image_as_average_source)
        ui.actionDocumentation.triggered.connect(self.open_web_help)
        ui.actionGithub.triggered.connect(self.open_github_page)

        # ────────────────────────────────────────────────────────────
        # Real-time Preview Editing Controls
        # ────────────────────────────────────────────────────────────
        # Connect exposure controls
        ui.exposureAdjustmentSlider.valueChanged.connect(self.on_exposure_adjustment_changed)
        ui.exposureAdjustmentDoubleSpinBox.valueChanged.connect(self.on_exposure_spinbox_changed)
        
        # Connect shadow/highlight controls - use sliderReleased for sliders to avoid continuous updates
        ui.shadowAdjustmentSlider.sliderReleased.connect(self.on_shadow_adjustment_changed)
        ui.shadowAdjustmentDoubleSpinBox.valueChanged.connect(self.on_shadow_spinbox_changed)
        ui.highlightAdjustmentSlider.sliderReleased.connect(self.on_highlight_adjustment_changed)
        ui.highlightAdjustmentDoubleSpinBox.valueChanged.connect(self.on_highlight_spinbox_changed)
        
        # Connect white balance controls
        ui.enableWhiteBalanceCheckBox.toggled.connect(self.on_white_balance_checkbox_changed)
        ui.whitebalanceSpinbox.valueChanged.connect(self.on_white_balance_changed)
        ui.sampleWhiteBalancePushButton.clicked.connect(self.sample_white_balance_from_image)
        
        # Connect denoise controls
        ui.denoiseImageCheckBox.toggled.connect(self.on_denoise_checkbox_changed)
        ui.denoiseHorizontalSlider.valueChanged.connect(self.on_denoise_slider_changed)
        ui.denoiseDoubleSpinBox.valueChanged.connect(self.on_denoise_spinbox_changed)
        
        # Connect sharpen controls
        ui.sharpenImageCheckBox.toggled.connect(self.on_sharpen_checkbox_changed)
        ui.sharpenHorizontalSlider.valueChanged.connect(self.on_sharpen_slider_changed)
        ui.sharpenDoubleSpinBox.valueChanged.connect(self.on_sharpen_spinbox_changed)
        
        # Sync sliders with spinboxes (bidirectional)
        self._sync_exposure_controls = True
        self._sync_shadow_controls = True
        self._sync_highlight_controls = True
        self._sync_denoise_controls = True
        self._sync_sharpen_controls = True

        ui.resetEditSettingsPushButton.clicked.connect(self.reset_image_editing_sliders)

        self.ui.actionLoadSettingsMenu.triggered.connect(self.open_settings_dialog)

        # ────────────────────────────────────────────────────────────
        # Thumbnail Grid View Controls
        # ────────────────────────────────────────────────────────────
        ui.displayImagesAsThumbnailsCheckBox.toggled.connect(self.toggle_thumbnail_view_mode)
        ui.thumbnailZoomLevelHorizontalSlider.valueChanged.connect(self.update_thumbnail_zoom_level)

        # ────────────────────────────────────────────────────────────
        # 9) Event Filters & Focus
        # ────────────────────────────────────────────────────────────
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        ui.thumbnailPreviewFrame.installEventFilter(self)
        ui.imagePreviewGraphicsView.viewport().installEventFilter(self)
        ui.thumbnailPreviewFrame.installEventFilter(self)

        # Keep track of the currently displayed pixmap
        self.current_image_pixmap = None

        # print to log that the application is ready
        self.log_info("Scan Space Image Processor initialized")
        
        # Load and apply settings from INI file
        self.apply_settings()
        
        # Set up file naming schema controls
        from ImageProcessor.fileNamingSchema import FileNamingSchema
        self.file_naming_schema = FileNamingSchema()
        self.file_naming_schema.setup_ui_controls(self)
        
        # Set up rescaling controls
        from ImageProcessor.editingTools import setup_rescaling_controls
        setup_rescaling_controls(self)

        # Connect quit action if it exists
        if hasattr(self.ui, 'actionQuit'):
            self.ui.actionQuit.triggered.connect(self.close)
            # Add keyboard shortcut for quit (Cmd+Q on Mac, Ctrl+Q on others)
            self.ui.actionQuit.setShortcut(QKeySequence.StandardKey.Quit)

    def switch_theme(self, theme_name):
        """Switch to the specified theme."""
        # Apply theme
        theme_manager.apply_theme(self, theme_name)
        theme_manager.save_theme_preference(theme_name)
        
        # Update graphics view background for better visibility
        colors = theme_manager.get_theme_colors(theme_name)
        self.ui.imagePreviewGraphicsView.setBackgroundBrush(QColor(colors['background']))
        
        # Log the change
        self.log_info(f"Switched to {theme_name} theme")
        
        # Initialize network mode after settings are applied
        self._update_network_mode()

    def apply_settings(self, update_network=False):
        """
        Apply settings from INI file to the main window UI controls.
        
        This method reads settings from QSettings and applies them to the
        actual UI controls in the main window. Called on startup and when
        settings are changed in the settings dialog.
        
        Args:
            update_network (bool): Whether to update network mode (only when network settings change)
        """
        settings = QSettings('ScanSpace', 'ImageProcessor')
        
        # Apply display log setting
        display_log = settings.value('display_log', True, type=bool)
        if display_log:
            self.ui.logOutputTextEdit.show()
        else:
            self.ui.logOutputTextEdit.hide()
        
        # Apply log level setting
        log_level = settings.value('log_level', LogLevel.INFO, type=int)
        if log_level == 0:
            self.current_log_level = LogLevel.INFO
        else:
            self.current_log_level = LogLevel(log_level)
        self.log_info(f"Log level set to {self.current_log_level.name} ({self.current_log_level.value})")
        
        # Apply default thread count setting
        thread_count = settings.value('thread_count', 4, type=int)
        self.threadpool.setMaxThreadCount(thread_count)
        self.log_debug(f"[Settings] Thread count set to {thread_count}")
        
        # Apply default export format setting
        export_format = settings.value('export_format', '.jpg', type=str)
        if hasattr(self.ui, 'exportFormatComboBox'):
            index = self.ui.exportFormatComboBox.findText(export_format)
            if index >= 0:
                self.ui.exportFormatComboBox.setCurrentIndex(index)
        
        # Apply bit depth 16 default setting
        bit_depth_16 = settings.value('bit_depth_16_default', False, type=bool)
        if hasattr(self.ui, 'sixteenBitRadioButton'):
            self.ui.sixteenBitRadioButton.setChecked(bit_depth_16)
            # Also uncheck 8-bit if 16-bit is enabled
            if bit_depth_16 and hasattr(self.ui, 'eightBitRadioButton'):
                self.ui.eightBitRadioButton.setChecked(False)
        
        # Apply default colorspace setting
        default_colorspace = settings.value('default_colorspace', 'sRGB', type=str)
        if hasattr(self.ui, 'exrColourSpaceComboBox'):
            index = self.ui.exrColourSpaceComboBox.findText(default_colorspace)
            if index >= 0:
                self.ui.exrColourSpaceComboBox.setCurrentIndex(index)
        
        # Load general settings
        self.correct_thumbnails = settings.value('correct_thumbnails', False, type=bool)
        
        # Apply import/export settings
        self.look_in_subfolders = settings.value('look_in_subfolders', False, type=bool)
        self.group_by_subfolder = settings.value('group_by_subfolder', False, type=bool)
        self.group_by_prefix = settings.value('group_by_prefix', False, type=bool)
        self.prefix_string = settings.value('prefix_string', '', type=str)
        self.ignore_formats = settings.value('ignore_formats', False, type=bool)
        self.ignore_string = settings.value('ignore_string', '', type=str)

        self.export_schema = settings.value('export_schema', False, type=str)
        self.use_export_schema = settings.value('use_export_schema', '', type=bool)
        self.use_import_rules = settings.value('use_import_rules', False, type=bool)
        
        # Update process button text based on calibration availability
        self.update_process_button_text()
        
        # Apply precalculated chart settings
        use_precalc_charts = settings.value('use_precalculated_charts', False, type=bool)
        chart_folder_path = settings.value('chart_folder_path', '', type=str)
        
        # Populate the precalc chart combo box if available
        if hasattr(self.ui, 'precalcChartComboBox'):
            self.populate_precalc_chart_combo(chart_folder_path)
            
        # Set the checkbox state if available
        if hasattr(self.ui, 'usePrecalcChartCheckbox'):
            self.ui.usePrecalcChartCheckbox.setChecked(use_precalc_charts)
            
        # Load precalculated chart if enabled
        if use_precalc_charts:
            self.load_selected_precalc_chart()
        
        # Update network mode only when explicitly requested (when network settings change)
        if update_network:
            self._update_network_mode()

        # Set the application theme
        theme = settings.value('dark_mode', False, type=bool)
        if theme:
            self.switch_theme('dark')
        else:
            self.switch_theme('light')
        
        self.log_info("[Settings] Settings applied successfully")

    def closeEvent(self, event):
        """Handle application close event - clean shutdown."""
        # Stop any running timers
        if hasattr(self, 'cpuTimer'):
            self.cpuTimer.stop()
        if hasattr(self, 'raw_load_timer'):
            self.raw_load_timer.stop()

        # Clean up thread pool
        if hasattr(self, 'threadpool'):
            self.threadpool.waitForDone(1000)  # Wait max 1 second for threads

        # Save window state/geometry if needed
        if hasattr(self, 'settings'):
            # You could save window geometry here if desired
            pass

        # Accept the close event
        event.accept()

        # Ensure application quits
        QApplication.instance().quit()

    def load_default_color_chart(self, chart_path):
        """
        Load default color chart from .npy file.
        
        Args:
            chart_path: Path to the .npy file containing chart swatch data
        """
        try:
            # Load the numpy array from file
            chart_swatches = np.load(chart_path)
            
            # Validate the array dimensions (should be N x 3 for RGB values)
            if len(chart_swatches.shape) != 2 or chart_swatches.shape[1] != 3:
                raise ValueError(f"Invalid chart data shape: {chart_swatches.shape}. Expected (N, 3)")
            
            # Store the loaded data
            self.calibration_file = chart_path
            self.chart_swatches = chart_swatches
            
            self.log_info(f"[Settings] Loaded default color chart: {os.path.basename(chart_path)} ({chart_swatches.shape[0]} swatches)")
            
        except Exception as e:
            self.log_error(f"[Settings] Failed to load default color chart: {e}")
            self.calibration_file = None
            self.chart_swatches = None

    def update_process_button_text(self):
        """Update the process button text based on available calibrations."""
        if self.group_calibrations:
            group_count = len(self.group_calibrations)
            self.ui.processImagesPushbutton.setText(f"Process images ({group_count} group calibrations)")
        elif self.calibration_file and self.chart_swatches is not None:
            self.ui.processImagesPushbutton.setText("Process images using default calibration")
        else:
            self.ui.processImagesPushbutton.setText("Process Images")

    def open_settings_dialog(self):
        """Open the settings dialog."""
        self.settings_dialog = ImageProcessorSettingsDialog(self)
        self.settings_dialog.load_settings()  # Load current settings
        
        if self.settings_dialog.exec() == QDialog.Accepted:
            # Save the settings from the dialog
            self.settings_dialog.save_settings()
            # Apply the updated settings to the main window
            self.apply_settings(update_network=True)
            self.log_info("[Settings] Settings saved and applied")

    def populate_precalc_chart_combo(self, chart_folder_path):
        """
        Populate the precalculated chart combo box with .npy files from the specified folder.
        
        Args:
            chart_folder_path: Path to folder containing .npy chart files
        """
        if not hasattr(self.ui, 'precalcChartComboBox'):
            return
            
        # Clear existing items
        self.ui.precalcChartComboBox.clear()
        
        if not chart_folder_path or not os.path.exists(chart_folder_path):
            self.ui.precalcChartComboBox.addItem("No chart folder selected")
            self.ui.precalcChartComboBox.setEnabled(False)
            return
            
        try:
            # Find all .npy files in the folder
            npy_files = []
            for file in os.listdir(chart_folder_path):
                if file.lower().endswith('.npy'):
                    npy_files.append(file)
            
            if not npy_files:
                self.ui.precalcChartComboBox.addItem("No .npy files found")
                self.ui.precalcChartComboBox.setEnabled(False)
                return
            
            # Sort files alphabetically
            npy_files.sort()
            
            # Add files to combo box
            for file in npy_files:
                # Store the full path in the user data
                full_path = os.path.join(chart_folder_path, file)
                self.ui.precalcChartComboBox.addItem(file, full_path)
            
            # Set default selection - try to restore saved selection, otherwise use first item
            settings = QSettings('ScanSpace', 'ImageProcessor')
            saved_chart = settings.value('selected_precalc_chart', '', type=str)
            
            selected_index = 0  # Default to first item
            if saved_chart:
                # Try to find the previously selected chart
                for i in range(self.ui.precalcChartComboBox.count()):
                    if self.ui.precalcChartComboBox.itemText(i) == saved_chart:
                        selected_index = i
                        break
                        
            self.ui.precalcChartComboBox.setCurrentIndex(selected_index)
            self.ui.precalcChartComboBox.setEnabled(True)
            
            if saved_chart and self.ui.precalcChartComboBox.itemText(selected_index) == saved_chart:
                self.log_info(f"[Charts] Found {len(npy_files)} precalculated chart files, restored selection: {saved_chart}")
            else:
                current_selection = self.ui.precalcChartComboBox.itemText(selected_index)
                self.log_info(f"[Charts] Found {len(npy_files)} precalculated chart files, selected: {current_selection}")
            
        except Exception as e:
            self.ui.precalcChartComboBox.addItem("Error reading folder")
            self.ui.precalcChartComboBox.setEnabled(False)
            self.log_error(f"[Charts] Error reading chart folder: {e}")

    def set_chart_type(self):
        current_index = self.ui.supportedColourChartsComboBox.currentIndex()
        self.selected_chart_type = self.ui.supportedColourChartsComboBox.itemText(current_index)

    def export_chart_config(self):
        """
        Export the current chart configuration to a .npy file.
        """
        # Check if we have a current chart to export
        if not hasattr(self, 'chart_swatches') or self.chart_swatches is None:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Chart Data",
                "No chart data available to export. Please load or detect a chart first."
            )
            return
        
        # Get the default save directory from settings
        settings = QSettings('ScanSpace', 'ImageProcessor')
        chart_folder_path = settings.value('chart_folder_path', '', type=str)
        
        if not chart_folder_path:
            chart_folder_path = os.getcwd()  # Use current directory if no folder set
        
        # Open file save dialog
        file_dialog = QFileDialog(self)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("NumPy files (*.npy)")
        file_dialog.setDefaultSuffix("npy")
        file_dialog.setDirectory(chart_folder_path)
        file_dialog.setWindowTitle("Export Chart Configuration")
        
        if file_dialog.exec() == QFileDialog.Accepted:
            file_path = file_dialog.selectedFiles()[0]
            
            try:
                # Save the chart swatches array
                np.save(file_path, self.chart_swatches)
                
                # Refresh the combo box if it's pointing to the same folder
                if os.path.dirname(file_path) == chart_folder_path:
                    self.populate_precalc_chart_combo(chart_folder_path)
                
                self.log_info(f"[Charts] Chart configuration exported to: {file_path}")
                
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.information(
                    self,
                    "Export Successful", 
                    f"Chart configuration saved to:\n{file_path}"
                )
                
            except Exception as e:
                self.log_error(f"[Charts] Error exporting chart: {e}")
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Failed to export chart configuration:\n{str(e)}"
                )


    def on_use_precalc_chart_toggled(self, checked):
        """
        Handle the precalculated chart checkbox toggle.
        
        Args:
            checked: True if checkbox is checked
        """
        if hasattr(self.ui, 'precalcChartComboBox'):
            self.ui.precalcChartComboBox.setEnabled(checked)
        
        if checked:
            # Refresh the combo box with latest charts from folder
            settings = QSettings('ScanSpace', 'ImageProcessor')
            chart_folder_path = settings.value('chart_folder_path', '', type=str)
            self.populate_precalc_chart_combo(chart_folder_path)
            
            # Load the selected precalculated chart
            self.load_selected_precalc_chart()
        else:
            # Clear precalc chart data, fall back to manual chart
            self.chart_swatches = None
            self.calibration_file = None
            self.log_info("[Charts] Precalculated chart disabled")
        
        # Update process button text
        self.update_process_button_text()

    def load_selected_precalc_chart(self):
        """Load the currently selected precalculated chart."""
        if not hasattr(self.ui, 'precalcChartComboBox') or not hasattr(self.ui, 'usePrecalcChartCheckbox'):
            return
            
        if not self.ui.usePrecalcChartCheckbox.isChecked():
            return
            
        # Get the selected chart path
        current_index = self.ui.precalcChartComboBox.currentIndex()
        if current_index < 0:
            return
            
        chart_path = self.ui.precalcChartComboBox.itemData(current_index)
        if not chart_path or not os.path.exists(chart_path):
            self.log_warning(f"[Charts] Selected chart file not found: {chart_path}")
            return
        
        # Load the chart using the existing method
        self.load_default_color_chart(chart_path)

    def on_precalc_chart_selection_changed(self, index):
        """Handle precalculated chart selection change in combo box."""
        # Save the selected chart to settings
        if hasattr(self.ui, 'precalcChartComboBox') and index >= 0:
            chart_filename = self.ui.precalcChartComboBox.itemText(index)
            if chart_filename and not chart_filename.startswith("No ") and not chart_filename.startswith("Error"):
                settings = QSettings('ScanSpace', 'ImageProcessor')
                settings.setValue('selected_precalc_chart', chart_filename)
                self.log_debug(f"[Charts] Saved chart selection: {chart_filename}")
        
        # Only load if the precalc checkbox is enabled
        if hasattr(self.ui, 'usePrecalcChartCheckbox') and self.ui.usePrecalcChartCheckbox.isChecked():
            self.load_selected_precalc_chart()

    def _update_network_mode(self):
        """Update network processing mode based on settings."""
        settings = QSettings('ScanSpace', 'ImageProcessor')
        enable_server = settings.value('enable_server', False, type=bool)
        host_server_address = settings.value('host_server_address', '', type=str)
        
        # If we have a host server address configured, use job-sending mode instead of processing client
        if host_server_address:
            self.network_mode = 'job_sender'
            self.log_info(f"[Network] Job sending mode enabled for server: {host_server_address}")
            self.update_server_status_label("Ready to send jobs")
            self.ui.sendProjectToServerPushButton.setEnabled(True)
            return
        
        # Only start network processing if server is enabled
        if not enable_server:
            self.network_mode = 'local'
            self.log_info("[Network] Server functionality is disabled in settings")
            self.ui.sendProjectToServerPushButton.setEnabled(False)
            return


    def log(self, msg, level=LogLevel.INFO):
        """
        Enhanced logging with level filtering.
        
        Args:
            msg (str): The message to log
            level (LogLevel): The importance level of the message
        """
        # Check if message should be displayed based on current log level
        if level < self.current_log_level:
            return
        
        # Format timestamp
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        # Format level name
        level_names = {
            LogLevel.CRITICAL: "CRITICAL",
            LogLevel.ERROR: "ERROR", 
            LogLevel.WARNING: "WARNING",
            LogLevel.INFO: "INFO",
            LogLevel.DEBUG: "DEBUG",
            LogLevel.TRACE: "TRACE"
        }
        level_name = level_names.get(level, "INFO")
        
        # Create formatted message
        if level >= LogLevel.ERROR:
            formatted_msg = f"[{timestamp}] [{level_name}] {msg}"
        elif level == LogLevel.WARNING:
            formatted_msg = f"[{timestamp}] [{level_name}] {msg}"
        elif level == LogLevel.DEBUG:
            formatted_msg = f"[{timestamp}] [DEBUG] {msg}"
        elif level == LogLevel.TRACE:
            formatted_msg = f"[{timestamp}] [TRACE] {msg}"
        else:
            # INFO level - cleaner format
            formatted_msg = f"[{timestamp}] {msg}"
        
        # Append the message
        self.ui.logOutputTextEdit.append(formatted_msg)
        
        # Auto-scroll to bottom to show latest messages
        scrollbar = self.ui.logOutputTextEdit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def log_critical(self, msg):
        """Log a critical error message."""
        self.log(msg, LogLevel.CRITICAL)
    
    def log_error(self, msg):
        """Log an error message."""
        self.log(msg, LogLevel.ERROR)
    
    def log_warning(self, msg):
        """Log a warning message."""
        self.log(msg, LogLevel.WARNING)
    
    def log_info(self, msg):
        """Log an info message."""
        self.log(msg, LogLevel.INFO)
    
    def log_debug(self, msg):
        """Log a debug message."""
        self.log(msg, LogLevel.DEBUG)
    
    def log_trace(self, msg):
        """Log a trace message."""
        self.log(msg, LogLevel.TRACE)

    def process_images_button_clicked(self):
        if not self.processing_active:
            # Start processing
            self.processing_active = True
            self.ui.processImagesPushbutton.setText("Cancel Processing")
            self.ui.processImagesPushbutton.setStyleSheet("background-color: #f77; color: black;")
            self.start_processing()
        else:
            # Cancel
            self.cancel_processing()

    def open_github_page(self):
        """Opens the github page."""
        webbrowser.open("https://github.com/ErikScansNz/scanSpaceImageProcessor")

    def open_web_help(self):
        """Opens the web help."""
        webbrowser.open("https://scanspace.nz/blogs/news/batch-process-photogrammetry-images-for-free")

    def browse_chart(self):
        """
        Open a file dialog to select a RAW chart file.
        On selection, update the chart‐path line edit and log the choice.
        """
        default = self.ui.chartPathLineEdit.text() or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select Chart Image', default,
            filter='RAW Files (*.nef *.cr2 *.cr3 *.dng *.arw *.raw)'
        )
        if path:
            self.ui.chartPathLineEdit.setText(path)
            # self.log_debug(f"[Browse] Chart set to {path}")

    def reset_all_state(self):
        """Clear all cached/cached/selection data and UI for a new session."""
        # Clear all metadata
        self.thumbnail_cache.clear()
        self.calibration_file = None
        self.chart_image = None
        self.chart_swatches = None
        self.temp_swatches = []
        self.flatten_swatch_rects = None
        
        # Clear group-specific data
        self.group_calibrations.clear()
        self.chart_groups.clear()
        self.current_chart_group = None
        
        # Update process button text
        self.update_process_button_text()
        self.cropped_fp = None
        self.original_preview_pixmap = None
        self.current_image_pixmap = None
        self.fp_image_array = None
        self.corrected_preview_pixmap = None
        self.manual_selection_mode = False
        self.flatten_mode = False
        self.corner_points = []
        self.showing_chart_preview = False

        # Clear preview, chart path, and instruction label
        self.previewScene.clear()
        self.ui.chartPathLineEdit.clear()

        # Clear debug tools and disable advanced UI actions
        self.ui.colourChartDebugToolsFrame.setVisible(False)
        self.ui.detectChartToolshelfFrame.setVisible(False)
        self.ui.flattenChartImagePushButton.setEnabled(False)
        self.ui.finalizeChartPushbutton.setEnabled(False)
        self.ui.showOriginalImagePushbutton.setEnabled(False)
        self.ui.detectChartShelfPushbutton.setEnabled(False)
        self.ui.finalizeChartPushbutton.setStyleSheet('')
        self.ui.chartInformationLabel.setVisible(False)

        # Reset edit fields
        self.reset_image_editing_sliders(update=False)

        # Remove all thumbnail widgets in the holder
        holder = self.ui.thumbnailPreviewDisplayFrame_holder
        for child in holder.findChildren(QWidget):
            child.setParent(None)
            child.deleteLater()

        # Clear image list widget
        self.ui.imagesListWidget.clear()

        self.apply_settings(update_network=False)


    @ChartTools.exit_manual_mode
    def browse_images(self):
        """
        Open a directory dialog to select a folder of RAW images.
        Supports grouping by subfolder and prefix, as well as ignore string filtering.
        """
        default = self.ui.rawImagesDirectoryLineEdit.text() or os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, 'Select Raw Image Directory', default)
        if not folder:
            return

        self.reset_all_state()

        self.ui.rawImagesDirectoryLineEdit.setText(folder)
        self.settings.setValue('rawFolder', folder)
        self.ui.imagesListWidget.clear()

        
        # Collect all image files with grouping logic
        image_groups = self._collect_images_with_grouping(folder)
        
        # Populate the list widget with grouped structure
        self._populate_grouped_list_widget(image_groups)
        
        if self.ui.imagesListWidget.count() > 0:
            self.ui.imagesListWidget.setCurrentRow(0)

    def _collect_images_with_grouping(self, base_folder):
        """
        Collect all image files from the specified folder with grouping logic.
        
        Args:
            base_folder: Base folder to scan for images
            
        Returns:
            dict: Dictionary with group names as keys and lists of image paths as values
        """
        image_groups = {}
        
        # Grouping settings are already loaded from QSettings in apply_settings()
        # No need to read from UI here as these are managed through the settings dialog
        
        # Get ignore strings if enabled
        ignore_strings = []
        if hasattr(self, 'ignore_formats') and self.ignore_formats and hasattr(self, 'ignore_string'):
            ignore_strings = [s.strip().lower() for s in self.ignore_string.split(',') if s.strip()]
        
        def should_ignore_file(filepath):
            """Check if file should be ignored based on ignore strings."""
            if not ignore_strings:
                return False
            filename = os.path.basename(filepath).lower()
            return any(ignore_str in filename or ignore_str in filepath.lower() for ignore_str in ignore_strings)
        
        def collect_from_folder(folder, relative_path=""):
            """Recursively collect images from a folder."""
            try:
                for item in sorted(os.listdir(folder)):
                    item_path = os.path.join(folder, item)
                    
                    if os.path.isfile(item_path):
                        # Check if it's a supported image format
                        if not item.lower().endswith(RAW_EXTENSIONS):
                            continue
                            
                        # Check ignore strings
                        if should_ignore_file(item_path):
                            self.log_debug(f"[Browse] Ignoring file: {item}")
                            continue
                        
                        # Determine which group this image belongs to
                        group_name = self._determine_image_group(item_path, relative_path, item, folder)
                        
                        # Debug logging
                        self.log_debug(f"[Browse] File: {item}, relative_path: '{relative_path}', group: '{group_name}'")
                        
                        if group_name not in image_groups:
                            image_groups[group_name] = []
                        image_groups[group_name].append(item_path)
                        
                    elif os.path.isdir(item_path) and hasattr(self, 'look_in_subfolders') and self.look_in_subfolders:
                        # Recursively scan subfolders if enabled
                        sub_relative = os.path.join(relative_path, item) if relative_path else item
                        collect_from_folder(item_path, sub_relative)
            except (OSError, PermissionError) as e:
                self.log_error(f"[Browse] Error accessing folder {folder}: {e}")

        if self.use_import_rules:
            collect_from_folder(base_folder)
        else:
            group_name = os.path.basename(base_folder)
            image_groups[group_name] = []
            for item in sorted(os.listdir(base_folder)):
                item_path = os.path.join(base_folder, item)

                if os.path.isfile(item_path):
                    if item.lower().endswith(RAW_EXTENSIONS):
                        image_groups[group_name].append(item_path)

        return image_groups
    
    def _determine_image_group(self, full_path, relative_path, filename, root_path):
        """
        Determine which group an image should belong to based on grouping settings.
        
        Args:
            full_path: Full path to the image file
            relative_path: Relative path from base folder
            filename: Just the filename
            root_path: The path the user input to look for images
            
        Returns:
            str: Group name for this image
        """
        # Debug logging
        self.log_debug(f"[Debug] _determine_image_group: full_path='{full_path}', root_path='{root_path}', filename='{filename}'")
        self.log_debug(f"[Debug] group_by_subfolder={getattr(self, 'group_by_subfolder', 'not set')}, group_by_prefix={getattr(self, 'group_by_prefix', 'not set')}")
        
        # Priority order: subfolder grouping, then prefix grouping, then default
        
        # Group by subfolder if enabled and image is in a subfolder
        if hasattr(self, 'group_by_subfolder') and self.group_by_subfolder and relative_path:
            # Use the first folder in the relative path as the group name
            # relative_path is already calculated correctly by the calling function
            
            # Normalize path separators and split into components
            path_parts = relative_path.replace('\\', '/').split('/')
            
            # Filter out empty parts
            path_parts = [part for part in path_parts if part]
            
            if path_parts:
                # Use the first directory component as the group name
                group_name = path_parts[0]
                self.log_debug(f"[Group] Subfolder group for '{filename}': '{group_name}'")
                return group_name
            else:
                # File is at root level - return a default group name
                self.log_debug(f"[Group] File '{filename}' is at root level")
                return "Root Files"
        
        # Group by prefix if enabled
        if hasattr(self, 'group_by_prefix') and self.group_by_prefix and hasattr(self, 'prefix_string') and self.prefix_string:
            separator = self.prefix_string.strip()
            if separator and separator in filename:
                # Split on the separator and take everything before it as the group name
                parts = filename.split(separator, 1)  # Split only on first occurrence
                if len(parts) > 1:  # Ensure there was something after the separator
                    group_name = parts[0]
                    # Log the grouping for debugging
                    # self.log_debug(f"[Group] File '{filename}' -> Group '{group_name}' (separator: '{separator}')")
                    return group_name
            
            # If separator not found or no text after separator, use the whole filename as group
            # This handles edge cases like files that don't contain the separator
            fallback_group = os.path.splitext(filename)[0]  # Remove extension for cleaner group name
            # self.log_debug(f"[Group] File '{filename}' -> Fallback group '{fallback_group}' (separator '{separator}' not found)")
            return fallback_group
        
        # Default group
        return "All Images"
    
    def _populate_grouped_list_widget(self, image_groups):
        """
        Populate the list widget with grouped structure.
        
        Args:
            image_groups: Dictionary with group names and image lists
        """
        
        total_images = 0
        for group_name, image_paths in image_groups.items():
            # Add group header if we have multiple groups
            if len(image_groups) > 1:
                header_item = QListWidgetItem(f"📁 {group_name} ({len(image_paths)} images)")
                header_item.setData(Qt.UserRole, {'is_group_header': True, 'group_name': group_name})
                header_item.setFlags(header_item.flags() & ~Qt.ItemIsSelectable)  # Make non-selectable
                # Style the group header
                if self.dark_mode:
                    header_item.setBackground(QColor('#5a5c57'))
                else:
                    header_item.setBackground(QColor('#9e7c44'))
                font = header_item.font()
                font.setBold(True)
                header_item.setFont(font)
                self.ui.imagesListWidget.addItem(header_item)
            
            # Add images in this group
            for image_path in sorted(image_paths):
                filename = os.path.basename(image_path)
                
                metadata = {
                    'input_path': image_path,
                    'output_path': None,
                    'status': 'raw',
                    'calibration': None,
                    'data_type': None,
                    'chart': False,
                    'processing_time': None,
                    'debug_images': {},
                    'group_name': group_name,
                    'is_group_header': False
                }
                
                # Indent images under group headers
                display_name = f"   {filename}" if len(image_groups) > 1 else filename
                
                item = QListWidgetItem(display_name)
                item.setData(Qt.UserRole, metadata)
                self.ui.imagesListWidget.addItem(item)
                
                # Pre-generate and cache thumbnail
                self._cache_thumbnail(image_path)
                total_images += 1
        
        self.log_info(f"[Browse] Loaded {total_images} images in {len(image_groups)} groups")

    def _cache_thumbnail(self, image_path):
        """Generate and cache thumbnail for an image."""
        arr = self.load_thumbnail_array(image_path, max_size=(500, 500))
        if arr is not None:
            arr_uint8 = (arr * 255).astype(np.uint8) 
            h, w, c = arr_uint8.shape
            img = QImage(arr_uint8.data, w, h, w * c, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(img)
            self.thumbnail_cache[image_path] = {'pixmap': pixmap, 'array': arr}
        else:
            self.thumbnail_cache[image_path] = None

    def browse_output_directory(self):
        """
        Open a directory dialog to select where processed images will be saved.
        Updates the output‐directory line edit and persists the choice in settings.
        """
        default = self.ui.outputDirectoryLineEdit.text() or os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, 'Select Output Directory', default)
        if folder:
            self.ui.outputDirectoryLineEdit.setText(folder)
            self.settings.setValue('outputFolder', folder)
            # self.log_debug(f"[Browse] Output folder set to {folder}")

    def _update_format_controls(self, ext: str):
        """
        Show the bit-depth frame only when TIFF is selected;
        show the JPEG-quality frame only when JPEG is selected.
        """
        fmt = ext.lower()
        # Show bit-depth options for TIFF
        self.ui.bitDepthFrame.setVisible(fmt == '.tiff')
        # Show JPEG-quality options for JPEG
        self.ui.jpgQualityFrame.setVisible(fmt in ('.jpg', '.jpeg'))
        # show exr options
        self.ui.exrOptionsFrame.setVisible(fmt == '.exr')

    def set_selected_as_chart(self, detect=True):
        """
        Mark the currently selected image in the list as the reference colour chart.
        Clears any prior chart flags, sets the 'chart' metadata on the new selection,
        updates the chart‐path line edit, enables manual selection, and triggers detection.
        """
        selected_item = self.ui.imagesListWidget.currentItem()
        if not selected_item:
            return
            
        selected_meta = selected_item.data(Qt.UserRole)
        
        # Don't allow group headers to be set as chart
        if selected_meta.get('is_group_header', False):
            self.log_debug("[Select] Cannot set group header as chart")
            return
        
        selected_group = selected_meta.get('group_name', 'All Images')
        chart_path = selected_meta['input_path']
        
        # Remove any existing chart assignment for this group
        existing_chart = None
        for chart_info in self.chart_groups:
            if chart_info[1] == selected_group:  # Same group
                existing_chart = chart_info[0]
                break
        
        # Clear the existing chart flag from the previous chart in this group
        if existing_chart:
            for i in range(self.ui.imagesListWidget.count()):
                item = self.ui.imagesListWidget.item(i)
                meta = item.data(Qt.UserRole)
                
                # Skip group headers
                if meta.get('is_group_header', False):
                    continue
                
                # Clear chart flag from previous chart in this group
                if meta.get('input_path') == existing_chart and meta.get('group_name', 'All Images') == selected_group:
                    meta['chart'] = False
                    meta.pop('debug_images', None)
                    item.setData(Qt.UserRole, meta)
                    item.setBackground(QColor('white'))
                    break
            
            # Remove from chart_groups list
            self.chart_groups = [[c, g] for c, g in self.chart_groups if not (c == existing_chart and g == selected_group)]

        # Set new chart for this group
        selected_meta['chart'] = True
        group_name = selected_group
        
        # Add to chart_groups tracking
        self.chart_groups.append([chart_path, group_name])
        
        # Update UI to show group context
        self.ui.chartPathLineEdit.setText(f"[{group_name}] {os.path.basename(chart_path)}")
        self.log_info(f"[Select] Chart set to {os.path.basename(chart_path)} for group '{group_name}'")

        # Store group information for calibration
        self.current_chart_group = group_name
        if not self.manual_selection_mode:
            self.detect_chart(input_source=chart_path, is_npy=False)

    @ChartTools.exit_manual_mode
    def preview_selected(self):
        """
        Show a preview of the currently selected image with real-time adjustment support.
        Loads RAW files in background for real-time editing when input images are selected.
        """
        item = self.ui.imagesListWidget.currentItem()
        if not item:
            self.show_debug_frame(False)
            self._clear_preview_state()
            return

        meta = item.data(Qt.UserRole)

        # Skip group headers
        if meta.get('is_group_header', False):
            return

        path_to_show = meta.get('output_path') or meta['input_path']
        input_path = meta.get('input_path')

        found_image = False

        if path_to_show:
            # Debug logging for path selection
            output_path = meta.get('output_path')
            status = meta.get('status', 'not processed')
            self.log_debug(f"[Preview Debug] Status: {status}, Input: {os.path.basename(input_path) if input_path else 'None'}, Output: {os.path.basename(output_path) if output_path else 'None'}, Showing: {os.path.basename(path_to_show)}")

            # Check if we should load RAW for real-time editing
            if (input_path and
                input_path.lower().endswith(('.nef', '.cr2', '.cr3', '.dng', '.arw', '.raw')) and
                input_path != self.current_raw_path and
                not output_path):  # Only for unprocessed RAW files

                # Show thumbnail immediately for instant feedback
                self.preview_thumbnail(path_to_show)

                # Set up delayed loading with race condition protection
                previous_pending = self.pending_raw_path
                self.pending_raw_path = input_path

                # If this is the same path as what was already pending, extend the timer
                if previous_pending == input_path and self.raw_load_timer.isActive():
                    self.log_debug(f"[Preview] waiting for image to load: {os.path.basename(input_path)}")
                else:
                    # Cancel previous timer and start new one
                    self.raw_load_timer.stop()
                    if self.is_loading_raw:
                        self.log_debug("[Preview] Cancelling previous RAW load due to image change")
                        # Note: The actual loading cancellation is handled in _load_raw_for_preview

                    self.log_debug(f"[Preview] Scheduling RAW load for: {os.path.basename(input_path)} (100ms delay)")
                    self.raw_load_timer.start(100)

            else:
                # Cancel any pending RAW load
                self.raw_load_timer.stop()
                self.pending_raw_path = None

                # Clear RAW state and show regular thumbnail
                self._clear_preview_state()
                self.preview_thumbnail(path_to_show)


        if self.ui.displayDebugExposureDataCheckBox.isChecked():
            self.show_exposure_debug_overlay()

        self.file_naming_schema.update_preview_for_selection_change(self)

    def _clear_preview_state(self):
        """Clear current RAW preview state and invalidate cache."""
        self.current_raw_path = None
        self.current_raw_array = None
        self.current_preview_array = None
        self.original_white_balance = 5500
        self.is_loading_raw = False

        # Hide loading spinner
        self.hide_loading_spinner()

        # Clear preview state

    def _load_raw_for_preview(self, raw_path: str):
        """
        Load RAW file in background for real-time preview editing.

        Args:
            raw_path: Path to the RAW image file
        """
        # Check if we should load this RAW file
        if self.is_loading_raw:
            self.log_debug(f"[Preview] Already loading RAW, skipping: {os.path.basename(raw_path)}")
            return

        # Verify this is still the pending path (not cancelled by user scrolling)
        if raw_path != self.pending_raw_path and self.pending_raw_path is not None:
            self.log_debug(f"[Preview] RAW path changed during delay, skipping: {os.path.basename(raw_path)}")
            return

        self.is_loading_raw = True
        self.current_raw_path = raw_path

        # Show loading spinner
        self.show_loading_spinner()
        self.log_debug(f"[Preview] Starting RAW load: {os.path.basename(raw_path)}")

        # Create and start RAW loading worker
        worker = RawLoadWorker(raw_path)
        worker.signals.loaded.connect(self._on_raw_loaded)
        worker.signals.error.connect(self._on_raw_load_error)

        self.threadpool.start(worker)

    def _on_raw_loaded(self, raw_array: np.ndarray):
        """
        Handle successful RAW loading.

        Args:
            raw_array: Loaded RAW image as float32 array
        """
        self.is_loading_raw = False

        # Hide loading spinner
        self.hide_loading_spinner()

        try:
            # Check if this RAW load is still relevant (user might have changed selection)
            if self.current_raw_path != self.pending_raw_path and self.pending_raw_path is not None:
                self.log_debug(f"[Preview] RAW loaded but selection changed, discarding result: {os.path.basename(self.current_raw_path)}")
                return

            # Validate the loaded RAW array
            if raw_array is None or raw_array.size == 0:
                self.log_debug(f"[Preview] RAW array is invalid, cannot process")
                return

            if len(raw_array.shape) != 3 or raw_array.shape[2] != 3:
                self.log_debug(f"[Preview] RAW array has unexpected shape: {raw_array.shape}")
                return

            # Store the RAW array
            self.current_raw_array = raw_array

            # Create preview-sized version (max 4MP)
            preview_array = self._resize_for_preview(raw_array)
            if preview_array is None or preview_array.size == 0:
                self.log_debug(f"[Preview] Preview resize failed")
                return

            # Extract white balance from EXIF with fallback
            try:
                wb_temp = self._extract_white_balance_from_exif(self.current_raw_path)
                self.original_white_balance = wb_temp if wb_temp is not None else 5500.0
            except Exception as e:
                self.log_debug(f"[Preview] White balance extraction failed: {e}")
                self.original_white_balance = 5500.0

            # Clear any stale preview state

            # Only assign preview array after all validations pass
            self.current_preview_array = preview_array

            self.log_debug(f"[Preview] RAW loaded: {raw_array.shape}, Preview: {self.current_preview_array.shape}, WB: {self.original_white_balance}K")

            # Apply current adjustments and display (force recalculation for new image)
            self._update_preview_display(force_recalculate=True)

        except Exception as e:
            self.log_debug(f"[Preview] Error processing loaded RAW: {e}")
            # Reset preview state on error
            self.current_preview_array = None

    def _on_raw_load_error(self, error_msg: str):
        """
        Handle RAW loading error.

        Args:
            error_msg: Error message from RAW loading
        """
        self.is_loading_raw = False

        # Hide loading spinner
        self.hide_loading_spinner()

        self.log_error(f"[Preview] Failed to load RAW: {error_msg}")

        # Fall back to regular thumbnail
        if self.current_raw_path:
            self.preview_thumbnail(self.current_raw_path)

    def _on_raw_load_timer(self):
        """
        Timer callback to load RAW file after delay.
        This prevents loading RAW files when user is quickly scrolling through images.
        """
        if self.pending_raw_path and not self.is_loading_raw:
            self.log_debug(f"[Preview] Timer fired - loading RAW: {os.path.basename(self.pending_raw_path)}")
            self._load_raw_for_preview(self.pending_raw_path)
            self.pending_raw_path = None
        else:
            self.log_debug("[Preview] Timer fired but no pending RAW path or already loading")

    def _resize_for_preview(self, img_array: np.ndarray) -> np.ndarray:
        """
        Resize image array to maximum 4MP for preview performance.

        Args:
            img_array: Full resolution image array

        Returns:
            np.ndarray: Resized preview array
        """
        h, w = img_array.shape[:2]
        current_pixels = h * w
        max_preview_pixels = 4_000_000  # 4MP max

        # Early exit if already small enough
        if current_pixels <= max_preview_pixels:
            return img_array

        # Calculate target dimensions maintaining aspect ratio
        scale_factor = np.sqrt(max_preview_pixels / current_pixels)
        new_w = max(1, int(w * scale_factor))
        new_h = max(1, int(h * scale_factor))

        # Resize using INTER_AREA for best downsampling quality
        resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

        self.log_debug(f"[Preview] Resize: {w}x{h} → {new_w}x{new_h} ({(new_w*new_h)/1e6:.1f}MP)")
        return resized

    def _update_preview_display(self, force_recalculate=False):
        """Apply current adjustment settings and update the preview display."""
        # Validate preview array state
        if self.current_preview_array is None:
            self.log_debug("[Preview] No preview array available for adjustment")
            return

        # Additional safety checks for array integrity
        if (self.current_preview_array.size == 0 or
            len(self.current_preview_array.shape) != 3 or
            self.current_preview_array.shape[2] != 3):
            self.log_debug(f"[Preview] Invalid preview array shape: {self.current_preview_array.shape}")
            return

        # Ensure white balance is valid
        if self.original_white_balance is None:
            self.original_white_balance = 5500.0

        # Get current adjustment values from UI
        exposure = self.ui.exposureAdjustmentDoubleSpinBox.value()
        shadows = self.ui.shadowAdjustmentDoubleSpinBox.value()
        highlights = self.ui.highlightAdjustmentDoubleSpinBox.value()
        target_wb = self.ui.whitebalanceSpinbox.value() if self.ui.enableWhiteBalanceCheckBox.isChecked() else self.original_white_balance
        denoise_strength = self.ui.denoiseDoubleSpinBox.value() if self.ui.denoiseImageCheckBox.isChecked() else 0.0
        sharpen_amount = self.ui.sharpenDoubleSpinBox.value() if self.ui.sharpenImageCheckBox.isChecked() else 0.0

        # Apply adjustments using editing tools
        adjusted_array = apply_all_adjustments(
            self.current_preview_array,
            exposure=exposure,
            shadows=shadows,
            highlights=highlights,
            current_wb=self.original_white_balance,
            target_wb=target_wb,
            wb_tint=0.0,
            denoise_strength=denoise_strength,
            sharpen_amount=sharpen_amount,
            sharpen_radius=1.0,  # Default radius - could be made configurable later
            sharpen_threshold=0.0  # Default threshold - could be made configurable later
        )

        # Convert to QPixmap and display
        pixmap = self._array_to_pixmap(adjusted_array)

        # Check if pixmap conversion was successful
        if pixmap is None or pixmap.isNull():
            self.log_debug(f"[Preview] ❌ Pixmap conversion failed, cannot display preview")
            return

        self._display_preview(pixmap)

    def _array_to_pixmap(self, img_array: np.ndarray) -> QPixmap:
        """
        Convert numpy array to QPixmap for display.

        Args:
            img_array: Float32 image array (0-1 range)

        Returns:
            QPixmap: Converted pixmap ready for display
        """
        # Performance optimization: Check if we can use faster conversion
        total_pixels = img_array.shape[0] * img_array.shape[1]

        if total_pixels <= 500_000:  # 0.5MP or smaller - use fastest conversion
            # Skip gamma correction for ultra-small preview images
            uint8_array = np.uint8(255 * np.clip(img_array, 0, 1))
        else:
            # Apply gamma correction and convert to uint8
            gamma_corrected = colour.cctf_encoding(np.clip(img_array, 0, 1))
            uint8_array = np.uint8(255 * gamma_corrected)

        # Ensure contiguous memory layout for fastest QImage creation
        if not uint8_array.flags['C_CONTIGUOUS']:
            uint8_array = np.ascontiguousarray(uint8_array)

        # Convert to QImage
        h, w, c = uint8_array.shape
        bytes_per_line = w * c
        qimg = QImage(uint8_array.data, w, h, bytes_per_line, QImage.Format_RGB888)

        return QPixmap.fromImage(qimg)


    def _extract_white_balance_from_exif(self, img_path: str) -> float:
        """
        Extract white balance color temperature from EXIF metadata.

        Args:
            img_path: Path to the image file

        Returns:
            float: White balance temperature in Kelvin (default 5500K if not found)
        """
        default_wb = 5500.0  # Default daylight temperature

        try:
            # Method 1: Try to get white balance from rawpy (most accurate for RAW files)
            if img_path.lower().endswith(('.nef', '.cr2', '.cr3', '.dng', '.arw', '.raw')):
                wb_temp = self._get_wb_from_rawpy(img_path)
                if wb_temp is not None:
                    self.log_debug(f"[WB] Found white balance from rawpy: {wb_temp}K")
                    return wb_temp

            # Method 2: Try to get white balance from EXIF using OpenImageIO
            wb_temp = self._get_wb_from_oiio_exif(img_path)
            if wb_temp is not None:
                self.log_debug(f"[WB] Found white balance from EXIF: {wb_temp}K")
                return wb_temp

        except Exception as e:
            self.log_debug(f"[WB] Error reading white balance from {img_path}: {e}")

        self.log_debug(f"[WB] Using default white balance: {default_wb}K")
        return default_wb

    def _get_wb_from_rawpy(self, img_path: str) -> float:
        """
        Extract white balance from RAW file using rawpy.

        Args:
            img_path: Path to RAW image

        Returns:
            float: White balance temperature in Kelvin, or None if not available
        """
        try:
            import rawpy
            with rawpy.imread(img_path) as raw:
                # Try to get white balance multipliers
                wb_coeffs = raw.camera_whitebalance
                if wb_coeffs is not None and len(wb_coeffs) >= 3:
                    # Convert RGB multipliers to approximate color temperature
                    r_mult, g_mult, b_mult = wb_coeffs[0], wb_coeffs[1], wb_coeffs[2]

                    # Normalize to green
                    if g_mult > 0:
                        r_ratio = r_mult / g_mult
                        b_ratio = b_mult / g_mult

                        # Estimate temperature from red/blue ratio
                        rb_ratio = r_ratio / (b_ratio + 1e-10)

                        # Empirical formula for RGB multipliers to temperature conversion
                        if rb_ratio > 1.5:
                            # Very warm light
                            temp = 2000 + (rb_ratio - 1.5) * 1000
                        elif rb_ratio > 1.0:
                            # Warm to neutral light
                            temp = 3000 + (rb_ratio - 1.0) * 4000
                        else:
                            # Cool light
                            temp = 5000 + (1.0 - rb_ratio) * 5000

                        # Clamp to reasonable range
                        temp = max(2000, min(12000, temp))
                        return float(temp)

        except Exception as e:
            self.log_debug(f"[WB] Error reading rawpy white balance: {e}")

        return None

    def _get_wb_from_oiio_exif(self, img_path: str) -> float:
        """
        Extract white balance from EXIF metadata using OpenImageIO.

        Args:
            img_path: Path to image file

        Returns:
            float: White balance temperature in Kelvin, or None if not available
        """
        try:
            from OpenImageIO import ImageInput

            img_input = ImageInput.open(img_path)
            if not img_input:
                return None

            spec = img_input.spec()
            img_input.close()

            # Try various EXIF white balance fields
            wb_fields = [
                'Exif:ColorTemperature',
                'Exif:WhiteBalance',
                'EXIF:ColorTemperature',
                'EXIF:WhiteBalance',
                'ColorTemperature',
                'WhiteBalance'
            ]

            for field in wb_fields:
                if hasattr(spec, 'getattribute'):
                    try:
                        value = spec.getattribute(field)
                        if value is not None:
                            # Try to convert to temperature
                            if isinstance(value, (int, float)):
                                temp = float(value)
                                if 1000 <= temp <= 12000:  # Reasonable temperature range
                                    return temp
                            elif isinstance(value, str):
                                # Try to parse string value
                                try:
                                    temp = float(value.replace('K', '').replace('k', '').strip())
                                    if 1000 <= temp <= 12000:
                                        return temp
                                except:
                                    pass
                    except:
                        continue

        except Exception as e:
            self.log_debug(f"[WB] Error reading OIIO EXIF white balance: {e}")

        return None

    def preview_thumbnail(self, path):
        """Load and display thumbnail using ImageLoader utility."""
        # Get exposure factor for this path
        exposure_factor = 1.0
        for i in range(self.ui.imagesListWidget.count()):
            item = self.ui.imagesListWidget.item(i)
            meta = item.data(Qt.UserRole)
            if meta.get('input_path') == path or meta.get('output_path') == path:
                exposure_factor = meta.get('average_exposure', 1.0)
                break

        # Load pixmap
        self.log_debug(f"[Preview Thumbnail] Loading image from: {path}, exists: {os.path.exists(path)}")
        pixmap = ImageLoader.create_pixmap_from_path(
            path, cache=self.thumbnail_cache,
            chart_swatches=self.chart_swatches,
            correct_thumbnails=self.correct_thumbnails,
            log_callback=self.log_debug
        )

        if not pixmap or pixmap.isNull():
            self.log_error(f"[Preview Thumbnail] ❌ Failed to load image for preview from {path}")
            self.log_debug(f"[Preview Thumbnail] Pixmap is None: {pixmap is None}, Pixmap isNull: {pixmap.isNull() if pixmap else 'N/A'}")
            return
        else:
            self.log_debug(f"[Preview Thumbnail] ✅ Successfully loaded pixmap from {path}, size: {pixmap.width()}x{pixmap.height()}")

        # Apply exposure brightness if needed
        if exposure_factor != 1.0:
            pixmap = self._adjust_pixmap_brightness(pixmap, exposure_factor)

        # Show in preview scene
        self.previewScene.clear()
        self.previewScene.addItem(QGraphicsPixmapItem(pixmap))
        self.ui.imagePreviewGraphicsView.resetTransform()
        self.ui.imagePreviewGraphicsView.fitInView(
            self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio
        )
        self.current_image_pixmap = pixmap

    @ChartTools.exit_manual_mode
    def start_processing(self):
        """
        Launch background workers to process every image in the list;
        switches the UI into 'processing' mode and shows the progress frame.
        """
        self.active_workers.clear()
        inf = self.ui.rawImagesDirectoryLineEdit.text().strip()
        outf = self.ui.outputDirectoryLineEdit.text().strip() or os.getcwd()
        thr = self.ui.imageProcessingThreadsSpinbox.value()
        qual = self.ui.jpegQualitySpinbox_2.value()

        export_masked = self.ui.exportMaskedImagesCheckBox.isChecked()
        output_ext = self.ui.imageFormatComboBox.currentText().lower()
        
        # Load export schema settings from settings dialog
        custom_name = self.ui.newImageNameLineEdit.text()
        root_folder = os.path.basename(inf) if inf else ""

        if output_ext == '.tiff':
            tiff_bits = 16 if self.ui.sixteenBitRadioButton.isChecked() else 8
        else:
            tiff_bits = None

        # Organize images by group and check calibrations
        image_groups = {}  # {group_name: [image_paths]}
        missing_calibrations = []
        
        for i in range(self.ui.imagesListWidget.count()):
            item = self.ui.imagesListWidget.item(i)
            metadata = item.data(Qt.UserRole)
            
            # Skip group headers
            if metadata.get('is_group_header', False):
                continue
            
            group_name = metadata.get('group_name', 'All Images')
            image_path = metadata['input_path']
            
            # Initialize group list if needed
            if group_name not in image_groups:
                image_groups[group_name] = []
            image_groups[group_name].append(image_path)
            
            # Check if this group has a chart assigned
            group_chart = None
            for chart_path, chart_group in self.chart_groups:
                if chart_group == group_name:
                    group_chart = chart_path
                    break
            
            # Check if this group has calibration data
            if group_name not in self.group_calibrations:
                # If no calibration but has chart, try to extract from chart
                if group_chart:
                    self.log_debug(f"[Process] No calibration found for group '{group_name}', will attempt to extract from chart")
                # Fall back to default calibration if available
                elif self.calibration_file and os.path.exists(self.calibration_file):
                    self.group_calibrations[group_name] = {
                        'file': self.calibration_file,
                        'swatches': self.chart_swatches
                    }
                    self.log_debug(f"[Process] Using default calibration for group '{group_name}'")
                else:
                    if group_name not in missing_calibrations:
                        missing_calibrations.append(group_name)
        
        # Extract calibrations from charts where needed
        for group_name in list(missing_calibrations):
            group_chart = None
            for chart_path, chart_group in self.chart_groups:
                if chart_group == group_name:
                    group_chart = chart_path
                    break
            
            if group_chart:
                # Try to extract calibration from the chart
                try:
                    self.current_chart_group = group_name
                    swatches, calib_file = self.extract_chart_swatches(group_chart)
                    if swatches is not None:
                        self.group_calibrations[group_name] = {
                            'file': calib_file,
                            'swatches': swatches
                        }
                        self.log_debug(f"[Process] Extracted calibration for group '{group_name}' from chart")
                        missing_calibrations.remove(group_name)
                    else:
                        self.log_debug(f"[Process] Failed to extract calibration from chart for group '{group_name}'")
                except Exception as e:
                    self.log_error(f"[Process] Error extracting calibration for group '{group_name}': {e}")
        
        if missing_calibrations and not self.ui.dontUseColourChartCheckBox.isChecked():
            self.log_info(f"❌ Missing calibrations for groups: {', '.join(missing_calibrations)}")
            self.log_info("❗ Please select and detect charts for all groups, or set a default calibration")
            return
        
        if not image_groups:
            self.log_info("❗ No RAW images found in the list.")
            return
        
        # Update metadata with group-specific calibration
        for i in range(self.ui.imagesListWidget.count()):
            item = self.ui.imagesListWidget.item(i)
            metadata = item.data(Qt.UserRole)
            
            if metadata.get('is_group_header', False):
                continue
                
            group_name = metadata.get('group_name', 'All Images')
            if not self.ui.dontUseColourChartCheckBox.isChecked():
                calibration = self.group_calibrations[group_name]
                metadata['calibration'] = calibration['file']
            item.setData(Qt.UserRole, metadata)

        # Flatten all images from all groups for processing while preserving original order
        # Iterate through the list widget to maintain the order images were loaded in
        all_images = []
        for i in range(self.ui.imagesListWidget.count()):
            item = self.ui.imagesListWidget.item(i)
            metadata = item.data(Qt.UserRole)
            
            # Skip group headers
            if metadata.get('is_group_header', False):
                continue
                
            image_path = metadata['input_path']
            all_images.append(image_path)
            
        self.total_images = len(all_images)
        self.finished_images = 0
        self.global_start = time.time()
        
        # Log processing summary
        group_summary = ", ".join([f"{name}: {len(images)}" for name, images in image_groups.items()])
        self.log_info(f"[Process] Processing {self.total_images} images across {len(image_groups)} groups ({group_summary})")
        self.log_info(f"[Process] Dispatching across {thr} parallel threads")

        # Show and initialize the status bar
        self.ui.processingStatusBarFrame.setVisible(True)
        self.ui.processingStatusProgressBar.setRange(0, self.total_images)
        self.ui.processingStatusProgressBar.setValue(0)
        self.ui.processingStatusProgressBar.setFormat(
            f"0 out of {self.total_images} images processed"
        )

        # Create rename map with numbering reset for each output directory
        rename_map = {}
        if self.use_export_schema and self.export_schema:
            # Group images by their output directory based on export schema
            output_dir_groups = {}  # {output_dir: [image_paths]}
            
            for group_name, group_images in image_groups.items():
                for image_path in group_images:
                    try:
                        # Get export schema settings
                        custom_name = self.ui.newImageNameLineEdit.text() if self.ui.newImageNameLineEdit.text() else ""
                        root_folder = os.path.basename(inf) if inf else ""
                        
                        # Apply naming schema with dummy image number (1) to get directory structure
                        from ImageProcessor.fileNamingSchema import apply_naming_schema
                        dummy_output_path = apply_naming_schema(
                            schema=self.export_schema,
                            input_path=image_path,
                            output_base_dir=outf,
                            custom_name=custom_name,
                            image_number=1,  # Dummy number, we only care about directory
                            output_extension=output_ext,
                            root_folder=root_folder,
                            group_name=group_name
                        )
                        
                        # Get the directory part of the output path
                        output_dir = os.path.dirname(dummy_output_path)
                        
                        # Group images by output directory
                        if output_dir not in output_dir_groups:
                            output_dir_groups[output_dir] = []
                        output_dir_groups[output_dir].append(image_path)
                        
                    except Exception as e:
                        self.log_error(f"[Rename Map] Error applying schema for {image_path}: {e}")
                        # Fallback to original group-based numbering for this image
                        if image_path not in rename_map:
                            rename_map[image_path] = 1
            
            # Create numbering within each output directory
            for output_dir, dir_images in output_dir_groups.items():
                for idx, image_path in enumerate(dir_images):
                    rename_map[image_path] = idx + 1
            
            # Log the rename mapping for debugging
            self.log_debug(f"[Process] Rename map created with per-output-directory numbering:")
            for output_dir, dir_images in output_dir_groups.items():
                dir_numbers = [rename_map[path] for path in dir_images]
                rel_dir = os.path.relpath(output_dir, outf) if output_dir.startswith(outf) else output_dir
                self.log_debug(f"  {rel_dir}: {len(dir_images)} images (numbers: {min(dir_numbers)}-{max(dir_numbers)})")
        else:
            # Fallback to original group-based numbering if no export schema
            for group_name, group_images in image_groups.items():
                # Number each group starting from 1
                for idx, image_path in enumerate(group_images):
                    rename_map[image_path] = idx + 1
            
            # Log the rename mapping for debugging
            self.log_debug(f"[Process] Rename map created with per-group numbering (no schema):")
            for group_name, group_images in image_groups.items():
                group_numbers = [rename_map[path] for path in group_images]
                self.log_debug(f"  {group_name}: {len(group_images)} images (numbers: {min(group_numbers)}-{max(group_numbers)})")
        name_base = self.ui.newImageNameLineEdit.text().strip()
        padding = self.ui.imagePaddingSpinBox.value()

        exr_cs = self.ui.exrColourSpaceComboBox.currentText()

        use_original_filenames = False # Temporary bool to block depricated feature

        # Process each group with its specific calibration (local processing)
        for group_name, group_images in image_groups.items():
            if self.ui.dontUseColourChartCheckBox.isChecked():
                calibration = None
                swatches = None
            else:
                calibration = self.group_calibrations[group_name]
                swatches = calibration['swatches']
            
            # Dispatch group images in batches
            size = max(1, len(group_images) // thr)  # Rough batching
            for i in range(0, len(group_images), size):
                chunk = group_images[i: i + size]
                
                sig = WorkerSignals()
                sig.log.connect(self.log)
                sig.preview.connect(self._from_worker_preview)
                sig.status.connect(self.update_image_status)
                
                # Create metadata map for this worker
                image_metadata_map = {
                    item.data(Qt.UserRole)['input_path']: item.data(Qt.UserRole)
                    for i in range(self.ui.imagesListWidget.count())
                    for item in [self.ui.imagesListWidget.item(i)]
                    if not item.data(Qt.UserRole).get('is_group_header', False)
                }

                exr_cs = None
                if output_ext == '.exr':
                    exr_cs = self.ui.exrColourSpaceComboBox.currentText()

                # Check if we should use chart-based correction or manual adjustments
                use_chart = True if swatches is not None else False
                
                # Get manual adjustment values from UI (only used when use_chart=False)
                exposure_adj = self.ui.exposureAdjustmentDoubleSpinBox.value()
                shadow_adj = self.ui.shadowAdjustmentDoubleSpinBox.value()
                highlight_adj = self.ui.highlightAdjustmentDoubleSpinBox.value()
                white_balance_adj = int(self.ui.whitebalanceSpinbox.value()) if self.ui.enableWhiteBalanceCheckBox.isChecked() else 5500
                denoise_strength = self.ui.denoiseDoubleSpinBox.value() if self.ui.denoiseImageCheckBox.isChecked() else 0.0
                sharpen_amount = self.ui.sharpenDoubleSpinBox.value() if self.ui.sharpenImageCheckBox.isChecked() else 0.0
                
                # Get downsample value from rescaling controls
                downsample_percentage = 100.0  # Default to no downsampling
                if hasattr(self.ui, 'enableRescaleCheckBox') and self.ui.enableRescaleCheckBox.isChecked():
                    if hasattr(self.ui, 'targetDownsamplePercentDoubleSpinbox'):
                        downsample_percentage = self.ui.targetDownsamplePercentDoubleSpinbox.value()
                
                worker = ImageCorrectionWorker(
                    chunk, swatches, outf, sig, qual, rename_map,
                    name_base=name_base, padding=padding,
                    export_masked=export_masked, output_format=output_ext, tiff_bitdepth=tiff_bits,
                    exr_colorspace=exr_cs, export_schema=self.export_schema,
                    use_export_schema=self.use_export_schema, custom_name=custom_name,
                    root_folder=root_folder, group_name=group_name,
                    use_chart=use_chart, exposure_adj=exposure_adj,
                    shadow_adj=shadow_adj, highlight_adj=highlight_adj,
                    white_balance_adj=white_balance_adj, denoise_strength=denoise_strength,
                    sharpen_amount=sharpen_amount, downsample_percentage=downsample_percentage
                )

                shadow_limit = self.ui.shadowLimitSpinBox.value() / 100.0
                highlight_limit = self.ui.highlightLimitSpinBox.value() / 100.0
                worker.shadow_limit = shadow_limit
                worker.highlight_limit = highlight_limit
                worker.denoise_strength = denoise_strength  # Add denoise strength
                worker.use_original_filenames = use_original_filenames
                worker.image_metadata_map = image_metadata_map
                
                self.log_debug(f"[Process] Starting worker for group '{group_name}' - batch {i//size + 1} ({len(chunk)} images)")
                self.active_workers.append(worker)
                self.threadpool.start(worker)
    
    def _unc_to_local_path(self, unc_path):
        """
        Convert UNC path back to local path for UI matching.
        
        Args:
            unc_path: UNC network path (e.g., \\\\server\\share\\folder\\file.txt)
            
        Returns:
            str: Local path that matches what's stored in the UI
        """
        try:
            # Import here to avoid circular import
            from ImageProcessor.networkProcessor import convert_to_unc_path
            
            # Simple approach: if it's already a local path, return as-is
            if not unc_path.startswith('\\\\'):
                return unc_path
            
            # For UNC paths, we need to find the corresponding local path
            # This is a simplified approach - in practice you might need more sophisticated mapping
            # For now, just use the original image paths stored in the UI
            for i in range(self.ui.imagesListWidget.count()):
                item = self.ui.imagesListWidget.item(i)
                metadata = item.data(Qt.UserRole)
                
                # Skip group headers
                if metadata.get('is_group_header', False):
                    continue
                
                local_path = metadata.get('input_path')
                if local_path and convert_to_unc_path(local_path) == unc_path:
                    return local_path
            
            # Fallback: return the UNC path as-is
            return unc_path
            
        except Exception as e:
            self.log_error(f"[Network] Error converting UNC path: {e}")
            return unc_path
    
    def _reset_processing_ui(self):
        """Reset the processing UI state (called from main thread)."""
        self.processing_active = False
        self.ui.processImagesPushbutton.setText("Process Images")
        self.ui.processImagesPushbutton.setStyleSheet("")
        self.ui.processingStatusBarFrame.setVisible(False)

    def cancel_processing(self):
        """
        Signal all active workers to cancel, reset the processing UI,
        and revert the 'Process' button to its original state.
        """
        self.threadpool.clear()
        for worker in self.active_workers:
            if hasattr(worker, "cancel"):
                worker.cancel()
        self.active_workers.clear()
        self.processing_active = False
        self.ui.processImagesPushbutton.setText("Process Images")
        self.ui.processImagesPushbutton.setStyleSheet("")
        self.ui.processingStatusBarFrame.setVisible(False)
        self.log_info("[Processing] All processing cancelled by user.")

    def update_image_status(self, image_path, status, processing_time, output_path):
        """
        Slot to update the background color of the corresponding QListWidgetItem
        based on the processing status for each image.
        """
        # Iterate over list items to find the matching filename
        for i in range(self.ui.imagesListWidget.count()):
            item = self.ui.imagesListWidget.item(i)
            metadata = item.data(Qt.UserRole)
            
            # Skip group headers
            if metadata.get('is_group_header', False):
                continue
                
            if metadata.get('input_path') == image_path:
                metadata['status'] = status
                metadata['processing_time'] = processing_time
                metadata['output_path'] = output_path
                
                # Debug logging for network output path updates
                self.log_debug(f"[Update Status] Image: {os.path.basename(image_path)}, Status: {status}")
                self.log_debug(f"[Update Status] Output path set to: {output_path}")
                self.log_debug(f"[Update Status] Output file exists: {os.path.exists(output_path) if output_path else 'No path'}")
                # Choose color by status
                if status == 'started':
                    bg = QColor('#FFA500')  # light orange
                elif status == 'finished':
                    bg = QColor('#4caf50')  # green
                elif status == 'error':
                    bg = QColor('#f44336')  # red
                elif status == 'sent_to_network':
                    bg = QColor('#a682fa')
                else:
                    bg = QColor('white')
                item.setBackground(bg)
                item.setData(Qt.UserRole, metadata)
                break

        if status in ('finished', 'error'):
            self.finished_images += 1

            # Update status bar
            self.ui.processingStatusProgressBar.setValue(self.finished_images)
            self.ui.processingStatusProgressBar.setFormat(
                f"{self.finished_images} out of {self.total_images} images processed"
            )

            if self.finished_images >= self.total_images and self.global_start:
                total = time.time() - self.global_start
                self.log_info(f"[Timing] {self.total_images} images processed in {total:.2f}s")
                self.processing_complete()

    def processing_complete(self):
        """
        Called when all images are finished (or cancelled).
        Hides the progress frame, resets the 'Process' button, and logs completion.
        """
        self.processing_active = False
        self.ui.processingStatusBarFrame.setVisible(False)
        self.ui.processImagesPushbutton.setText("Process Images")
        self.ui.processImagesPushbutton.setStyleSheet("")
        self.log_info("[Processing] All processing complete.")

    def _from_worker_preview(self, data):
        """
        Displays a provided npy image array from the image processing worker
        Called whenever an image finishes processing
        Does not show images from networked processing clients
        """

        image_path = data

        if image_path is not None:
            for i in range(self.ui.imagesListWidget.count()):
                item = self.ui.imagesListWidget.item(i)
                meta = item.data(Qt.UserRole)

                if meta.get('input_path') == image_path or meta.get('output_path') == image_path:
                    self.ui.imagesListWidget.setCurrentRow(i)
                    break

        self.preview_selected()

    def _display_preview(self, pixmap):
        """
        Clear the preview scene and display the provided pixmap.
        """
        if not pixmap or pixmap.isNull():
            return
            
        self.current_image_pixmap = pixmap
        self.previewScene.clear()
        pixmap_item = QGraphicsPixmapItem(pixmap)
        self.previewScene.addItem(pixmap_item)
        
        self.ui.imagePreviewGraphicsView.resetTransform()
        self.ui.imagePreviewGraphicsView.fitInView(
            self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio)


    def update_system_usage(self):
        """
        Updates the sytem usage progress bars with current values based on the QTimer created in __init__
        """
        cpu = psutil.cpu_percent(None)
        self.ui.cpuUsageProgressBar.setValue(cpu)
        # show text on progress bar

        self.ui.cpuUsageProgressBar.setTextVisible(True)
        self.ui.cpuUsageProgressBar.setStyleSheet(self.get_usage_style(cpu))
        mem = psutil.virtual_memory().percent
        self.ui.memoryUsageProgressBar.setValue(mem)
        # show text on progress bar
        self.ui.memoryUsageProgressBar.setTextVisible(True)
        self.ui.memoryUsageProgressBar.setStyleSheet(self.get_usage_style(mem))

        # show or clear high-memory warning
        if mem > 95:
            self.memoryWarningLabel.setText(
                "Memory usage too high, Lowering Image Threads may produce faster results"
            )
        else:
            self.memoryWarningLabel.setText('')

    def get_usage_style(self, pct):
        """
        sets the stylesheet for the usage bar
        Uses colours depending on the percentage
        """
        if pct < 50:
            c = "#4caf50"
        elif pct < 80:
            c = "#ffc107"
        else:
            c = "#f44336"
        return f"QProgressBar{{border:1px solid #bbb;border-radius:5px;text-align:center}}QProgressBar::chunk{{background-color:{c};width:1px}}"

    def load_raw_image(self, path: str) -> np.ndarray | None:
        """Load RAW image using ImageLoader utility."""
        ImageLoader.load_thumbnail_array(path)
        return ImageLoader.load_raw_image_sync(path, self.threadpool, self.log_error)

    def extract_chart_swatches(self, chart_path):
        """
        Load a RAW or numpy chart file, detect the 24 swatch colours,
        save them to a temporary .npy file, and return (swatches, filename).
        """
        img = self.load_raw_image(chart_path)

        if img is None:
            return None, None

        for result in detect_colour_checkers_segmentation(img, additional_data=True, swatch_area_minimum_area_factor=20):
            swatches = result.swatch_colours
            if isinstance(swatches, np.ndarray) and swatches.shape == (24, 3):
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                np.save(tmp_file.name, np.array(swatches))
                return swatches, tmp_file.name
        self.log_info("[Swatches] No valid colour chart swatches detected.")
        return None, None

    def finalize_manual_chart_selection(self, *, canceled: bool = False):
        """Exit manual chart selection using ChartTools."""
        ChartTools.finalize_manual_chart_selection(self, canceled=canceled)

    def _set_chart_tools_enabled(self, *, manual=False, detect=False, show=False, flatten=False, finalize=False):
        """Enable or disable chart tools using ChartTools."""
        ChartTools.set_chart_tools_enabled(self, manual=manual, detect=detect, show=show, flatten=flatten, finalize=finalize)

    @ChartTools.exit_manual_mode
    def manually_select_chart(self):
        """Enter manual chart selection mode using ChartTools."""
        # check if the selected image is marked as chart, if not, mark it but don't detect chart)
        self.set_selected_as_chart(detect=False)
        ChartTools.manually_select_chart(self)

    def on_manual_crop_complete(self, rect: QRect):
        """Called when manual crop is complete using ChartTools."""
        ChartTools.on_manual_crop_complete(self, rect)

    def flatten_chart_image(self):
        """Enter corner-picking mode to flatten chart - moved to ChartTools."""
        return ChartTools.flatten_chart_image(self)

    def eventFilter(self, source, event):
        """
        Master event handler for manual-selection, flatten modes,
        idle zoom/pan, and thumbnail scroll/resize.
        """
        viewport = self.ui.imagePreviewGraphicsView.viewport()
        thumbnail_frame = self.ui.thumbnailPreviewFrame
        ev = event.type()

        # 1) Thumbnail scroll & resize
        if source == thumbnail_frame:
            if ev == QEvent.Wheel:
                delta = event.angleDelta().y()
                if delta > 0:
                    self.select_previous_image()
                elif delta < 0:
                    self.select_next_image()
                return True
            if ev == QEvent.Resize:
                self.update_thumbnail_strip()
                return True
            return super().eventFilter(source, event)

        # 2) Preview viewport events
        if source == viewport:
            # a) Idle zoom
            if ev == QEvent.Wheel and not (self.manual_selection_mode or self.flatten_mode):
                return self._handle_preview_zoom(event)
            # b) Idle pan
            if ev == QEvent.MouseButtonPress and event.button() == Qt.LeftButton \
                    and not (self.manual_selection_mode or self.flatten_mode or self.sampling_white_balance):
                return self._handle_preview_pan_press(event)
            if ev == QEvent.MouseMove and hasattr(self, '_pan_start_pos'):
                return self._handle_preview_pan_move(event)
            if ev == QEvent.MouseButtonRelease and hasattr(self, '_pan_start_pos'):
                return self._handle_preview_pan_release(event)
            # c) Idle resize (respect auto-fit)
            if ev == QEvent.Resize and not (self.manual_selection_mode or self.flatten_mode):
                if getattr(self, '_auto_fit_enabled', True):
                    self.ui.imagePreviewGraphicsView.fitInView(
                        self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio)
                return True
            # d) Block stray pointer events when idle
            if not (self.manual_selection_mode or self.flatten_mode or self.sampling_white_balance) and ev in (
                    QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease):
                return False
            # e) Manual-selection logic
            if self.manual_selection_mode and not self.flatten_mode:
                if ev == QEvent.MouseButtonPress:
                    pos = event.position().toPoint()
                    scene_pt = self.ui.imagePreviewGraphicsView.mapToScene(pos)
                    if not self.previewScene.itemsBoundingRect().contains(scene_pt):
                        return False
                    return ChartTools.handle_manual_press(self, event)
                elif ev == QEvent.MouseMove:
                    return ChartTools.handle_manual_move(self, event)
                elif ev == QEvent.MouseButtonRelease:
                    return ChartTools.handle_manual_release(self, event)
            # f) Flatten mode logic
            if self.flatten_mode and ev == QEvent.MouseButtonPress:
                return ChartTools.handle_flatten_press(self, event)
            # g) White balance sampling mode logic
            if self.sampling_white_balance and ev == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self._sample_white_balance_click(event)
                return True

        # 3) Default fallback
        return super().eventFilter(source, event)

    def _handle_manual_press(self, event):
        """Handle manual press - moved to ChartTools."""
        return ChartTools.handle_manual_press(self, event)

    def _handle_manual_move(self, event):
        """Handle manual move - moved to ChartTools."""
        return ChartTools.handle_manual_move(self, event)

    def _handle_manual_release(self, event):
        """Handle manual release - moved to ChartTools."""
        return ChartTools.handle_manual_release(self, event)

    def _handle_flatten_press(self, event):
        """Handle flatten press - moved to ChartTools."""
        return ChartTools.handle_flatten_press(self, event)

    def _perform_flatten_transform(self):
        """Perform flatten transform - moved to ChartTools."""
        return ChartTools.perform_flatten_transform(self)

    def _handle_preview_zoom(self, event):
        """
        Zoom in/out on the preview viewport with mouse wheel when idle.
        Uses smooth zoom based on angleDelta and transforms around cursor.
        Disables auto-fit once user zooms.
        """
        # Ensure auto-fit flag exists
        if not hasattr(self, '_auto_fit_enabled'):
            self._auto_fit_enabled = True
        # Normalize wheel delta (120 units = one notch)
        delta = event.angleDelta().y() / 120.0
        factor = 1.15 ** delta
        view = self.ui.imagePreviewGraphicsView
        view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        view.scale(factor, factor)
        # After manual zoom, prevent automatic fit-on-resize
        self._auto_fit_enabled = False
        return True

    def _handle_preview_pan_press(self, event):
        """
        Start panning on left mouse button press when idle.
        """
        if event.button() == Qt.LeftButton:
            self._pan_start_pos = event.position().toPoint()
            view = self.ui.imagePreviewGraphicsView
            self._pan_h_scroll = view.horizontalScrollBar().value()
            self._pan_v_scroll = view.verticalScrollBar().value()
            return True
        return False

    def _handle_preview_pan_move(self, event):
        """
        Pan the view as the mouse moves when dragging.
        """
        if hasattr(self, '_pan_start_pos'):
            delta = event.position().toPoint() - self._pan_start_pos
            view = self.ui.imagePreviewGraphicsView
            view.horizontalScrollBar().setValue(self._pan_h_scroll - delta.x())
            view.verticalScrollBar().setValue(self._pan_v_scroll - delta.y())
            return True
        return False

    def _handle_preview_pan_release(self, event):
        """
        End panning on mouse release.
        """
        if hasattr(self, '_pan_start_pos'):
            del self._pan_start_pos, self._pan_h_scroll, self._pan_v_scroll
            return True
        return False

    def preview_manual_swatch_correction(self):
        """
        Extracts mean colours from swatch rectangles in self.flatten_swatch_rects,
        applies manual color correction, and displays the corrected image preview.
        Assumes self.cropped_fp is the current floating-point chart image,
        and self.flatten_swatch_rects is a list of 24 QRect objects.
        """
        self.log_info("[Manual Swatch] Starting preview manual swatch correction...")
        
        if not (hasattr(self, 'flatten_swatch_rects') and self.flatten_swatch_rects and len(
                self.flatten_swatch_rects) == 24):
            self.log_info("[Manual Swatch] Swatch grid not found or incomplete.")
            return
        if not hasattr(self, 'cropped_fp') or self.cropped_fp is None:
            self.log_error("[Manual Swatch] No image to process.")
            return

        self.log_debug(f"[Manual Swatch] Processing {len(self.flatten_swatch_rects)} swatches...")
        
        img_fp = self.cropped_fp
        swatch_colours = []
        for i, rect in enumerate(self.flatten_swatch_rects):
            x0, y0, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            region = img_fp[int(y0):int(y0 + h), int(x0):int(x0 + w), :]
            mean_color = region.mean(axis=(0, 1))
            swatch_colours.append(mean_color)
            
        swatch_colours = np.array(swatch_colours)
        self.log_debug(f"[Manual Swatch] Extracted swatches shape: {swatch_colours.shape}")
        
        self.temp_swatches = swatch_colours
        self.chart_swatches = swatch_colours
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
        np.save(tmp_file.name, swatch_colours)
        self.calibration_file = tmp_file.name
        
        # Store group-specific calibration (only the essential color values)
        group_name = getattr(self, 'current_chart_group', 'All Images')
        
        # Create a clean copy of swatch colors for serialization efficiency
        optimized_swatches = np.array(swatch_colours, dtype=np.float32)  # Use float32 instead of float64
        
        self.group_calibrations[group_name] = {
            'file': tmp_file.name,
            'swatches': optimized_swatches
        }
        self.log_debug(f"[Manual Swatch] Calibration saved for group '{group_name}' - {optimized_swatches.shape} colors")
        
        # Clear large image data from memory after chart processing to improve performance
        if hasattr(self, 'cropped_fp') and self.cropped_fp is not None:
            chart_size_mb = (self.cropped_fp.nbytes / 1024 / 1024) if hasattr(self.cropped_fp, 'nbytes') else 0
            if chart_size_mb > 5.0:  # Clear if chart image is larger than 5MB
                self.log_debug(f"[Memory] Clearing large chart image data ({chart_size_mb:.1f} MB) to improve performance")
                self.cropped_fp = None
        
        # Update process button text
        self.update_process_button_text()
        
        self.log_debug("[Manual Swatch] Creating worker thread...")
        
        worker = SwatchPreviewWorker(
            self.cropped_fp,
            swatch_colours,
            reference_swatches
        )
        worker.signals.finished.connect(self._on_manual_swatch_preview_done)
        worker.signals.error.connect(self._on_manual_swatch_preview_error)

        self.log_debug("[Manual Swatch] Starting worker thread...")
        self.threadpool.start(worker)
        self.log_debug("[Manual Swatch] Worker thread started")

    def _on_manual_swatch_preview_done(self, img_uint8):
        """
        Receives the uint8 array, converts to QPixmap, updates preview, re-enables UI.
        """
        self.log_debug("[Manual Swatch] Recieved manual swatch preview...")
        pixmap = self.pixmap_from_array(img_uint8)
        self._display_preview(pixmap)
        self.log_debug("[Manual Swatch] Manual correction preview displayed.")

    def _on_manual_swatch_preview_error(self, message):
        """
        Called if the worker threw an exception.
        """
        self.log_debug(f"[Manual Swatch] Preview failed: {message}")


    def detect_chart(self, input_source=None, is_npy=False):
        """
        Detects the color checker chart using either manual grid (if available)
        or auto-detection as fallback.
        """

        detected = False

        if input_source is not None:
            img_fp = np.load(input_source) if is_npy else input_source
        else:
            if hasattr(self, 'cropped_fp') and self.cropped_fp is not None:
                img_fp = self.cropped_fp
            else:
                self.log_debug("[Detect Chart] No image available for detection.")
                return

        # RAW file check
        if isinstance(img_fp, str) and img_fp.lower().endswith(
                ('.arw', '.nef', '.cr2', '.cr3', '.dng', '.rw2', '.raw')):
            img_fp = self.load_raw_image(input_source)

        selected_item = self.ui.imagesListWidget.currentItem()
        if selected_item:
            meta = selected_item.data(Qt.UserRole)
            selected_item.setData(Qt.UserRole, meta)
            selected_item.setBackground(QColor('#ADD8E6'))

        self.set_chart_type()
        self.log_debug(f"[Detect Chart] Detected chart type: {self.selected_chart_type}")

        results = None
        
        # Create progress dialog
        progress = QProgressDialog("Attempting to detect chart...", "Cancel", 0, 0, self)
        progress.setWindowTitle("Chart Detection")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)  # Show immediately
        progress.setValue(0)
        progress.show()
        
        # Process events to show the dialog
        QApplication.processEvents()
        
        try:
            results = detect_colour_checkers_segmentation(img_fp, additional_data=True,
                                                          swatch_minimum_area_factor=30)
        finally:
            # Close progress dialog
            progress.close()

        # noinspection PyUnreachableCode
        for colour_checker_data in results:
            if hasattr(colour_checker_data, 'swatch_colours'):
                swatch_colours = colour_checker_data.swatch_colours
                swatch_masks = colour_checker_data.swatch_masks
                colour_checker_image = colour_checker_data.colour_checker
            else:
                swatch_colours, swatch_masks, colour_checker_image = colour_checker_data[:3]

            # Save calibration data
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
            np.save(tmp_file.name, swatch_colours)
            self.calibration_file = tmp_file.name
            self.log_info(f'[Detect] Calibration saved to {self.calibration_file}')
            self.chart_swatches = swatch_colours

            # Store group-specific calibration (optimized for serialization)
            group_name = getattr(self, 'current_chart_group', 'All Images')

            # Create optimized swatch colors for better serialization performance
            optimized_swatches = np.array(swatch_colours, dtype=np.float32)  # Use float32 instead of float64

            self.group_calibrations[group_name] = {
                'file': tmp_file.name,
                'swatches': optimized_swatches
            }
            self.log_debug(f"[Detect] Calibration saved for group '{group_name}' - {optimized_swatches.shape} colors")

            # Update process button text
            self.update_process_button_text()

            # Generate Corrected Image
            corrected = colour.colour_correction(img_fp, swatch_colours, reference_swatches)
            corrected = np.clip(corrected, 0, 1)
            corrected_uint8 = np.uint8(255 * colour.cctf_encoding(corrected))

            # Swatch Overlay
            masks_overlay = np.zeros_like(colour_checker_image)
            for mask in swatch_masks:
                masks_overlay[mask[0]:mask[1], mask[2]:mask[3], :] = 0.25
            swatch_overlay = np.clip(colour_checker_image + masks_overlay, 0, 1)
            swatch_overlay_uint8 = np.uint8(255 * colour.cctf_encoding(swatch_overlay))

            # Detection Debug (Segmented Image)
            detection_debug_uint8 = np.uint8(255 * colour.cctf_encoding(colour_checker_image))

            # Swatches and Clusters (Swatches contours)
            swatches_clusters_img = np.uint8(255 * colour.cctf_encoding(colour_checker_image.copy()))
            for mask in swatch_masks:
                start_row, end_row, start_col, end_col = mask
                cv2.rectangle(
                    swatches_clusters_img,
                    (start_col, start_row),
                    (end_col, end_row),
                    color=(255, 0, 255),
                    thickness=3
                )
            # Store all debug images together
            debug_images = {
                'corrected_image': corrected_uint8,
                'swatch_overlay': swatch_overlay_uint8,
                'detection_debug': detection_debug_uint8,
                'swatches_and_clusters': swatches_clusters_img,
            }

            # Update metadata of currently selected item
            selected_item = self.ui.imagesListWidget.currentItem()
            if selected_item:
                meta = selected_item.data(Qt.UserRole)
                meta['debug_images'] = debug_images
                meta['chart'] = True
                selected_item.setData(Qt.UserRole, meta)
                selected_item.setBackground(QColor('#ADD8E6'))

            # Immediately display corrected preview
            self.corrected_preview_pixmap = self.pixmap_from_array(corrected_uint8)
            self._display_preview(self.corrected_preview_pixmap)

            detected = True

            if is_npy:
                self.ui.finalizeChartPushbutton.setEnabled(True)
                self.instruction_label.setText("Please Finalize your chart")

            break  # Only process first detected checker

        if detected:
            # self.log_debug("[Detect] Chart detected successfully. Debug images generated.")
            self.show_debug_frame(True)
            self.ui.finalizeChartPushbutton.setEnabled(True)

        else:
            self.log_info("[Detect] No valid chart detected.")
            self.show_debug_frame(False)

    def toggle_chart_preview(self):
        if not hasattr(self, 'cropped_preview_pixmap'):
            return
        if self.showing_chart_preview:
            pix = self.cropped_preview_pixmap
            self.ui.showOriginalImagePushbutton.setText('Show Chart Preview')
            self.showing_chart_preview = False
        else:
            pix = self.original_preview_pixmap
            self.ui.showOriginalImagePushbutton.setText('Show Chart Image')
            self.showing_chart_preview = True
        self.previewScene.clear()
        self.previewScene.addItem(QGraphicsPixmapItem(pix))
        self.ui.imagePreviewGraphicsView.resetTransform()
        self.ui.imagePreviewGraphicsView.fitInView(self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def revert_image(self):
        """
        Restore the original preview and fully reset the manual‐selection UI.

        - Displays the saved original_preview_pixmap.
        - Hides any rubber‐band or instruction label.
        - Closes the debug frame.
        - Disables all chart‐tool buttons (detect, show, flatten, finalize).
        - Enables only the manual‐select button for a new crop.
        """
        # 1) Show original preview
        if hasattr(self, 'original_preview_pixmap'):
            self.current_image_pixmap = self.original_preview_pixmap
            self._display_preview(self.original_preview_pixmap)
            self.showing_chart_preview = False

        # 2) Hide any debug/chart overlay
        self.show_debug_frame(False)

        # 3) Remove any rubber‐band selection
        if hasattr(self, 'rubberBand') and self.rubberBand.isVisible():
            self.rubberBand.hide()
        # Prepare for a fresh manual selection
        self.manual_selection_mode = True
        self.flatten_mode = False
        self.corner_points.clear()

        self._set_chart_tools_enabled(manual=True)

        # 6) Clear any temp data from prior selection:
        if hasattr(self, 'temp_swatches'):
            del self.temp_swatches
        if hasattr(self, 'temp_calibration_file'):
            del self.temp_calibration_file


    def show_debug_frame(self, visible):
        """
        Shows the debug window for colour chart debugging
        :param visible:
        :return:
        """
        self.ui.colourChartDebugToolsFrame.setVisible(visible)

    def setup_debug_views(self):
        """
        Setup the UI for debugging
        :return:
        """
        self.ui.correctedImageRadioButton.toggled.connect(lambda checked: checked and self.corrected_image_view())
        self.ui.swatchOverlayRadioButton.toggled.connect(lambda checked: checked and self.swatch_overlay_view())
        self.ui.swatchAndClusterRadioButton.toggled.connect(
            lambda checked: checked and self.swatches_and_clusters_view())
        self.ui.detectionDebugRadioButton.toggled.connect(lambda checked: checked and self.detection_debug_view())

    def swatches_and_clusters_view(self):
        """
        Display the debug image showing swatch bounding‐boxes and cluster outlines in the preview area.
        """
        item = self.ui.imagesListWidget.currentItem()
        if item is None:
            self.log_debug("[Debug View] No item selected.")
            return

        meta = item.data(Qt.UserRole)
        debug_images = meta.get('debug_images', {})

        img = debug_images.get('swatches_and_clusters')
        if img is not None:
            pixmap = self.pixmap_from_array(img)
            self._display_preview(pixmap)
            # self.log_debug("[Debug View] Swatches and Clusters shown.")
        else:
            self.log_debug("[Debug View] Swatches and Clusters unavailable.")

    def corrected_image_view(self):
        """
        Display the colour‐corrected debug image for the selected item in the preview area.
        """
        item = self.ui.imagesListWidget.currentItem()
        if item is None:
            self.log_debug("[Debug View] No item selected.")
            return

        meta = item.data(Qt.UserRole)
        debug_images = meta.get('debug_images', {})

        img = debug_images.get('corrected_image')
        if img is not None:
            pixmap = self.pixmap_from_array(img)
            self._display_preview(pixmap)
            # self.log_debug("[Debug View] Corrected image shown.")
        else:
            self.log_debug("[Debug View] Corrected image unavailable.")

    def swatch_overlay_view(self):
        """
        Display the swatch‐overlay debug image (chart with semi‐transparent swatches) in the preview area.
        """
        item = self.ui.imagesListWidget.currentItem()
        if item is None:
            self.log_debug("[Debug View] No item selected.")
            return

        meta = item.data(Qt.UserRole)
        debug_images = meta.get('debug_images', {})

        img = debug_images.get('swatch_overlay')
        if img is not None:
            pixmap = self.pixmap_from_array(img)
            self._display_preview(pixmap)
            # self.log_debug("[Debug View] Swatch overlay shown.")
        else:
            self.log_debug("[Debug View] Swatch overlay unavailable.")

    def detection_debug_view(self):
        """
        Display the segmentation (detected patch shapes) debug image for the selected item.
        """
        item = self.ui.imagesListWidget.currentItem()
        if item is None:
            self.log_debug("[Debug View] No item selected.")
            return

        meta = item.data(Qt.UserRole)
        debug_images = meta.get('debug_images', {})

        img = debug_images.get('detection_debug')
        if img is not None:
            pixmap = self.pixmap_from_array(img)
            self._display_preview(pixmap)
            # self.log_debug("[Debug View] Detection debug shown.")
        else:
            self.log_debug("[Debug View] Detection debug unavailable.")

    def pixmap_from_array(self, array):
        """
        Convert a H×W×3 uint8 NumPy array into a QPixmap for display.
        Assumes RGB888 layout.
        """
        h, w, c = array.shape
        bytes_per_line = c * w
        img = QImage(array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(img)

    @ChartTools.exit_manual_mode
    def calculate_average_exposure(self):
        """
        Compute and store exposure normalization multipliers for all images.
        Uses chart image as reference if available, otherwise uses average.
        Ignores top 2% (hot spots) and bottom 5% (black areas) of pixel luminance.
        """
        brightness_list = []
        item_to_brightness = {}
        chart_brightness = None
        self.average_enabled = True

        if self.selected_average_source is None:
            self.log_debug("[Exposure Calc] No selected average source.")

        else:

            for i in range(self.ui.imagesListWidget.count()):
                max_size = (512, 512)
                item = self.ui.imagesListWidget.item(i)
                meta = item.data(Qt.UserRole)
                
                # Skip group headers
                if meta.get('is_group_header', False):
                    continue
                    
                img_path = meta.get('input_path')
                arr = None
                cache = self.thumbnail_cache.get(img_path)
                if cache and 'array' in cache and cache['array'] is not None:
                    arr = cache['array']

                if arr is None:
                    self.log_debug(f"[Exposure Calc] Failed on {img_path}: could not load thumbnail")
                    continue
                shadow_threshold = self.ui.shadowLimitSpinBox.value() / 100.0
                highlight_threshold = self.ui.highlightLimitSpinBox.value() / 100.0

                lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

                mask = (lum > shadow_threshold) & (lum < highlight_threshold)

                valid = lum[mask]
                mean_brightness = valid.mean() if valid.size > 0 else lum.mean()
                brightness_list.append(mean_brightness)
                item_to_brightness[img_path] = mean_brightness
                if meta.get('average_source'):
                    chart_brightness = mean_brightness  # Only one chart expected, take the first found

            if not brightness_list:
                self.log_debug("[Exposure Calc] No valid images for exposure normalization.")
                return

            # Reference: chart if present, otherwise mean of all
            reference_brightness = chart_brightness if chart_brightness is not None else np.mean(brightness_list)
            if chart_brightness is None:
                self.log_debug(f"[Exposure Calc] Using average image brightness as reference ({reference_brightness:.3f})")

            # Set exposure multiplier for each image
            for i in range(self.ui.imagesListWidget.count()):
                item = self.ui.imagesListWidget.item(i)
                meta = item.data(Qt.UserRole)
                
                # Skip group headers
                if meta.get('is_group_header', False):
                    continue
                    
                img_path = meta.get('input_path')
                img_brightness = item_to_brightness.get(img_path)
                if img_brightness is None:
                    continue
                multiplier = reference_brightness / img_brightness if img_brightness > 0 else 1.0
                meta['average_exposure'] = multiplier
                item.setData(Qt.UserRole, meta)
                self.log_debug(f"[Exposure Calc] {meta.get('input_path')} multiplier: {multiplier:.3f}")

            self.log_info("[Exposure Calc] Average exposure multipliers calculated and stored.")

    def load_thumbnail_array(self, path, max_size=(512, 512)):
        """Load thumbnail array using ImageLoader utility."""
        return ImageLoader.load_thumbnail_array(
            path, max_size, self.thumbnail_cache, 
            self.chart_swatches, self.correct_thumbnails, self.log
        )

    def remove_average_exposure_data(self):
        """
        Remove the exposure normalization multipliers from all images.
        """
        for i in range(self.ui.imagesListWidget.count()):
            item = self.ui.imagesListWidget.item(i)
            meta = item.data(Qt.UserRole)
            
            # Skip group headers
            if meta.get('is_group_header', False):
                continue
                
            if 'average_exposure' in meta:
                del meta['average_exposure']
                item.setData(Qt.UserRole, meta)

        self.average_enabled = False
        self.update_thumbnail_strip()
        self.log_info("[Exposure Calc] All exposure multipliers removed.")

    def _get_next_image_index(self, current_idx):
        """Get the next non-header image index."""
        for i in range(current_idx + 1, self.ui.imagesListWidget.count()):
            item = self.ui.imagesListWidget.item(i)
            meta = item.data(Qt.UserRole)
            if not meta.get('is_group_header', False):
                return i
        return None
    
    def _get_previous_image_index(self, current_idx):
        """Get the previous non-header image index."""
        for i in range(current_idx - 1, -1, -1):
            item = self.ui.imagesListWidget.item(i)
            meta = item.data(Qt.UserRole)
            if not meta.get('is_group_header', False):
                return i
        return None

    def show_exposure_debug_overlay(self):
        """
        Show an overlay for hot (highlight) and dark (shadow) areas using absolute luminance thresholds.
        Hot spots are red, dark areas are blue.
        """
        item = self.ui.imagesListWidget.currentItem()
        if item is None:
            self.log_debug("[Exposure Debug] No image selected.")
            return

        meta = item.data(Qt.UserRole)
        img_path = meta.get('input_path')
        arr = self.load_thumbnail_array(img_path, max_size=(800, 800))
        if arr is None:
            self.log_debug(f"[Exposure Debug] Could not load image: {img_path}")
            return

        # Absolute thresholds from spinboxes (expected range: 0–100)
        shadow_threshold = self.ui.shadowLimitSpinBox.value() / 100.0
        highlight_threshold = self.ui.highlightLimitSpinBox.value() / 100.0

        lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
        mask_dark = lum <= shadow_threshold
        mask_hot = lum >= highlight_threshold

        overlay = (arr * 255).astype(np.uint8)
        overlay[mask_hot] = [255, 60, 60]  # Red for hot spots
        overlay[mask_dark] = [60, 60, 255]  # Blue for shadows

        alpha = 0.4
        blend = (arr * (1 - alpha) + (overlay / 255.0) * alpha)
        blend = np.clip(blend * 255, 0, 255).astype(np.uint8)

        h, w, c = blend.shape
        img = QImage(blend.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.previewScene.clear()
        self.previewScene.addItem(QGraphicsPixmapItem(pixmap))
        self.ui.imagePreviewGraphicsView.resetTransform()
        self.ui.imagePreviewGraphicsView.fitInView(self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio)
        self.current_image_pixmap = pixmap
        # self.log_debug(f"[Exposure Debug] Overlay shown (Highlight: ≥{highlight_threshold:.2f}, Shadow: ≤{shadow_threshold:.2f})")

    @ChartTools.exit_manual_mode
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.select_next_image()
            event.accept()
        elif event.key() == Qt.Key_Left:
            self.select_previous_image()
            event.accept()
        else:
            super().keyPressEvent(event)

    def select_next_image(self):
        idx = self.ui.imagesListWidget.currentRow()
        next_idx = self._get_next_image_index(idx)
        if next_idx is not None:
            self.ui.imagesListWidget.setCurrentRow(next_idx)

    def select_previous_image(self):
        idx = self.ui.imagesListWidget.currentRow()
        prev_idx = self._get_previous_image_index(idx)
        if prev_idx is not None:
            self.ui.imagesListWidget.setCurrentRow(prev_idx)

    def select_image_from_thumbnail(self, idx):
        # First check if this index is valid and not a group header
        if idx < self.ui.imagesListWidget.count():
            item = self.ui.imagesListWidget.item(idx)
            meta = item.data(Qt.UserRole)
            if not meta.get('is_group_header', False):
                self.ui.imagesListWidget.setCurrentRow(idx)

    def set_selected_image_as_average_source(self):
        selected_item = self.ui.imagesListWidget.currentItem()
        if not selected_item:
            return
            
        data = selected_item.data(Qt.UserRole)
        
        # Don't allow group headers to be set as average source
        if data.get('is_group_header', False):
            self.log_info("[Average] Cannot set group header as average source")
            return
            
        self.selected_average_source = selected_item
        img_path = data.get("input_path")
        data["average_source"] = True
        self.selected_average_source.setData(Qt.UserRole, data)
        self.log_info(f"set {img_path} as average source")

    def _adjust_pixmap_brightness(self, pixmap: QPixmap, factor: float) -> QPixmap:
        """
        Return a new QPixmap with brightness adjusted by `factor`.
        Factor >1 brightens, <1 darkens.
        """
        img = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        w, h = img.width(), img.height()

        ptr = img.bits()
        arr = np.ndarray((h, w, 4), dtype=np.uint8, buffer=ptr)

        rgb = arr[..., :3].astype(np.float32) * factor
        arr[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)

        return QPixmap.fromImage(img)

    def update_thumbnail_strip(self):
        """
        Refresh and layout all thumbnail widgets; apply any average-exposure brightness tweak.
        """
        holder = self.ui.thumbnailPreviewDisplayFrame_holder
        # Clear previous
        for child in holder.findChildren(QWidget):
            child.setParent(None)
            child.deleteLater()

        count = self.ui.imagesListWidget.count()
        if count == 0:
            return

        sel_idx = self.ui.imagesListWidget.currentRow()
        frame_width = holder.width()

        # Collect only image indices (skip group headers) and their widths
        image_indices = []
        widths = []
        default_aspect = 1.5
        default_width = int(default_aspect * 60)
        
        for i in range(count):
            item = self.ui.imagesListWidget.item(i)
            meta = item.data(Qt.UserRole)
            
            # Skip group headers
            if meta.get('is_group_header', False):
                continue
                
            image_indices.append(i)
            cache = self.thumbnail_cache.get(meta['input_path'], {})
            pixmap = cache.get('pixmap')
            if pixmap and not pixmap.isNull():
                aspect = pixmap.width() / pixmap.height()
                widths.append(int(aspect * 60))
            else:
                widths.append(default_width)

        if not image_indices:
            return

        # Find selected image position in the image_indices list
        try:
            sel_pos = image_indices.index(sel_idx)
        except ValueError:
            # Selected item is a group header, find first image after it
            for i, img_idx in enumerate(image_indices):
                if img_idx > sel_idx:
                    sel_pos = i
                    sel_idx = img_idx
                    self.ui.imagesListWidget.setCurrentRow(sel_idx)
                    break
            else:
                sel_pos = 0
                sel_idx = image_indices[0]
                self.ui.imagesListWidget.setCurrentRow(sel_idx)

        # Build thumbnail strip around selected image
        thumb_positions = [sel_pos]
        total_width = widths[sel_pos]
        left, right = sel_pos - 1, sel_pos + 1
        
        while left >= 0 or right < len(image_indices):
            if left >= 0 and (right >= len(image_indices) or len(thumb_positions) % 2 == 1):
                if total_width + widths[left] > frame_width and len(thumb_positions) >= 3:
                    break
                thumb_positions.insert(0, left)
                total_width += widths[left]
                left -= 1
            elif right < len(image_indices):
                if total_width + widths[right] > frame_width and len(thumb_positions) >= 3:
                    break
                thumb_positions.append(right)
                total_width += widths[right]
                right += 1
            else:
                break

        display_widths = [widths[pos] for pos in thumb_positions]
        if 0 < total_width < frame_width:
            scale = frame_width / total_width
            display_widths = [int(w * scale) for w in display_widths]

        # Build thumbnails
        x_offset = 0
        for pos_idx, pos in enumerate(thumb_positions):
            list_idx = image_indices[pos]
            item = self.ui.imagesListWidget.item(list_idx)
            meta = item.data(Qt.UserRole)
            cache = self.thumbnail_cache.get(meta['input_path'], {})
            pixmap = cache.get('pixmap')
            label = ClickableLabel(list_idx, holder)
            
            if pixmap and not pixmap.isNull():
                thumb = pixmap.scaled(display_widths[pos_idx], 60,
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)
                # Apply brightness adjust if enabled
                factor = meta.get('average_exposure', 1.0)
                if factor != 1.0:
                    thumb = self._adjust_pixmap_brightness(thumb, factor)
                label.setPixmap(thumb)
                label.setFixedSize(thumb.width(), thumb.height())
            else:
                label.setText("No preview")
                label.setFixedSize(display_widths[pos_idx], 60)

            label.setStyleSheet(
                "border: 2px solid #2196F3;" if list_idx == sel_idx else "border: 1px solid #999;"
            )
            label.clicked.connect(self.select_image_from_thumbnail)
            label.move(x_offset, 0)
            label.show()
            x_offset += label.width()

    def draw_colorchecker_swatch_grid(self, scene, image_rect, n_cols=6, n_rows=4, color=QColor(0, 255, 0, 90)):
        """Draw swatch grid using ChartTools."""
        return ChartTools.draw_colorchecker_swatch_grid(scene, image_rect, n_cols, n_rows, color)

    # ────────────────────────────────────────────────────────────
    # Real-time Preview Editing Control Handlers
    # ────────────────────────────────────────────────────────────
    
    def on_exposure_adjustment_changed(self):
        """Handle exposure slider change and sync with spinbox."""
        value = self.ui.exposureAdjustmentSlider.value()
        if self._sync_exposure_controls:
            self._sync_exposure_controls = False
            # Convert slider (-100 to 100) to EV stops (-10 to +10)
            ev_value = value / 10.0
            self.ui.exposureAdjustmentDoubleSpinBox.setValue(ev_value)
            self._sync_exposure_controls = True
        self._update_preview_display()

    def on_exposure_spinbox_changed(self, value):
        """Handle exposure spinbox change and sync with slider."""
        if self._sync_exposure_controls:
            self._sync_exposure_controls = False
            # Convert EV stops to slider range
            slider_value = int(value * 10)
            self.ui.exposureAdjustmentSlider.setValue(slider_value)
            self._sync_exposure_controls = True
        self._update_preview_display()

    def on_shadow_adjustment_changed(self):
        """Handle shadow slider change and sync with spinbox."""
        value = self.ui.shadowAdjustmentSlider.value()
        if self._sync_shadow_controls:
            self._sync_shadow_controls = False
            value = value / 1000.0
            self.ui.shadowAdjustmentDoubleSpinBox.setValue(value)
            self._sync_shadow_controls = True
        self._update_preview_display()

    def on_shadow_spinbox_changed(self, value):
        """Handle shadow spinbox change and sync with slider."""
        if self._sync_shadow_controls:
            self._sync_shadow_controls = False
            value = int(value * 1000)
            self.ui.shadowAdjustmentSlider.setValue(value)
            self._sync_shadow_controls = True
        self._update_preview_display()

    def on_highlight_adjustment_changed(self):
        """Handle highlight slider change and sync with spinbox."""
        value = self.ui.highlightAdjustmentSlider.value()
        if self._sync_highlight_controls:
            self._sync_highlight_controls = False
            value = value / 1000.0
            self.ui.highlightAdjustmentDoubleSpinBox.setValue(value)
            self._sync_highlight_controls = True
        self._update_preview_display()

    def on_highlight_spinbox_changed(self, value):
        """Handle highlight spinbox change and sync with slider."""
        if self._sync_highlight_controls:
            self._sync_highlight_controls = False
            value = int(value * 1000)
            self.ui.highlightAdjustmentSlider.setValue(value)
            self._sync_highlight_controls = True
        self._update_preview_display()

    def reset_image_editing_sliders(self, update=True):
        """Reset image editing sliders."""
        self._sync_highlight_controls = False
        self._sync_denoise_controls = False
        self._sync_sharpen_controls = False
        
        # Reset all adjustment controls
        self.ui.highlightAdjustmentDoubleSpinBox.setValue(0)
        self.ui.exposureAdjustmentDoubleSpinBox.setValue(0)
        self.ui.shadowAdjustmentDoubleSpinBox.setValue(0)
        self.ui.exposureAdjustmentSlider.setValue(0)
        self.ui.shadowAdjustmentSlider.setValue(0)
        self.ui.highlightAdjustmentSlider.setValue(0)
        
        # Reset white balance controls
        self.ui.enableWhiteBalanceCheckBox.setChecked(False)
        
        # Reset denoise controls
        self.ui.denoiseImageCheckBox.setChecked(False)
        self.ui.denoiseDoubleSpinBox.setValue(0)
        self.ui.denoiseHorizontalSlider.setValue(0)
        
        # Reset sharpen controls
        self.ui.sharpenImageCheckBox.setChecked(False)
        self.ui.sharpenDoubleSpinBox.setValue(0)
        self.ui.sharpenHorizontalSlider.setValue(0)
        
        self._sync_highlight_controls = True
        self._sync_denoise_controls = True
        self._sync_sharpen_controls = True
        if update:
            self._update_preview_display()

    def on_white_balance_checkbox_changed(self, checked):
        """Handle white balance checkbox change."""
        self._update_preview_display()

    def on_white_balance_changed(self, value):
        """Handle white balance spinbox change."""
        self._update_preview_display()
    
    def sample_white_balance_from_image(self):
        """
        Enable click-to-sample white balance mode.
        User clicks on a neutral area in the preview image to set white balance.
        """
        if self.current_preview_array is None:
            self.log_debug("No image loaded for white balance sampling")
            return
        
        # Enable sampling mode - this will be handled by the event filter
        self.sampling_white_balance = True
        self.ui.sampleWhiteBalancePushButton.setText("Click on neutral area...")
        cursor_pixmap = QPixmap(r'resources/icons/color-picker.svg')
        self.cursor = QCursor(cursor_pixmap, hotX=0, hotY=0)
        self.ui.imagePreviewGraphicsView.setCursor(self.cursor)
        self.ui.sampleWhiteBalancePushButton.setEnabled(False)
        
        self.log_info("Click on a neutral (gray/white) area in the image to sample white balance")
    
    def _sample_white_balance_click(self, event):
        """
        Handle mouse click for white balance sampling.
        
        Args:
            event: QMouseEvent from graphics view
        """
        try:
            if not self.sampling_white_balance or self.current_preview_array is None:
                return
            
            # Get click position using the same pattern as chartTools.py
            pt = event.position().toPoint()
            scene_pos = self.ui.imagePreviewGraphicsView.mapToScene(pt)
            
            # Scene coordinates directly correspond to image pixel coordinates
            x = int(scene_pos.x())
            y = int(scene_pos.y())
            
            # Get image dimensions
            img_height, img_width = self.current_preview_array.shape[:2]
            
            # Ensure coordinates are within image bounds
            if x < 0 or x >= img_width or y < 0 or y >= img_height:
                self.log_info("Click outside image area. Try again.")
                return
            
            # Import the sampling function
            from ImageProcessor.editingTools import sample_white_balance_from_point
            
            # Sample white balance at clicked location
            new_wb = sample_white_balance_from_point(
                self.current_preview_array,
                sample_x=x,
                sample_y=y,
                current_temp=self.original_white_balance,
                sample_radius=5
            )
            
            # Update the white balance spinbox
            self.ui.whitebalanceSpinbox.setValue(int(new_wb))
            
            # Restore normal mode
            self._end_white_balance_sampling()
            
            self.log_info(f"Sampled white balance: {int(new_wb)}K from position ({x}, {y})")
            
        except Exception as e:
            self.log_error(f"Error sampling white balance: {e}")
            self._end_white_balance_sampling()
    
    def _end_white_balance_sampling(self):
        """End white balance sampling mode and restore normal UI."""
        self.sampling_white_balance = False
        self.ui.sampleWhiteBalancePushButton.setText("Sample Whitebalance")
        self.ui.sampleWhiteBalancePushButton.setEnabled(True)
        self.ui.imagePreviewGraphicsView.unsetCursor()

    def on_denoise_checkbox_changed(self, checked):
        """Handle denoise checkbox toggle."""
        # Enable/disable the denoise controls
        self.ui.denoiseHorizontalSlider.setEnabled(checked)
        self.ui.denoiseDoubleSpinBox.setEnabled(checked)
        self._update_preview_display()
    
    def on_denoise_slider_changed(self):
        """Handle denoise slider change and sync with spinbox."""
        value = self.ui.denoiseHorizontalSlider.value()
        if self._sync_denoise_controls:
            self._sync_denoise_controls = False
            # Slider range 0-100 maps directly to denoise strength 0-100
            self.ui.denoiseDoubleSpinBox.setValue(value)
            self._sync_denoise_controls = True
        self._update_preview_display()

    def on_denoise_spinbox_changed(self, value):
        """Handle denoise spinbox change and sync with slider."""
        if self._sync_denoise_controls:
            self._sync_denoise_controls = False
            # Spinbox value maps directly to slider (0-100)
            self.ui.denoiseHorizontalSlider.setValue(int(value))
            self._sync_denoise_controls = True
        self._update_preview_display()

    def on_sharpen_checkbox_changed(self, checked):
        """Handle sharpen checkbox change."""
        self._update_preview_display()

    def on_sharpen_slider_changed(self):
        """Handle sharpen slider change and sync with spinbox."""
        value = self.ui.sharpenHorizontalSlider.value()
        if self._sync_sharpen_controls:
            self._sync_sharpen_controls = False
            # Slider value maps directly to spinbox (0-100)
            self.ui.sharpenDoubleSpinBox.setValue(float(value))
            self._sync_sharpen_controls = True
        self._update_preview_display()

    def on_sharpen_spinbox_changed(self, value):
        """Handle sharpen spinbox change and sync with slider."""
        if self._sync_sharpen_controls:
            self._sync_sharpen_controls = False
            # Spinbox value maps directly to slider (0-100)
            self.ui.sharpenHorizontalSlider.setValue(int(value))
            self._sync_sharpen_controls = True
        self._update_preview_display()

    def _position_loading_spinner(self):
        """Position the loading spinner in the bottom-right corner of the graphics view."""
        if hasattr(self, 'loading_spinner'):
            view = self.ui.imagePreviewGraphicsView
            view_rect = view.rect()
            spinner_rect = self.loading_spinner.geometry()
            
            # Position in bottom-right with 20px margin
            x = view_rect.width() - spinner_rect.width() - 20
            y = view_rect.height() - spinner_rect.height() - 20
            
            self.loading_spinner.move(x, y)
    
    def show_loading_spinner(self):
        """Show and start the loading spinner animation."""
        if hasattr(self, 'loading_spinner'):
            # Ensure we're on the main thread
            if QThread.currentThread() != QApplication.instance().thread():
                QMetaObject.invokeMethod(self, "show_loading_spinner", Qt.QueuedConnection)
                return
            self._position_loading_spinner()  # Update position
            self.loading_spinner.start_spinning()
    
    def hide_loading_spinner(self):
        """Hide and stop the loading spinner animation."""
        if hasattr(self, 'loading_spinner'):
            # Ensure we're on the main thread
            if QThread.currentThread() != QApplication.instance().thread():
                QMetaObject.invokeMethod(self, "hide_loading_spinner", Qt.QueuedConnection)
                return
            self.loading_spinner.stop_spinning()

    def resizeEvent(self, event):
        """Handle window resize to reposition spinner."""
        super().resizeEvent(event)
        if hasattr(self, 'loading_spinner'):
            self._position_loading_spinner()

    # ────────────────────────────────────────────────────────────
    # Thumbnail Grid View Methods
    # ────────────────────────────────────────────────────────────
    
    def toggle_thumbnail_view_mode(self, enabled):
        """
        Toggle between text list view and thumbnail grid view.
        
        Args:
            enabled (bool): True to enable thumbnail view, False for text list view
        """
        self.is_thumbnail_view_mode = enabled
        
        if enabled:
            self.log_debug("[Thumbnail] Switching to thumbnail grid view")
            self._create_thumbnail_grid_view()
        else:
            self.log_debug("[Thumbnail] Switching to text list view")
            self._show_text_list_view()
    
    def update_thumbnail_zoom_level(self, value):
        """
        Update the thumbnail zoom level (number of columns) and refresh the grid if in thumbnail mode.
        
        Args:
            value (int): Number of columns from 1-6 (slider range)
        """
        self.thumbnail_zoom_level = value
        
        # Only update grid if we're currently in thumbnail mode
        if self.is_thumbnail_view_mode:
            # Use populate instead of refresh to handle column count changes
            self._populate_thumbnail_grid()
    
    def _create_thumbnail_grid_view(self):
        """Create and show the thumbnail grid container using existing UI scroll area."""
        # Hide the text list widget
        self.ui.imagesListWidget.hide()
        
        # Create thumbnail grid widget if it doesn't exist
        if self.thumbnail_grid_widget is None:
            from PySide6.QtWidgets import QWidget, QGridLayout
            
            # Create the grid widget
            self.thumbnail_grid_widget = QWidget()
            self.thumbnail_grid_layout = QGridLayout(self.thumbnail_grid_widget)
            self.thumbnail_grid_layout.setContentsMargins(5, 5, 5, 5)
            self.thumbnail_grid_layout.setSpacing(5)
            
            # Set the grid widget as the scroll area's widget
            self.ui.thumbnailContainerScrollArea.setWidget(self.thumbnail_grid_widget)
        
        # Show the existing thumbnail container from UI
        self.ui.thumbnailContainerScrollArea.show()
        
        # Populate the grid with thumbnails
        self._populate_thumbnail_grid()
    
    def _show_text_list_view(self):
        """Show the text list widget and hide the thumbnail grid."""
        # Hide the thumbnail container scroll area
        self.ui.thumbnailContainerScrollArea.hide()
        
        # Show the text list widget
        self.ui.imagesListWidget.show()
    
    def _populate_thumbnail_grid(self):
        """Populate the thumbnail grid with images and group headers."""
        if not self.thumbnail_grid_layout:
            return
        
        # Check if column count changed or grid is empty
        current_layout_columns = getattr(self, '_current_grid_columns', 0)
        grid_is_empty = self.thumbnail_grid_layout.count() == 0
        column_count_changed = current_layout_columns != self.thumbnail_zoom_level
        
        if column_count_changed:
            # Clear existing items when column count changes
            self._clear_thumbnail_grid()
            self._current_grid_columns = self.thumbnail_zoom_level
        elif not grid_is_empty:
            # Grid already populated with correct columns, just refresh styling
            self._refresh_thumbnail_grid()
            return
        
        # Use zoom level directly as number of columns (1-6)
        columns = self.thumbnail_zoom_level
        
        # Calculate thumbnail size to fit within maximum area width (320px)
        max_container_width = 300  # Maximum preview area width
        container_width = self.ui.thumbnailContainerScrollArea.viewport().width()
        
        # Use the smaller of actual width or max width
        effective_width = min(container_width, max_container_width) if container_width > 0 else max_container_width
        
        # Calculate thumbnail size to fit columns within effective width
        # Account for margins (10px total) and spacing between thumbnails (5px each)
        available_width = effective_width - 3  # 5px margin on each side
        spacing_width = (columns - 1)  # 5px spacing between columns
        thumbnail_size = max(16, (available_width - spacing_width) // columns)  # Minimum 64px thumbnails
        
        row = 0
        col = 0
        
        # Track the currently selected item for restoration
        current_selection = self.ui.imagesListWidget.currentRow()
        
        for i in range(self.ui.imagesListWidget.count()):
            list_item = self.ui.imagesListWidget.item(i)
            metadata = list_item.data(Qt.UserRole)
            
            if metadata.get('is_group_header', False):
                # Add group header spanning all columns
                group_label = self._create_group_header_widget(list_item.text(), columns)
                if col > 0:  # Move to next row if we're not at the start
                    row += 1
                    col = 0
                self.thumbnail_grid_layout.addWidget(group_label, row, 0, 1, columns)
                row += 1
                col = 0
            else:
                # Add thumbnail widget
                thumbnail_widget = self._create_thumbnail_widget(
                    list_item, i, thumbnail_size, current_selection == i
                )
                self.thumbnail_grid_layout.addWidget(thumbnail_widget, row, col)
                
                col += 1
                if col >= columns:
                    col = 0
                    row += 1
        
        self.log_debug(f"[Thumbnail] Grid populated: {row + 1} rows, {columns} columns, {thumbnail_size}px thumbnails")
    
    def _clear_thumbnail_grid(self):
        """Clear all widgets from the thumbnail grid."""
        if not self.thumbnail_grid_layout:
            return
        
        # Remove all widgets from the layout
        while self.thumbnail_grid_layout.count():
            item = self.thumbnail_grid_layout.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
    
    def _create_group_header_widget(self, group_text, span_columns):
        """
        Create a group header widget for the thumbnail grid.
        
        Args:
            group_text (str): The group header text
            span_columns (int): Number of columns to span
        
        Returns:
            QLabel: Styled group header label
        """
        from PySide6.QtWidgets import QLabel
        from PySide6.QtGui import QFont

        header_label = QLabel(group_text)
        header_label.setObjectName("thumbnailGroupHeader")
        
        # Style the header to match the original list widget headers
        header_label.setStyleSheet("""
            QLabel#thumbnailGroupHeader {
                background-color: #E3F2FD;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
                color: #1565C0;
            }
        """)
        
        # Set font to bold
        font = header_label.font()
        font.setBold(True)
        header_label.setFont(font)
        
        return header_label
    
    def _create_thumbnail_widget(self, list_item, item_index, thumbnail_size, is_selected):
        """
        Create a thumbnail widget for the grid.
        
        Args:
            list_item (QListWidgetItem): The source list item
            item_index (int): Index in the original list
            thumbnail_size (int): Size of the thumbnail in pixels
            is_selected (bool): Whether this item is currently selected
        
        Returns:
            QWidget: Thumbnail widget with image and metadata
        """
        from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
        from PySide6.QtGui import QFont

        # Create container widget
        container = QWidget()
        container.setObjectName("thumbnailItemContainer")
        container.setFixedSize(thumbnail_size + 2, thumbnail_size + 20)  # +10 for border, +40 for text
        
        # Create layout
        layout = QVBoxLayout(container)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        # Create thumbnail label
        thumbnail_label = ClickableLabel(item_index)
        thumbnail_label.setObjectName("thumbnailImage")
        thumbnail_label.setFixedSize(thumbnail_size, thumbnail_size)
        thumbnail_label.setStyleSheet("border: 1px solid #ccc; background: #f5f5f5;")
        thumbnail_label.setAlignment(Qt.AlignCenter)
        thumbnail_label.setScaledContents(True)
        
        # Load thumbnail image
        metadata = list_item.data(Qt.UserRole)
        image_path = metadata.get('input_path')
        
        if image_path and image_path in self.thumbnail_cache:
            cache_entry = self.thumbnail_cache[image_path]
            if cache_entry and 'pixmap' in cache_entry:
                pixmap = cache_entry['pixmap']
                if pixmap and not pixmap.isNull():
                    # Scale pixmap to fit thumbnail size
                    scaled_pixmap = pixmap.scaled(
                        thumbnail_size, thumbnail_size,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation
                    )
                    
                    # Apply exposure adjustment if available
                    exposure_factor = metadata.get('average_exposure', 1.0)
                    if exposure_factor != 1.0:
                        scaled_pixmap = self._adjust_pixmap_brightness(scaled_pixmap, exposure_factor)
                    
                    thumbnail_label.setPixmap(scaled_pixmap)
                else:
                    thumbnail_label.setText("Loading...")
            else:
                thumbnail_label.setText("Loading...")
        else:
            thumbnail_label.setText("No Preview")
        
        # Create filename label
        filename = os.path.basename(image_path) if image_path else list_item.text()
        # Truncate filename if too long
        if len(filename) > 15:
            filename = filename[:12] + "..."
        
        filename_label = QLabel(filename)
        filename_label.setObjectName("thumbnailFilename")
        filename_label.setAlignment(Qt.AlignCenter)
        filename_label.setWordWrap(True)
        
        # Set smaller font for filename
        font = filename_label.font()
        font.setPointSize(8)
        filename_label.setFont(font)
        
        # Style based on processing status
        status = metadata.get('status', 'raw')
        if status == 'finished':
            border_color = '#4caf50'  # Green
        elif status == 'started':
            border_color = '#FFA500'  # Orange
        elif status == 'error':
            border_color = '#f44336'  # Red
        elif status == 'sent_to_network':
            border_color = '#a682fa'  # Purple
        else:
            border_color = '#ccc'  # Default gray
        
        # Apply selection styling
        if is_selected:
            border_color = '#2196F3'
            border_width = 3
        else:
            border_width = 1
            
        thumbnail_label.setStyleSheet(f"""
            QLabel#thumbnailImage {{
                border: {border_width}px solid {border_color};
                background: #f5f5f5;
            }}
        """)
        
        # Connect click handler
        thumbnail_label.clicked.connect(self._on_thumbnail_clicked)
        
        # Add to layout
        layout.addWidget(thumbnail_label)
        layout.addWidget(filename_label)
        
        return container
    
    def _on_thumbnail_clicked(self, item_index):
        """Handle thumbnail click to select the corresponding image."""
        # Update selection in the original list widget
        self.ui.imagesListWidget.setCurrentRow(item_index)
        
        # Update selection styling without rebuilding the entire grid
        if self.is_thumbnail_view_mode:
            self._refresh_thumbnail_grid()
    
    def _refresh_thumbnail_grid(self):
        """Refresh the thumbnail grid (update selection styling only)."""
        if not self.is_thumbnail_view_mode or not self.thumbnail_grid_layout:
            return
        
        # Update selection styling for existing thumbnail widgets
        current_selection = self.ui.imagesListWidget.currentRow()
        
        # Iterate through all widgets in the grid layout
        for i in range(self.thumbnail_grid_layout.count()):
            layout_item = self.thumbnail_grid_layout.itemAt(i)
            if layout_item and layout_item.widget():
                widget = layout_item.widget()
                
                # Check if this is a thumbnail container (not a group header)
                if widget.objectName() == "thumbnailItemContainer":
                    # Find the ClickableLabel inside this container
                    thumbnail_label = widget.findChild(ClickableLabel)
                    if thumbnail_label:
                        item_index = thumbnail_label.index
                        is_selected = (item_index == current_selection)
                        
                        # Update the border styling based on selection
                        # Get the current status for border color
                        if item_index < self.ui.imagesListWidget.count():
                            list_item = self.ui.imagesListWidget.item(item_index)
                            if list_item:
                                metadata = list_item.data(Qt.UserRole)
                                if not metadata.get('is_group_header', False):
                                    status = metadata.get('status', 'raw')
                                    
                                    # Determine border color based on status
                                    if status == 'finished':
                                        border_color = '#4caf50'  # Green
                                    elif status == 'started':
                                        border_color = '#FFA500'  # Orange
                                    elif status == 'error':
                                        border_color = '#f44336'  # Red
                                    elif status == 'sent_to_network':
                                        border_color = '#a682fa'  # Purple
                                    else:
                                        border_color = '#ccc'  # Default gray
                                    
                                    # Override with selection styling if selected
                                    if is_selected:
                                        border_color = '#2196F3'
                                        border_width = 3
                                    else:
                                        border_width = 1
                                    
                                    # Apply the styling
                                    thumbnail_label.setStyleSheet(f"""
                                        QLabel#thumbnailImage {{
                                            border: {border_width}px solid {border_color};
                                            background: #f5f5f5;
                                        }}
                                    """)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.showMaximized()  # Maximize window to fill screen on launch
    sys.exit(app.exec())
