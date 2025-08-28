import os
import time
import datetime
import platform
import json
import tempfile
import urllib.request
import urllib.error
import re
import gc

import numpy as np
from PySide6.QtCore import Qt, QRunnable, QObject, Signal, Slot, QSettings, QThreadPool
from PySide6.QtWidgets import (QFileDialog, QMessageBox, QProgressDialog, QPushButton, 
                               QDialog, QVBoxLayout, QHBoxLayout, QLabel, QTextEdit, QFrame)
from PySide6.QtGui import QFont


def export_current_project(self):
    """
    Export the current project configuration to a JSON file.

    This creates a comprehensive project file containing all image paths,
    settings, chart configurations, and metadata for batch processing.
    """
    # Check if we have any images to export
    if not hasattr(self, 'ui') or self.ui.imagesListWidget.count() == 0:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(
            self,
            "No Images",
            "No images available to export. Please load images first."
        )
        return

    # Open file save dialog
    file_dialog = QFileDialog(self)
    file_dialog.setAcceptMode(QFileDialog.AcceptSave)
    file_dialog.setNameFilter("Project files (*.json)")
    file_dialog.setDefaultSuffix("json")
    file_dialog.setWindowTitle("Export Current Project")

    # Set default filename with current date
    from datetime import datetime
    default_name = f"ImageSpace_Project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    file_dialog.selectFile(default_name)

    if file_dialog.exec() == QFileDialog.Accepted:
        file_path = file_dialog.selectedFiles()[0]

        try:
            project_data = self._build_project_data()

            # Sanitize the project data to remove emojis and non-standard characters
            sanitized_data = self._sanitize_project_data(project_data)

            # Write JSON file with pretty formatting
            import json
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_data, f, indent=2, ensure_ascii=True)

            self.log_info(f"[Project] Project exported to: {file_path}")

            from PySide6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Export Successful",
                f"Project exported successfully to:\n{file_path}\n\n"
                f"Images: {len(project_data.get('images', []))}\n"
                f"Groups: {len(project_data.get('image_groups', {}))}"
            )

        except Exception as e:
            self.log_error(f"[Project] Error exporting project: {e}")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(
                self,
                "Export Error",
                f"Failed to export project:\n{str(e)}"
            )


class ProjectSubmissionWorker(QRunnable):
    """Worker thread for sending project data to server without blocking UI."""

    class Signals(QObject):
        progress = Signal(str)  # Progress message
        error = Signal(str)  # Error message
        success = Signal(dict)  # Success with response data
        finished = Signal()  # Finished signal

    def __init__(self, main_window, host, port, project_data, image_count):
        super().__init__()
        self.main_window = main_window
        self.host = host
        self.port = port
        self.project_data = project_data
        self.image_count = image_count
        self.signals = self.Signals()

    @Slot()
    def run(self):
        """Execute the project submission in background thread."""
        try:
            import json
            import urllib.request
            import urllib.error
            import tempfile
            import os

            # Update progress
            self.signals.progress.emit("Serializing project data...")

            # Try to serialize and catch any numpy array issues
            try:
                json_string = json.dumps(self.project_data, indent=2)
                json_data = json_string.encode('utf-8')

                # Save to temp file for debugging
                temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, prefix='project_debug_')
                temp_file.write(json_string)
                temp_file.close()

            except TypeError as e:
                self.signals.error.emit(f"JSON serialization error: {e}")
                return

            # Update progress
            self.signals.progress.emit(f"Connecting to server {self.host}:{self.port}...")

            # Prepare HTTP request to standalone server API
            api_url = f"http://{self.host}:{self.port}/api/jobs/submit"

            # Create HTTP request
            req = urllib.request.Request(
                api_url,
                data=json_data,
                headers={
                    'Content-Type': 'application/json',
                    'Content-Length': str(len(json_data))
                },
                method='POST'
            )

            # Update progress
            self.signals.progress.emit(f"Sending {self.image_count} images to server...")

            # Send request with timeout
            try:
                with urllib.request.urlopen(req, timeout=30) as response:
                    response_data = json.loads(response.read().decode('utf-8'))

                # Success
                self.signals.success.emit({
                    'response': response_data,
                    'image_count': self.image_count,
                    'host': self.host,
                    'port': self.port
                })

            except urllib.error.HTTPError as http_err:
                error_msg = f"HTTP {http_err.code}: {http_err.reason}"
                try:
                    # Try to get detailed error from server response
                    error_response = http_err.read().decode('utf-8')
                    if error_response:
                        try:
                            error_details = json.loads(error_response)
                            error_msg += f" - {error_details.get('error', error_response)}"
                        except json.JSONDecodeError:
                            error_msg += f" - {error_response}"
                except:
                    pass
                self.signals.error.emit(error_msg)

            except urllib.error.URLError as url_err:
                self.signals.error.emit(f"Failed to connect to server: {url_err.reason}")

            except json.JSONDecodeError:
                self.signals.error.emit("Invalid response from server")

            except Exception as e:
                self.signals.error.emit(f"Network error: {str(e)}")

        except Exception as e:
            self.signals.error.emit(f"Unexpected error: {str(e)}")
        finally:
            self.signals.finished.emit()


def send_project_to_server(self):
    """
    Send the current project to an external processing server for distributed processing.
    Shows a progress dialog and runs the submission in a background thread.
    """
    # Check if we have any images to send
    if not hasattr(self, 'ui') or self.ui.imagesListWidget.count() == 0:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(
            self,
            "No Images",
            "No images available to send. Please load images first."
        )
        return

    # Get server address from settings
    settings = QSettings('ScanSpace', 'ImageProcessor')
    server_address = settings.value('host_server_address', '', type=str)

    if not server_address:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(
            self,
            "No Server Address",
            "Please configure the server address in Settings first."
        )
        return

    # Parse host and port
    try:
        if ':' in server_address:
            host, port = server_address.split(':', 1)
            port = int(port)
        else:
            host = server_address
            port = 8889  # Default API port for standalone server
    except ValueError:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(
            self,
            "Invalid Server Address",
            "Server address must be in format 'host:port' or just 'host' (default API port 8889)."
        )
        return

    try:
        # Show submission confirmation dialog before proceeding
        if not self._show_submission_confirmation_dialog(host, port):
            return  # User cancelled

        # Update status
        self.update_server_status_label("Preparing project data...")

        # Build and sanitize project data using existing function
        project_data = self._build_project_data()
        if project_data is None:  # Check if _build_project_data returned None (error)
            return

        sanitized_data = self._sanitize_project_data(project_data)

        # Count actual images (not group headers)
        image_count = 0
        for i, image in enumerate(sanitized_data.get('images', [])):
            try:
                # Check if this is a group header using the metadata field
                metadata = image.get('metadata', {})
                if not metadata.get('is_group_header', False) and image.get('full_path'):
                    image_count += 1
            except Exception as e:
                self.log_error(f"Error processing image {i}: {e}")
                continue

        if image_count == 0:
            self.update_server_status_label("No images to process")
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "No Images",
                "No processable images found. Please load images first."
            )
            return

        # Create and show progress dialog
        from PySide6.QtWidgets import QProgressDialog, QPushButton
        self.progress_dialog = QProgressDialog(
            "Preparing to send project to server...",
            "Cancel",
            0, 0,
            self
        )
        self.progress_dialog.setWindowTitle("Sending Project to Server")
        self.progress_dialog.setModal(True)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setWindowFlags(self.progress_dialog.windowFlags() & ~Qt.WindowCloseButtonHint)

        # Create custom cancel button that actually works
        cancel_button = QPushButton("Cancel")
        self.progress_dialog.setCancelButton(cancel_button)

        # Show the dialog
        self.progress_dialog.show()

        # Create and start worker thread
        self.submission_worker = self.ProjectSubmissionWorker(
            self, host, port, sanitized_data, image_count
        )

        # Connect signals
        self.submission_worker.signals.progress.connect(self._on_submission_progress)
        self.submission_worker.signals.success.connect(self._on_submission_success)
        self.submission_worker.signals.error.connect(self._on_submission_error)
        self.submission_worker.signals.finished.connect(self._on_submission_finished)

        # Start the worker
        QThreadPool.globalInstance().start(self.submission_worker)

    except Exception as e:
        self.update_server_status_label("Send failed - Ready to retry")
        self.log_error(f"[Server] Error preparing to send jobs to server: {e}")
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(
            self,
            "Preparation Error",
            f"Failed to prepare project for sending:\n{str(e)}"
        )


def _show_submission_confirmation_dialog(self, host, port):
    """
    Show a confirmation dialog with submission settings before sending to server.

    Args:
        host: Server host address
        port: Server port

    Returns:
        bool: True if user confirmed submission, False if cancelled
    """
    from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QFrame
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont

    # Create dialog
    dialog = QDialog(self)
    dialog.setWindowTitle("Confirm Project Submission")
    dialog.setModal(True)
    dialog.setFixedSize(600, 500)

    layout = QVBoxLayout(dialog)

    # Title
    title_label = QLabel("Project Submission Confirmation")
    title_font = QFont()
    title_font.setPointSize(14)
    title_font.setBold(True)
    title_label.setFont(title_font)
    title_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(title_label)

    # Separator
    separator = QFrame()
    separator.setFrameShape(QFrame.HLine)
    separator.setFrameShadow(QFrame.Sunken)
    layout.addWidget(separator)

    # Create submission details
    details_text = self._build_submission_details(host, port)

    # Details text area
    details_edit = QTextEdit()
    details_edit.setPlainText(details_text)
    details_edit.setReadOnly(True)
    details_edit.setFont(QFont("Consolas", 9))  # Monospace font
    layout.addWidget(details_edit)

    # Warning label
    warning_label = QLabel(
        "⚠️ Verify the settings above before proceeding. This will send your project to the server for processing.")
    warning_label.setWordWrap(True)
    warning_label.setStyleSheet(
        "color: #f57c00; font-weight: bold; padding: 10px; background-color: #fff3e0; border-radius: 5px;")
    layout.addWidget(warning_label)

    # Buttons
    button_layout = QHBoxLayout()

    cancel_btn = QPushButton("Cancel")
    cancel_btn.clicked.connect(dialog.reject)
    button_layout.addWidget(cancel_btn)

    button_layout.addStretch()

    send_btn = QPushButton("Send to Server")
    send_btn.clicked.connect(dialog.accept)
    send_btn.setStyleSheet(
        "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; padding: 8px 16px; }")
    send_btn.setDefault(True)
    button_layout.addWidget(send_btn)

    layout.addLayout(button_layout)

    # Show dialog and return result
    return dialog.exec() == QDialog.Accepted


def _build_submission_details(self, host, port):
    """
    Build detailed submission information for the confirmation dialog.

    Args:
        host: Server host address
        port: Server port

    Returns:
        str: Formatted submission details
    """
    # Get input and output paths
    input_path = self.ui.rawImagesDirectoryLineEdit.text().strip()
    output_path = self.ui.outputDirectoryLineEdit.text().strip()

    # Get image format and settings
    image_format = self.ui.imageFormatComboBox.currentText()

    # Determine bit depth
    bit_depth = "16-bit" if hasattr(self.ui,
                                    'sixteenBitRadioButton') and self.ui.sixteenBitRadioButton.isChecked() else "8-bit"

    # Count images and groups
    image_count = 0
    group_count = 0
    groups = set()

    for i in range(self.ui.imagesListWidget.count()):
        item = self.ui.imagesListWidget.item(i)
        metadata = item.data(Qt.UserRole)

        if metadata.get('is_group_header', False):
            group_count += 1
        else:
            image_count += 1
            group_name = metadata.get('group_name', 'All Images')
            groups.add(group_name)

    # Get image adjustments
    adjustments = []

    # Exposure adjustments
    exposure = self.ui.exposureAdjustmentDoubleSpinBox.value()
    if exposure != 0:
        adjustments.append(f"Exposure: {exposure:+.1f} EV")

    shadows = self.ui.shadowAdjustmentDoubleSpinBox.value()
    if shadows != 0:
        adjustments.append(f"Shadows: {shadows:+.3f}")

    highlights = self.ui.highlightAdjustmentDoubleSpinBox.value()
    if highlights != 0:
        adjustments.append(f"Highlights: {highlights:+.3f}")

    # White balance
    if self.ui.enableWhiteBalanceCheckBox.isChecked():
        wb_temp = self.ui.whitebalanceSpinbox.value()
        adjustments.append(f"White Balance: {wb_temp}K")

    # Denoise
    if self.ui.denoiseImageCheckBox.isChecked():
        denoise_strength = self.ui.denoiseDoubleSpinBox.value()
        adjustments.append(f"Denoise: {denoise_strength:.0f}%")

    # Sharpen
    if self.ui.sharpenImageCheckBox.isChecked():
        sharpen_amount = self.ui.sharpenDoubleSpinBox.value()
        adjustments.append(f"Sharpen: {sharpen_amount:.0f}%")

    # Build the details string
    details = f"""SERVER INFORMATION:
Host: {host}
Port: {port}

INPUT/OUTPUT PATHS:
Input Path: {input_path}
Output Path: {output_path}

IMAGE SETTINGS:
Format: {image_format}
Bit Depth: {bit_depth}

PROJECT STATISTICS:
Image Groups: {len(groups)}
Total Images: {image_count}

IMAGE GROUPS:
{chr(10).join([f"  • {group}" for group in sorted(groups)])}

IMAGE ADJUSTMENTS:"""

    if adjustments:
        details += f"\n{chr(10).join([f'  • {adj}' for adj in adjustments])}"
    else:
        details += "\n  • None (using chart-based color correction only)"

    # Add calibration information
    if hasattr(self, 'group_calibrations') and self.group_calibrations:
        details += f"\n\nCOLOR CALIBRATION:\n  • Group-specific calibrations: {len(self.group_calibrations)} groups"
    elif hasattr(self, 'chart_swatches') and self.chart_swatches is not None:
        details += f"\n\nCOLOR CALIBRATION:\n  • Global chart calibration available"
    else:
        details += f"\n\nCOLOR CALIBRATION:\n  • No calibration data found"

    return details


@Slot(str)
def _on_submission_progress(self, message):
    """Handle progress updates from the submission worker."""
    if hasattr(self, 'progress_dialog') and self.progress_dialog:
        self.progress_dialog.setLabelText(message)
        self.update_server_status_label(message)


@Slot(dict)
def _on_submission_success(self, result):
    """Handle successful submission."""
    response_data = result['response']
    image_count = result['image_count']
    host = result['host']
    port = result['port']

    jobs_created = response_data.get('jobs_created', 0)

    self.log_info(f"[Server] Successfully submitted project with {image_count} images to {host}:{port}")
    self.log_info(f"[Server] Server response: {response_data.get('message', 'Job submitted')}")

    self.update_server_status_label(f"Sent project ({jobs_created} jobs created) - Ready for more")

    from PySide6.QtWidgets import QMessageBox
    QMessageBox.information(
        self,
        "Project Sent Successfully",
        f"Successfully submitted project to server!\n\n"
        f"Server: {host}:{port}\n"
        f"Images processed: {image_count}\n"
        f"Jobs created: {jobs_created}\n\n"
        f"Jobs will be distributed to connected processing clients."
    )


@Slot(str)
def _on_submission_error(self, error_message):
    """Handle submission error."""
    self.update_server_status_label("Send failed - Ready to retry")
    self.log_error(f"[Server] Error sending jobs to server: {error_message}")

    from PySide6.QtWidgets import QMessageBox
    QMessageBox.critical(
        self,
        "Send Error",
        f"Failed to send jobs to server:\n{error_message}"
    )


@Slot()
def _on_submission_finished(self):
    """Handle submission completion (success or error)."""
    if hasattr(self, 'progress_dialog') and self.progress_dialog:
        self.progress_dialog.close()
        self.progress_dialog = None


def update_server_status_label(self, status):
    """Update the server status label with current status and clickable web link."""
    if hasattr(self, 'ui') and hasattr(self.ui, 'serverStatusLabel'):
        # Get server address from settings
        settings = QSettings('ScanSpace', 'ImageProcessor')
        server_address = settings.value('standalone_server_host', 'localhost', type=str)

        # Determine the web interface URL
        if ':' in server_address:
            # If address already has port, extract host and use default web port
            host = server_address.split(':')[0]
        else:
            host = server_address

        # Default web interface port (API server typically runs on 8889)
        web_port = 8889

        # Build the web URL
        if host in ['localhost', '127.0.0.1', '0.0.0.0']:
            # For local addresses, use localhost
            self.server_web_url = f"http://localhost:{web_port}"
        else:
            # For remote addresses, use the actual host
            self.server_web_url = f"http://{host}:{web_port}"

        # Create HTML with clickable link
        if self.dark_mode:
            html_text = (
                f'<span style="color: #d9d6d0;">Server Control Panel: '
                f'<a href="{self.server_web_url}" style="color: #6ab2fa; text-decoration: none;">{self.server_web_url}</a>'
                f' | Server Status: {status}</span>'
            )
        else:
            html_text = (
                f'<span style="color: #414245;">Server Control Panel: '
                f'<a href="{self.server_web_url}" style="color: #6ab2fa; text-decoration: none;">{self.server_web_url}</a>'
                f' | Server Status: {status}</span>'
            )

        # Enable rich text and open external links
        self.ui.serverStatusLabel.setTextFormat(Qt.RichText)
        self.ui.serverStatusLabel.setOpenExternalLinks(True)
        self.ui.serverStatusLabel.setText(html_text)


def _build_project_data(self):
    """
    Build the complete project data structure for export.

    Returns:
        dict: Complete project data ready for JSON serialization
    """
    from datetime import datetime
    import platform

    # Get application version (you may want to define this elsewhere)
    app_version = "1.0.0"  # Update this with actual version

    # Build metadata
    metadata = {
        "export_date": datetime.now().isoformat(),
        "software": "Scan Space Image Processor",
        "software_version": app_version,
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "export_format_version": "1.0"
    }

    # Get current settings
    settings = QSettings('ScanSpace', 'ImageProcessor')

    # get root folder name
    input_folder = self.ui.rawImagesDirectoryLineEdit.text().strip()
    root_folder = os.path.basename(input_folder)

    # Build processing settings
    processing_settings = {
        "output_directory": self.ui.outputDirectoryLineEdit.text(),
        "export_format": self.ui.imageFormatComboBox.currentText() if hasattr(self.ui,
                                                                              'imageFormatComboBox') else ".jpg",
        "thread_count": settings.value('thread_count', 4, type=int),
        "bit_depth_16": settings.value('bit_depth_16_default', False, type=bool),
        "default_colorspace": settings.value('default_colorspace', 'sRGB', type=str),
        "correct_thumbnails": settings.value('correct_thumbnails', False, type=bool),
        "dont_use_chart": self.ui.dontUseColourChartCheckBox.isChecked(),
        "export_schema": settings.value('export_schema', '', type=str),
        "use_export_schema": settings.value('use_export_schema', False, type=bool),
        "custom_name": getattr(self.ui, 'newImageNameLineEdit', None).text() if hasattr(self.ui,
                                                                                        'newImageNameLineEdit') else '',
        "root_folder": root_folder,
        # Image adjustment parameters from UI controls
        "exposure_adj": self.ui.exposureAdjustmentDoubleSpinBox.value(),  # Convert from slider range to EV stops
        "shadow_adj": self.ui.shadowAdjustmentDoubleSpinBox.value(),
        "highlight_adj": self.ui.highlightAdjustmentDoubleSpinBox.value(),
        "white_balance_adj": self.ui.whitebalanceSpinbox.value(),
        "enable_white_balance": self.ui.enableWhiteBalanceCheckBox.isChecked(),
        "denoise_strength": self.ui.denoiseDoubleSpinBox.value(),
        "sharpen_amount": self.ui.sharpenDoubleSpinBox.value(),
        # Export format parameters that were missing
        "jpeg_quality": settings.value('jpeg_quality', 100, type=int),
        "output_format": self.ui.imageFormatComboBox.currentText() if hasattr(self.ui,
                                                                              'imageFormatComboBox') else ".jpg",
        "tiff_bitdepth": settings.value('tiff_bitdepth', 8, type=int),
        "exr_colorspace": settings.value('exr_colorspace', 'sRGB', type=str)
    }

    # Build chart configuration
    chart_config = {
        "use_precalculated_charts": settings.value('use_precalculated_charts', False, type=bool),
        "chart_folder_path": settings.value('chart_folder_path', '', type=str),
        "selected_precalc_chart": settings.value('selected_precalc_chart', '', type=str),
        "manual_chart_path": getattr(self, 'calibration_file', None),
        "has_chart_swatches": hasattr(self, 'chart_swatches') and self.chart_swatches is not None
    }

    # Build image data
    images = []
    for i in range(self.ui.imagesListWidget.count()):
        item = self.ui.imagesListWidget.item(i)
        if item:
            image_metadata = item.data(Qt.UserRole)

            # Skip group headers - they have is_group_header flag set to True
            if image_metadata and isinstance(image_metadata, dict) and image_metadata.get('is_group_header', False):
                continue

            # For server compatibility, full_path should be the file path string
            if image_metadata and isinstance(image_metadata, dict):
                # Extract the input_path as the full_path
                full_path = image_metadata.get('input_path', '')
                group_name = image_metadata.get('group_name', 'All Images')
            else:
                # If metadata is a string or None (shouldn't happen but handle gracefully)
                full_path = image_metadata if isinstance(metadata, str) else ""
                group_name = 'All Images'

            # Only add images that have a valid input path (skip invalid entries)
            if not full_path:
                continue

            image_data = {
                "index": i,
                "filename": item.text(),
                "full_path": full_path,  # Server expects this to be a string path (input_path)
                # "metadata": metadata_copy,  # Complete metadata for export compatibility
                "group": group_name,  # Use group name from metadata
                "selected": True,  # Mark all images as selected for processing
                "has_user_data": item.data(Qt.UserRole + 1) is not None,
                "user_data_keys": list(item.data(Qt.UserRole + 1).keys()) if item.data(Qt.UserRole + 1) else []
            }
            images.append(image_data)

    # Build image groups data with chart swatches
    image_groups = {}

    # First, collect all unique groups from the images
    all_groups = set()
    for image in images:
        all_groups.add(image['group'])

    # Find available calibration data
    available_calibrations = {}
    fallback_calibration = None

    dont_use_chart = self.ui.dontUseColourChartCheckBox.isChecked()

    # Skip chart calibration collection if "don't use chart" is checked
    if not dont_use_chart:
        if hasattr(self, 'group_calibrations') and self.group_calibrations:
            for group_name, calibration_data in self.group_calibrations.items():
                if calibration_data and 'swatches' in calibration_data:
                    available_calibrations[group_name] = calibration_data
                    if fallback_calibration is None:
                        fallback_calibration = calibration_data

        # Check for global chart swatches as fallback
        if not available_calibrations and hasattr(self, 'chart_swatches') and self.chart_swatches is not None:
            fallback_calibration = {
                'swatches': self.chart_swatches,
                'file': getattr(self, 'calibration_file', '')
            }

        # If no calibration data found anywhere, show error dialog
        if not available_calibrations and not fallback_calibration:
            from PySide6.QtWidgets import QMessageBox
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setWindowTitle("No Chart Calibration Found")
            msg.setText("No color chart calibration data was found in the scene.")
            msg.setInformativeText("Please load a color chart calibration before exporting to the server.")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec()
            return None
    else:
        # When don't use chart is checked, log that we're skipping calibration
        self.log_debug("Chart calibration disabled - 'Don't use chart' is checked")

    # Build image groups with calibration data
    for group_name in all_groups:

        # Skip calibration if dont_use_chart is checked
        if dont_use_chart:
            calibration_data = None
        else:
            # Use group-specific calibration if available, otherwise use fallback
            calibration_data = available_calibrations.get(group_name, fallback_calibration)

        if calibration_data and 'swatches' in calibration_data:
            # Convert numpy array to list for JSON serialization
            swatches_array = calibration_data['swatches']

            # DIAGNOSTIC: Deep analysis of swatches data structure

            if hasattr(swatches_array, 'shape'):
                array_size_mb = (swatches_array.nbytes / 1024 / 1024) if hasattr(swatches_array, 'nbytes') else 0
                # Check if this is massive data that shouldn't be here
                if array_size_mb > 50:  # More than 50MB is definitely wrong
                    # Show detailed shape analysis
                    if len(swatches_array.shape) > 2:
                        self.log_error(f"ERROR: Array has {len(swatches_array.shape)} dimensions, expected 2 (24, 3)")

                    # Create emergency fallback with just 24 color values
                    fallback_colors = np.array([
                        [0.4, 0.3, 0.2], [0.7, 0.5, 0.4], [0.3, 0.4, 0.6], [0.2, 0.3, 0.2],
                        [0.5, 0.5, 0.7], [0.3, 0.7, 0.6], [0.8, 0.4, 0.2], [0.2, 0.2, 0.5],
                        [0.7, 0.3, 0.4], [0.3, 0.2, 0.4], [0.6, 0.7, 0.3], [0.8, 0.6, 0.2],
                        [0.2, 0.3, 0.6], [0.3, 0.5, 0.3], [0.6, 0.2, 0.2], [0.9, 0.8, 0.3],
                        [0.7, 0.3, 0.6], [0.2, 0.5, 0.6], [0.9, 0.9, 0.9], [0.6, 0.6, 0.6],
                        [0.4, 0.4, 0.4], [0.2, 0.2, 0.2], [0.05, 0.05, 0.05], [0.0, 0.0, 0.0]
                    ], dtype=np.float32)
                    swatches_array = fallback_colors

            start_time = time.time()
            try:
                # Convert to list for JSON serialization
                if hasattr(swatches_array, 'tolist'):
                    swatches_list = swatches_array.tolist()
                elif isinstance(swatches_array, (list, tuple)):
                    swatches_list = list(swatches_array)
                else:
                    swatches_list = list(swatches_array)

                # Validate and trim if necessary
                if len(swatches_list) != 24:
                    if len(swatches_list) > 24:
                        swatches_list = swatches_list[:24]

                # Round for efficiency
                if swatches_list and isinstance(swatches_list[0], (list, tuple, np.ndarray)):
                    swatches_list = [[round(float(c), 6) for c in swatch] for swatch in swatches_list]

            except Exception as e:
                self.log_warning(f"[Project] Failed to convert swatches for group {group_name}: {e}")
                swatches_list = []

            conversion_time = time.time() - start_time

            # Determine if this is using fallback calibration
            is_fallback = (group_name not in available_calibrations)

            image_groups[group_name] = {
                "has_calibration": True,
                "chart_file": calibration_data.get('file', ''),
                "chart_swatches": swatches_list,
                "chart_swatches_count": len(swatches_list),
                "using_fallback_calibration": is_fallback
            }

            if is_fallback:
                self.log_debug(f"Group {group_name} using fallback calibration")
        else:
            image_groups[group_name] = {
                "has_calibration": False,
                "chart_file": '',
                "chart_swatches": [],
                "chart_swatches_count": 0,
                "using_fallback_calibration": False
            }

    # Import/Export settings
    import_export_settings = {
        "look_in_subfolders": settings.value('look_in_subfolders', False, type=bool),
        "group_by_subfolder": settings.value('group_by_subfolder', False, type=bool),
        "group_by_prefix": settings.value('group_by_prefix', False, type=bool),
        "prefix_string": settings.value('prefix_string', '', type=str),
        "ignore_formats": settings.value('ignore_formats', False, type=bool),
        "ignore_string": settings.value('ignore_string', '', type=str),
        "use_import_rules": settings.value('use_import_rules', False, type=bool)
    }

    # Network settings (if applicable)
    network_settings = {
        "network_mode": getattr(self, 'network_mode', 'local'),
        "enable_server": settings.value('enable_server', False, type=bool),
        "is_host_server": settings.value('is_host_server', True, type=bool),
        "process_on_host": settings.value('process_on_host', True, type=bool),
        "server_address": settings.value('server_address', '', type=str),
        "host_server_ip": settings.value('host_server_ip', '', type=str),
        "standalone_server_host": settings.value('standalone_server_host', 'localhost', type=str),
        "standalone_server_port": settings.value('standalone_server_port', 8889, type=int)
    }

    # Build complete project structure
    project_data = {
        "metadata": metadata,
        "processing_settings": processing_settings,
        "chart_configuration": chart_config,
        "import_export_settings": import_export_settings,
        "network_settings": network_settings,
        "images": images,
        "image_groups": image_groups,
        "raw_images_directory": self.ui.rawImagesDirectoryLineEdit.text(),
        "total_images": len(images),
        "selected_images": sum(1 for img in images if img["selected"])
    }

    return project_data


def _sanitize_project_data(self, data):
    """
    Recursively sanitize project data to remove emojis and non-standard characters.

    This ensures JSON export compatibility and prevents encoding issues when
    the project file is loaded on different systems.

    Args:
        data: The data structure to sanitize (dict, list, str, or other)

    Returns:
        Sanitized data structure with emojis and non-standard characters removed
    """
    import re

    def sanitize_string(text):
        """Remove emojis and non-standard characters from a string."""
        if not isinstance(text, str):
            return text

        # Remove emojis using regex
        # This pattern matches most emoji ranges in Unicode
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"  # enclosed characters
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U0001F018-\U0001F270"  # various symbols
            "\U0001F300-\U0001F5FF"  # misc symbols
            "]+",
            flags=re.UNICODE
        )

        # Remove emojis
        text = emoji_pattern.sub('', text)

        # Remove other non-printable characters but keep basic punctuation and newlines
        # Keep: letters, numbers, spaces, basic punctuation, newlines, tabs
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\_\(\)\[\]\{\}\"\'\/\\\r\n\t]', '', text)

        # Clean up multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def sanitize_recursive(obj):
        """Recursively sanitize data structures."""
        if isinstance(obj, dict):
            return {sanitize_string(k): sanitize_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize_recursive(item) for item in obj]
        elif isinstance(obj, str):
            return sanitize_string(obj)
        elif hasattr(obj, 'tolist'):
            # Handle numpy arrays by converting to list
            try:
                # Convert to list but don't recursively sanitize -
                # chart swatches are just numbers, no strings to sanitize
                return obj.tolist()
            except Exception:
                return obj
        else:
            # For other types (int, float, bool, None), return as-is
            return obj

    return sanitize_recursive(data)


def _debug_find_unserializable_objects(self, obj, path="root"):
    """Debug helper to find non-JSON-serializable objects."""
    import json
    try:
        json.dumps(obj)
    except TypeError as e:
        if isinstance(obj, dict):
            for key, value in obj.items():
                try:
                    json.dumps(value)
                except TypeError:
                    self.log_error(f"[Debug] Non-serializable object at {path}.{key}: {type(value)}")
                    if hasattr(value, 'shape'):  # Likely numpy array
                        self.log_error(
                            f"[Debug] Object shape: {value.shape}, dtype: {getattr(value, 'dtype', 'unknown')}")
                    self._debug_find_unserializable_objects(value, f"{path}.{key}")
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                try:
                    json.dumps(item)
                except TypeError:
                    self.log_error(f"[Debug] Non-serializable object at {path}[{i}]: {type(item)}")
                    if hasattr(item, 'shape'):  # Likely numpy array
                        self.log_error(
                            f"[Debug] Object shape: {item.shape}, dtype: {getattr(item, 'dtype', 'unknown')}")
                    self._debug_find_unserializable_objects(item, f"{path}[{i}]")
        else:
            self.log_error(f"[Debug] Non-serializable leaf object at {path}: {type(obj)}")
            if hasattr(obj, 'shape'):  # Likely numpy array
                self.log_error(f"[Debug] Object shape: {obj.shape}, dtype: {getattr(obj, 'dtype', 'unknown')}")
