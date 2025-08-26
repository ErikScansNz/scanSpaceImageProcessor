"""
Chart Tools Module

This module contains all functions related to manual chart selection and chart flattening operations.
Extracted from the main image_space.py file for better code organization.

Functions include:
- Manual chart selection with rubber band interface
- Chart flattening with 4-corner perspective correction
- UI helper classes for visual markers
- Chart swatch grid drawing utilities
"""

import functools
import gc
import os

import colour
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QRubberBand, QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsRectItem, QLabel, QGraphicsPixmapItem
)
from PySide6.QtCore import Qt, QRect, QSize
from PySide6.QtGui import QColor, QImage, QPixmap, QPainter
from PySide6.QtSvg import QSvgRenderer


class FixedSizeText(QGraphicsTextItem):
    """Graphics text item that ignores transformations for consistent display."""
    
    def __init__(self, text, x, y, color=QColor('red'), parent=None):
        super().__init__(text, parent)
        self.setPos(x, y)
        self.setDefaultTextColor(color)
        self.setFlag(QGraphicsTextItem.ItemIgnoresTransformations, True)


class FixedSizeEllipse(QGraphicsEllipseItem):
    """Graphics ellipse item that ignores transformations for consistent display."""
    
    def __init__(self, x, y, radius=8, color=QColor('red'), parent=None):
        super().__init__(parent)
        self.setPos(x, y)
        self.radius = radius
        self.color = color
        self.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)
        # The ellipse is centered on (x, y)
        self.setRect(-radius/2, -radius/2, radius, radius)
    
    def paint(self, painter, option, widget):
        painter.setBrush(self.color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(self.rect())


class ChartTools:
    """
    Chart manipulation tools for manual selection and perspective correction.
    
    This class contains methods for:
    - Manual chart region selection using rubber band interface
    - 4-corner perspective flattening with OpenCV
    - ColorChecker swatch grid overlay
    - UI state management for chart tools
    """
    
    @staticmethod
    def create_colored_svg_pixmap(svg_path: str, color: QColor, size: QSize = QSize(16, 16)) -> QPixmap:
        """
        Create a properly antialiased colored SVG pixmap.
        
        Args:
            svg_path: Path to the SVG file
            color: Color to apply to the SVG
            size: Desired size of the output pixmap
            
        Returns:
            QPixmap: Colored SVG rendered as pixmap with proper antialiasing
        """
        # Create SVG renderer
        renderer = QSvgRenderer(svg_path)
        if not renderer.isValid():
            # Fallback to simple circle if SVG fails to load
            pixmap = QPixmap(size)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(2, 2, size.width()-4, size.height()-4)
            painter.end()
            return pixmap
        
        # First render SVG in black to create a mask
        mask_pixmap = QPixmap(size)
        mask_pixmap.fill(Qt.transparent)
        painter = QPainter(mask_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        renderer.render(painter)
        painter.end()
        
        # Create final colored pixmap
        colored_pixmap = QPixmap(size)
        colored_pixmap.fill(Qt.transparent)
        painter = QPainter(colored_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Draw the colored shape using the SVG as a mask
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.drawPixmap(0, 0, mask_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(colored_pixmap.rect(), color)
        
        painter.end()
        return colored_pixmap
    
    @staticmethod
    def exit_manual_mode(func):
        """Decorator that exits manual mode before executing the wrapped function."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, 'manual_selection_mode', False):
                ChartTools.finalize_manual_chart_selection(self, canceled=True)
            result = func(self, *args, **kwargs)
            return result
        return wrapper
    
    @staticmethod
    def finalize_manual_chart_selection(main_window, *, canceled: bool = False):
        """
        Exit manual chart selection (either commit or cancel) and restore UI.

        Args:
            main_window: Reference to the main application window
            canceled: if True, we're aborting manual mode and do NOT commit temp_swatches.
                     if False, this is a real "Finalize Chart" and we copy temp_swatches to chart_swatches.
        """
        # 1) Log & commit (only if finalize, not cancel)
        if canceled:
            main_window.log("[Manual] Manual chart selection canceled.")
        else:
            main_window.log("[Manual] Manual chart selection finalized.")
            # commit the temporary swatches/calibration
            if hasattr(main_window, 'temp_swatches'):
                main_window.chart_swatches = main_window.temp_swatches
            if hasattr(main_window, 'temp_calibration_file'):
                main_window.calibration_file = main_window.temp_calibration_file

        # 2) Reset internal modes & data
        main_window.manual_selection_mode = False
        main_window.flatten_mode = False
        if hasattr(main_window, 'corner_points'):
            main_window.corner_points.clear()
        main_window.flatten_swatch_rects = None

        # 3) Hide any visible rubberband
        if hasattr(main_window, 'rubberBand') and main_window.rubberBand.isVisible():
            main_window.rubberBand.hide()

        # 4) Restore the *original* preview if we have it
        if hasattr(main_window, 'original_preview_pixmap') and main_window.original_preview_pixmap:
            main_window._display_preview(main_window.original_preview_pixmap)
            main_window.showing_chart_preview = False
            main_window.ui.showOriginalImagePushbutton.setText("Show Chart Preview")

        # 5) Disable & reset style of all charttools buttons
        ChartTools.set_chart_tools_enabled(main_window)

        # 6) Hide the toolshelf & instruction overlay
        main_window.ui.detectChartToolshelfFrame.setVisible(False)
        main_window.ui.chartInformationLabel.setVisible(False)

        # 7) Hide debug frame if present
        main_window.show_debug_frame(False)
    
    @staticmethod
    def set_chart_tools_enabled(main_window, *, manual=True, detect=False, show=False, flatten=False, finalize=False):
        """
        Enable or disable all chart-tools buttons with optional highlight on Manual & Flatten.
        
        Args:
            main_window: Reference to the main application window
            manual: Enable manual selection button
            detect: Enable detect chart button
            show: Enable show original image button
            flatten: Enable flatten chart button
            finalize: Enable finalize chart button
        """
        mapping = {
            'manual': main_window.ui.manuallySelectChartPushbutton,
            'detect':   main_window.ui.detectChartShelfPushbutton,
            'show':     main_window.ui.showOriginalImagePushbutton,
            'flatten':  main_window.ui.flattenChartImagePushButton,
            'finalize': main_window.ui.finalizeChartPushbutton,
        }
        
        local_vars = locals()
        for key, btn in mapping.items():
            enabled = local_vars[key]
            btn.setEnabled(enabled)

    
    @staticmethod
    def manually_select_chart(main_window):
        """
        Enter manual chart selection mode:
          - Load RAW + thumbnail, reset UI, show instructions, enable only Manual-Select.
        
        Args:
            main_window: Reference to the main application window
        """
        chart_text = main_window.ui.chartPathLineEdit.text().strip()
        if not chart_text:
            main_window.log('[Manual] No chart selected')
            return
        cursor_pixmap = QPixmap(r'resources/icons/crosshair.svg')
        main_window.cursor = cursor_pixmap
        main_window.ui.imagePreviewGraphicsView.setCursor(main_window.cursor)
        
        # Extract actual file path from formatted text like "[Group Name] filename.raw"
        if chart_text.startswith('[') and '] ' in chart_text:
            # Format: "[Group Name] filename.raw" -> get the actual path from chart_groups
            selected_item = main_window.ui.imagesListWidget.currentItem()
            if selected_item:
                meta = selected_item.data(Qt.UserRole)
                if not meta.get('is_group_header', False):
                    path = meta['input_path']
                else:
                    main_window.log('[Manual] Cannot select chart from group header')
                    return
            else:
                main_window.log('[Manual] No image selected')
                return
        else:
            # Direct file path
            path = chart_text
        
        full_fp = main_window.load_raw_image(path)
        if full_fp is None:
            main_window.log(f'[Manual] RAW load failed: {path}')
            return
        main_window.fp_image_array = full_fp

        # build 8-bit thumbnail with proper gamma correction
        # Create a safe copy of full_fp to avoid memory conflicts during color correction
        full_fp_copy = full_fp.copy()
        thumb_corrected = colour.cctf_encoding(full_fp_copy)
        thumb_arr = np.uint8(255 * np.clip(thumb_corrected, 0.0, 1.0))
        
        # Create QImage BEFORE deleting intermediate arrays
        h, w, _ = thumb_arr.shape
        bytes_per_line = w * 3
        qimg = QImage(thumb_arr.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        main_window.original_preview_pixmap = pixmap
        main_window.current_image_pixmap = pixmap
        main_window._display_preview(pixmap)

        # Disable background RAW loading to prevent conflicts during chart selection
        main_window._clear_preview_state()
        main_window.raw_load_timer.stop()
        
        # Reset modes and hide overlays/debug
        main_window.manual_selection_mode = True
        main_window.flatten_mode = False
        main_window.show_debug_frame(False)
        if hasattr(main_window, 'rubberBand') and main_window.rubberBand.isVisible():
            main_window.rubberBand.hide()

        # Reset and enable only the Manual-Select button
        ChartTools.set_chart_tools_enabled(main_window, manual=True)

        # Show instruction label
        main_window.ui.chartInformationLabel.setVisible(True)
        main_window.ui.chartInformationLabel.setText("Click and drag box around the colour chart")

        # Reveal toolshelf
        main_window.ui.detectChartToolshelfFrame.setVisible(True)
    
    @staticmethod
    def on_manual_crop_complete(main_window, rect: QRect):
        """
        Called when the QRect is drawn in manual crop mode
        Enables the buttons for Show Original Image, Flatten Chart Mode and Detect Chart
        
        Args:
            main_window: Reference to the main application window
            rect: The selected rectangle area
        """
        main_window.ui.detectChartShelfPushbutton.setEnabled(True)
        main_window.ui.showOriginalImagePushbutton.setEnabled(True)
        main_window.ui.imagePreviewGraphicsView.unsetCursor()

        main_window.ui.flattenChartImagePushButton.setEnabled(True)
    
    @staticmethod
    def flatten_chart_image(main_window):
        """
        Enter cornerpicking mode to flatten the manually selected chart region;
        on completion, stores the cropped-preview pixmap and enables flattening.
        
        Args:
            main_window: Reference to the main application window
        """
        if not hasattr(main_window, 'cropped_preview_pixmap'):
            main_window.log('[Flatten] No cropped image, select region first')
            return

        main_window.log('[Flatten] Select the 4 corners of the chart')
        main_window.ui.chartInformationLabel.setText('Select the 4 corners of the chart')

        cursor_pixmap = QPixmap(r'resources/icons/crosshair.svg')
        main_window.cursor = cursor_pixmap
        main_window.ui.imagePreviewGraphicsView.setCursor(main_window.cursor)

        main_window.flatten_mode = True
        main_window.corner_points = []

        main_window._display_preview(main_window.cropped_preview_pixmap)
        # enable only Flatten & Finalize
        ChartTools.set_chart_tools_enabled(main_window, flatten=True, finalize=True)
    
    @staticmethod
    def handle_manual_press(main_window, event):
        """
        Handle mouse press event for manual chart selection rubber band.
        
        Args:
            main_window: Reference to the main application window
            event: Mouse press event
            
        Returns:
            bool: True if event was handled
        """
        main_window.origin = event.position().toPoint()
        if not hasattr(main_window, 'rubberBand'):
            main_window.rubberBand = QRubberBand(QRubberBand.Rectangle, main_window.ui.imagePreviewGraphicsView)
        main_window.rubberBand.setGeometry(QRect(main_window.origin, QSize()))
        main_window.rubberBand.show()
        return True
    
    @staticmethod
    def handle_manual_move(main_window, event):
        """
        Handle mouse move event for manual chart selection rubber band.
        
        Args:
            main_window: Reference to the main application window
            event: Mouse move event
            
        Returns:
            bool: True if event was handled
        """
        if hasattr(main_window, 'rubberBand') and main_window.rubberBand.isVisible():
            rect = QRect(main_window.origin, event.position().toPoint()).normalized()
            main_window.rubberBand.setGeometry(rect)
            return True
        return False
    
    @staticmethod
    def handle_manual_release(main_window, event):
        """
        Handle mouse release event for manual chart selection rubber band.
        
        Args:
            main_window: Reference to the main application window
            event: Mouse release event
            
        Returns:
            bool: True if event was handled
        """
        if not hasattr(main_window, 'rubberBand') or not main_window.rubberBand.isVisible():
            return False
            
        # Map rubber-band rect to scene coords
        rect = main_window.rubberBand.geometry()
        tl = main_window.ui.imagePreviewGraphicsView.mapToScene(rect.topLeft())
        br = main_window.ui.imagePreviewGraphicsView.mapToScene(rect.bottomRight())
        x, y = int(tl.x()), int(tl.y())
        w, h = abs(int(br.x() - tl.x())), abs(int(br.y() - tl.y()))

        # Constrain crop region to image boundaries to prevent out-of-bounds access
        img_height, img_width = main_window.fp_image_array.shape[:2]
        pixmap_width = main_window.current_image_pixmap.width()
        pixmap_height = main_window.current_image_pixmap.height()
        
        # Clamp coordinates to valid ranges
        x = max(0, min(x, pixmap_width - 1))
        y = max(0, min(y, pixmap_height - 1))
        w = min(w, pixmap_width - x)
        h = min(h, pixmap_height - y)
        
        # Ensure minimum crop size
        w = max(1, w)
        h = max(1, h)
        
        # For the float array, ensure we don't exceed its bounds
        fp_x = max(0, min(x, img_width - 1))
        fp_y = max(0, min(y, img_height - 1))
        fp_w = min(w, img_width - fp_x)
        fp_h = min(h, img_height - fp_y)
        fp_w = max(1, fp_w)
        fp_h = max(1, fp_h)

        main_window.rubberBand.hide()
        main_window.manual_selection_mode = False

        # Crop data with bounds checking
        main_window.cropped_preview_pixmap = main_window.current_image_pixmap.copy(x, y, w, h)
        main_window.cropped_fp = main_window.fp_image_array[fp_y:fp_y+fp_h, fp_x:fp_x+fp_w, :]

        main_window._display_preview(main_window.cropped_preview_pixmap)
        ChartTools.set_chart_tools_enabled(main_window, detect=True, flatten=True)
        return True
    
    @staticmethod
    def handle_flatten_press(main_window, event):
        """
        Handle mouse press event for chart flattening corner selection.
        
        Args:
            main_window: Reference to the main application window
            event: Mouse press event
            
        Returns:
            bool: True if event was handled
        """
        # Create properly antialiased colored SVG icon
        point_colour = QColor(255, 0, 0)  # Red
        colored_point_pixmap = ChartTools.create_colored_svg_pixmap(
            r'resources/icons/circle-dot.svg', 
            point_colour, 
            QSize(20, 20)  # Larger size for better visibility
        )

        pt = event.position().toPoint()
        sp = main_window.ui.imagePreviewGraphicsView.mapToScene(pt)
        idx = len(main_window.corner_points) + 1
        main_window.corner_points.append((sp.x(), sp.y()))
        main_window.log(f"[Flatten] Point {idx}: ({int(sp.x())}, {int(sp.y())})")

        # Draw UI markers  
        label = FixedSizeText(str(idx), sp.x()+12, sp.y()-8, color=QColor('red'))
        
        # Convert QPixmap to QGraphicsPixmapItem and position it
        point_item = QGraphicsPixmapItem(colored_point_pixmap)
        point_item.setPos(sp.x() - colored_point_pixmap.width()/2, sp.y() - colored_point_pixmap.height()/2)
        
        main_window.previewScene.addItem(point_item)
        main_window.previewScene.addItem(label)

        # Once four points selected, perform flatten
        if idx == 4:
            ChartTools.perform_flatten_transform(main_window)
        return True
    
    @staticmethod
    def detect_chart_rotation_from_corners(corner_points, warped_image, swatch_width, swatch_height, padding):
        """
        Detect chart rotation using corner point distances and gradient analysis.
        
        Uses the corner points to determine if we have a 6x4 or 4x6 orientation,
        then finds the white-to-black gradient and calculates the rotation needed.
        
        Args:
            corner_points: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            warped_image: numpy array of the warped chart image  
            swatch_width: width of each swatch in pixels
            swatch_height: height of each swatch in pixels
            padding: padding around the chart
            
        Returns:
            int: rotation needed in 90-degree steps (0, 1, 2, 3 for 0°, 90°, 180°, 270°)
        """
        
        def distance(p1, p2):
            """Calculate Euclidean distance between two points"""
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        def get_swatch_lightness(image, row, col, sw_w, sw_h, pad):
            """Extract average lightness from a swatch position"""
            x_start = pad + col * sw_w + sw_w // 4
            x_end = pad + col * sw_w + 3 * sw_w // 4
            y_start = pad + row * sw_h + sw_h // 4
            y_end = pad + row * sw_h + 3 * sw_h // 4
            
            # Bounds checking for image dimensions
            img_h, img_w = image.shape[:2]
            if x_end >= img_w or y_end >= img_h or x_start < 0 or y_start < 0:
                return None
            
            # Extract center region of swatch to avoid edge artifacts
            swatch_region = image[y_start:y_end, x_start:x_end]
            if swatch_region.size == 0:
                return None
            
            # Calculate perceptual lightness (Y in YUV)
            avg_color = np.mean(swatch_region, axis=(0, 1))
            return 0.299 * avg_color[0] + 0.587 * avg_color[1] + 0.114 * avg_color[2]
        
        def check_gradient_quality(lightness_values):
            """Check how well a sequence matches white-to-black gradient"""
            valid_values = [v for v in lightness_values if v is not None]
            if len(valid_values) != 6:
                return float('inf')
            
            total_decrease = 0
            violations = 0
            
            for i in range(len(valid_values) - 1):
                diff = valid_values[i] - valid_values[i + 1]
                if diff > 0:
                    total_decrease += diff
                else:
                    violations += 1
            
            lightness_range = max(valid_values) - min(valid_values)
            score = violations * 100 - total_decrease - lightness_range
            
            if lightness_range < 80:
                score += 200
                
            return score
        
        # Calculate distances between adjacent corner points to determine orientation
        # Points are: [top-left, top-right, bottom-right, bottom-left] (clockwise)
        side1_dist = distance(corner_points[0], corner_points[1])  # top edge
        side2_dist = distance(corner_points[1], corner_points[2])  # right edge
        
        print(f"[DEBUG] Corner distances: side1={side1_dist:.1f}, side2={side2_dist:.1f}")
        
        # Determine if we have a landscape (6x4) or portrait (4x6) orientation
        if side1_dist > side2_dist:
            # Landscape: longer horizontal edge = 6 swatches wide, 4 tall
            orientation = '6x4'
            sw_w = int(swatch_width / 6)
            sw_h = int(swatch_height / 4)
            print(f"[DEBUG] Detected 6x4 orientation (landscape)")
        else:
            # Portrait: longer vertical edge = 4 swatches wide, 6 tall  
            orientation = '4x6'
            sw_w = int(swatch_width / 4)
            sw_h = int(swatch_height / 6)
            print(f"[DEBUG] Detected 4x6 orientation (portrait)")
        
        # Now find the 6-swatch white-to-black gradient
        best_rotation = 0
        best_score = float('inf')
        best_edge_name = None
        
        if orientation == '6x4':
            # Check horizontal edges (6 swatches each)
            edges = [
                ('bottom', [(3, col) for col in range(6)]),
                ('top', [(0, col) for col in range(5, -1, -1)])  # reversed for gradient direction
            ]
        else:  # 4x6
            # Check vertical edges (6 swatches each)
            edges = [
                ('right', [(row, 3) for row in range(6)]),
                ('left', [(row, 0) for row in range(5, -1, -1)])  # reversed for gradient direction
            ]
        
        for edge_name, positions in edges:
            lightness_values = []
            for row, col in positions:
                lightness = get_swatch_lightness(warped_image, row, col, sw_w, sw_h, padding)
                lightness_values.append(lightness)
            
            valid_count = len([v for v in lightness_values if v is not None])
            if valid_count == 6:
                score = check_gradient_quality(lightness_values)
                
                if score < best_score:
                    best_score = score
                    best_edge_name = edge_name
                    
                    valid_lightness = [v for v in lightness_values if v is not None]
                    print(f"[DEBUG] Best gradient found: {orientation} {edge_name} edge")
                    print(f"[DEBUG] Lightness values: {[round(v, 1) for v in valid_lightness]}")
                    print(f"[DEBUG] Score: {score}")
        
        # Calculate rotation needed
        if orientation == '6x4':
            if best_edge_name == 'bottom':
                best_rotation = 0  # Already correct
            elif best_edge_name == 'top':
                best_rotation = 2  # 180° - flip top to bottom
        else:  # 4x6
            if best_edge_name == 'right':
                best_rotation = 1  # 90° clockwise - right becomes bottom
            elif best_edge_name == 'left':
                best_rotation = 3  # 90° counterclockwise - left becomes bottom
        
        print(f"[DEBUG] Final rotation: {best_rotation} ({best_rotation * 90}°)")
        return best_rotation

    @staticmethod
    def detect_chart_rotation(warped_image, swatch_width, swatch_height, padding):
        """
        Detect chart rotation by finding the white-to-black gradient in both 6x4 and 4x6 orientations.
        
        ColorChecker24 has a distinctive 6-swatch white-to-black gradient. We check both possible
        orientations and all edges to find where this gradient exists, then calculate the
        rotation needed to move it to the bottom row.
        
        Args:
            warped_image: numpy array of the warped chart image
            swatch_width: width of each swatch in pixels
            swatch_height: height of each swatch in pixels
            padding: padding around the chart
            
        Returns:
            int: rotation needed in 90-degree steps (0, 1, 2, 3 for 0°, 90°, 180°, 270°)
        """
        
        def get_swatch_lightness(image, row, col, sw_w, sw_h, pad):
            """Extract average lightness from a swatch position"""
            x_start = pad + col * sw_w + sw_w // 4
            x_end = pad + col * sw_w + 3 * sw_w // 4
            y_start = pad + row * sw_h + sw_h // 4
            y_end = pad + row * sw_h + 3 * sw_h // 4
            
            # Bounds checking for image dimensions
            img_h, img_w = image.shape[:2]
            if x_end >= img_w or y_end >= img_h or x_start < 0 or y_start < 0:
                return None
            
            # Extract center region of swatch to avoid edge artifacts
            swatch_region = image[y_start:y_end, x_start:x_end]
            if swatch_region.size == 0:
                return None
            
            # Calculate perceptual lightness (Y in YUV)
            avg_color = np.mean(swatch_region, axis=(0, 1))
            return 0.299 * avg_color[0] + 0.587 * avg_color[1] + 0.114 * avg_color[2]
        
        def check_gradient_quality(lightness_values):
            """
            Check how well a sequence matches white-to-black gradient.
            Returns a score where lower is better.
            """
            # Filter out None values and check length
            valid_values = [v for v in lightness_values if v is not None]
            if len(valid_values) != 6:  # We only care about 6-swatch gradients
                return float('inf')
            
            # Calculate how much lightness decreases across the sequence
            total_decrease = 0
            violations = 0
            
            for i in range(len(valid_values) - 1):
                diff = valid_values[i] - valid_values[i + 1]
                if diff > 0:  # Decreasing (good)
                    total_decrease += diff
                else:  # Increasing (bad)
                    violations += 1
            
            # Good gradient should have high total decrease and few violations
            # Also check if we have sufficient contrast (white to black range)
            lightness_range = max(valid_values) - min(valid_values)
            
            # Score: penalize violations heavily, reward total decrease and range
            score = violations * 100 - total_decrease - lightness_range
            
            # Extra penalty if range is too small (not really white-to-black)
            if lightness_range < 80:
                score += 200
                
            return score
        
        # Test both 6x4 and 4x6 orientations to find the 6-swatch gradient
        orientations = [
            {
                'name': '6x4',
                'sw_w': int(swatch_width / 6),
                'sw_h': int(swatch_height / 4),
                'edges': [
                    ('bottom', [(3, col) for col in range(6)]),           # Bottom row: 6 swatches
                    ('top', [(0, col) for col in range(5, -1, -1)]),     # Top row: 6 swatches (reversed)
                    ('right', [(row, 5) for row in range(4)]),           # Right col: 4 swatches 
                    ('left', [(row, 0) for row in range(3, -1, -1)])     # Left col: 4 swatches (reversed)
                ]
            },
            {
                'name': '4x6',
                'sw_w': int(swatch_width / 4),
                'sw_h': int(swatch_height / 6),
                'edges': [
                    ('bottom', [(5, col) for col in range(4)]),          # Bottom row: 4 swatches
                    ('top', [(0, col) for col in range(3, -1, -1)]),     # Top row: 4 swatches (reversed)
                    ('right', [(row, 3) for row in range(6)]),           # Right col: 6 swatches
                    ('left', [(row, 0) for row in range(5, -1, -1)])     # Left col: 6 swatches (reversed)
                ]
            }
        ]
        
        best_rotation = 0
        best_score = float('inf')
        
        for orientation in orientations:
            sw_w = orientation['sw_w']
            sw_h = orientation['sw_h']
            
            for edge_name, positions in orientation['edges']:
                # Extract lightness values for this edge
                lightness_values = []
                for row, col in positions:
                    lightness = get_swatch_lightness(warped_image, row, col, sw_w, sw_h, padding)
                    lightness_values.append(lightness)
                
                # Only consider edges with 6 valid swatches (the gradient we're looking for)
                valid_count = len([v for v in lightness_values if v is not None])
                if valid_count == 6:
                    score = check_gradient_quality(lightness_values)
                    
                    if score < best_score:
                        best_score = score

                        # Calculate rotation needed to move this edge to bottom
                        if orientation['name'] == '6x4':
                            # For 6x4 orientation, we want the 6-swatch edge at bottom
                            if edge_name == 'bottom':
                                best_rotation = 0  # Already correct
                            elif edge_name == 'top':
                                best_rotation = 2  # 180° - flip top to bottom
                            elif edge_name == 'right':
                                best_rotation = 1  # 90° clockwise - right becomes bottom
                            elif edge_name == 'left':
                                best_rotation = 3  # 90° counterclockwise - left becomes bottom
                        else:  # 4x6 orientation
                            # For 4x6 orientation, the 6-swatch edges are on left/right
                            # We need to rotate to make it a horizontal bottom edge
                            if edge_name == 'right':
                                best_rotation = 1  # 90° clockwise - right becomes bottom
                            elif edge_name == 'left':
                                best_rotation = 3  # 90° counterclockwise - left becomes bottom
                            # Note: bottom/top edges in 4x6 only have 4 swatches, so they won't be selected

        
        return best_rotation

    @staticmethod
    def perform_flatten_transform(main_window):
        """
        Executes perspective warp and grid overlay after four corner points are set.
        Now includes automatic rotation detection to ensure proper ColorChecker orientation.
        
        Args:
            main_window: Reference to the main application window
        """

        main_window.ui.imagePreviewGraphicsView.unsetCursor()

        PADDING = 600
        pts_src = np.array(main_window.corner_points, dtype=np.float32)
        
        # Calculate initial dimensions based on corner points
        side1_dist = np.linalg.norm(pts_src[1] - pts_src[0])  # top edge
        side2_dist = np.linalg.norm(pts_src[2] - pts_src[1])  # right edge
        
        # Determine the actual chart orientation from corner distances
        if side1_dist > side2_dist:
            # Landscape: 6x4 orientation - use standard ColorChecker proportions
            width = side1_dist
            height = int(width * 9.0 / 14.0)
        else:
            # Portrait: 4x6 orientation - swap the proportions
            height = side1_dist  # The "longer" side becomes height
            width = int(height * 14.0 / 9.0)  # Derive width from height
        
        dst_pts = np.array([
            [PADDING, PADDING], [PADDING+width, PADDING],
            [PADDING+width, PADDING+height], [PADDING, PADDING+height]
        ], dtype=np.float32)

        # Extract byte buffer from cropped_preview_pixmap
        qimg = main_window.cropped_preview_pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        h0 = qimg.height()
        stride = qimg.bytesPerLine()
        buf = bytes(qimg.constBits())[:h0*stride]
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h0, stride))
        arr = arr[:, :qimg.width()*3].reshape((h0, qimg.width(), 3))

        # Pad and compute warp
        arr_p = cv2.copyMakeBorder(arr, PADDING, PADDING, PADDING, PADDING,
                                  borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        pts_src_p = pts_src + PADDING
        M = cv2.getPerspectiveTransform(pts_src_p, dst_pts)
        warped = cv2.warpPerspective(arr_p, M, (int(width+2*PADDING), int(height+2*PADDING)))

        # Detect chart rotation using corner point distances to determine orientation
        rotation_needed = ChartTools.detect_chart_rotation_from_corners(main_window.corner_points, warped, width, height, PADDING)
        
        if rotation_needed > 0:
            main_window.log(f"[Flatten] Detected chart rotation, correcting by {rotation_needed * 90}°")
            
            # Apply rotation to warped image
            if rotation_needed == 1:  # 90° clockwise
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_needed == 2:  # 180°
                warped = cv2.rotate(warped, cv2.ROTATE_180)
            elif rotation_needed == 3:  # 270° clockwise (90° counterclockwise)
                warped = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            main_window.log("[Flatten] Chart orientation is correct")

        # Display warped image
        q2 = QImage(warped.data, warped.shape[1], warped.shape[0], warped.shape[1]*3,
                    QImage.Format_RGB888)
        main_window._display_preview(QPixmap.fromImage(q2))

        # Draw swatch grid - adjust dimensions if we rotated 90° or 270°
        if rotation_needed == 1 or rotation_needed == 3:
            # 90° or 270° rotation swaps width and height
            swatch_rect = QRect(PADDING, PADDING, int(height), int(width))
        else:
            # 0° or 180° rotation keeps original dimensions
            swatch_rect = QRect(PADDING, PADDING, int(width), int(height))
            
        main_window.flatten_swatch_rects = ChartTools.draw_colorchecker_swatch_grid(
            main_window.previewScene, swatch_rect, n_cols=6, n_rows=4)

        # Transform float-precision data with same rotation
        fparr = cv2.copyMakeBorder(main_window.cropped_fp, PADDING, PADDING, PADDING, PADDING,
                                   borderType=cv2.BORDER_REFLECT)
        cropped_fp_warped = cv2.warpPerspective(fparr.astype(np.float32), M,
                                              (warped.shape[1], warped.shape[0]))
        
        # Apply same rotation to float-precision data
        if rotation_needed > 0:
            if rotation_needed == 1:
                cropped_fp_warped = cv2.rotate(cropped_fp_warped, cv2.ROTATE_90_CLOCKWISE)
            elif rotation_needed == 2:
                cropped_fp_warped = cv2.rotate(cropped_fp_warped, cv2.ROTATE_180)
            elif rotation_needed == 3:
                cropped_fp_warped = cv2.rotate(cropped_fp_warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        main_window.cropped_fp = cropped_fp_warped
        main_window.flatten_mode = False
        main_window.ui.chartInformationLabel.setText(
            "Please Run Detect Chart or Revert image to select new region")
        main_window.ui.finalizeChartPushbutton.setEnabled(True)
        
        # Call preview with error handling to prevent hanging
        try:
            main_window.preview_manual_swatch_correction()
        except Exception as e:
            main_window.log(f"[Flatten] Preview error: {e}")
    
    @staticmethod
    def draw_colorchecker_swatch_grid(scene, image_rect, n_cols=6, n_rows=4, color=QColor(0, 255, 0, 90)):
        """
        Draws a 6x4 swatch grid (for ColorChecker) over the image_rect on the provided QGraphicsScene.
        
        Args:
            scene: QGraphicsScene to draw on
            image_rect: QRect defining the image area
            n_cols: Number of columns (default 6 for ColorChecker)
            n_rows: Number of rows (default 4 for ColorChecker)
            color: Color for the swatch overlays
            
        Returns:
            list: List of QRect for each swatch, row-wise
        """
        x0, y0, w, h = image_rect.left(), image_rect.top(), image_rect.width(), image_rect.height()
        swatch_rects = []
        
        for row in range(n_rows):
            for col in range(n_cols):
                sw_x0 = x0 + col * w / n_cols
                sw_y0 = y0 + row * h / n_rows
                sw_x1 = x0 + (col + 1) * w / n_cols
                sw_y1 = y0 + (row + 1) * h / n_rows
                rect = QRect(int(sw_x0), int(sw_y0), int(sw_x1 - sw_x0), int(sw_y1 - sw_y0))
                swatch_rects.append(rect)
                
                item = QGraphicsRectItem(rect)
                item.setBrush(QColor(color))
                item.setPen(QColor(50, 150, 50, 220))
                scene.addItem(item)
                
                # Draw swatch index label (1-based, row-wise)
                label_idx = row * n_cols + col + 1
                label = QGraphicsTextItem(str(label_idx))
                label.setDefaultTextColor(QColor(0, 120, 0))
                label.setPos(rect.left() + 6, rect.top() + 4)
                scene.addItem(label)
                
        return swatch_rects  # list of QRect for each swatch, row-wise
    
    @staticmethod
    def get_current_chart_assignment(main_window):
        """
        Get the currently assigned chart for the selected image or its group.
        
        Args:
            main_window: Reference to the main application window
            
        Returns:
            tuple: (chart_path, source_type) where source_type is 'image' or 'group' or None if no chart found
        """
        selected_item = main_window.ui.imagesListWidget.currentItem()
        if not selected_item:
            return None, None, None
            
        selected_meta = selected_item.data(Qt.UserRole)

        group_name = selected_meta.get('group_name', 'All Images')

        # Skip group headers
        if selected_meta.get('is_group_header', False):
            return None, None, None

        # First check if this specific image has a chart assigned in userData
        if selected_meta.get('chart', False):
            return selected_meta['input_path'], 'image', group_name
            
        # If not, check if the group has a chart assigned
        for chart_path, chart_group in main_window.chart_groups:
            if chart_group == group_name:
                return chart_path, 'group', group_name
                
        return None, None, None
    
    @staticmethod 
    def copy_chart_assignment(main_window):
        """
        Copy the currently selected image's chart assignment to clipboard.
        
        Args:
            main_window: Reference to the main application window
        """
        chart_path, source_type, group_name = ChartTools.get_current_chart_assignment(main_window)
        
        if chart_path:
            main_window.copied_chart_assignment = [chart_path, group_name]
            source_desc = "image" if source_type == 'image' else "group"
            main_window.log(f"[Copy Chart] Copied chart assignment from {source_desc}: {os.path.basename(chart_path)}")
        else:
            main_window.log("[Copy Chart] No chart assignment found for selected image or its group")
            main_window.copied_chart_assignment = None
    
    @staticmethod
    def paste_chart_assignment(main_window):
        """
        Paste the copied chart assignment to the currently selected group.
        
        Args:
            main_window: Reference to the main application window
        """
        if not hasattr(main_window, 'copied_chart_assignment') or not main_window.copied_chart_assignment:
            main_window.log("[Paste Chart] No chart assignment copied")
            return
            
        selected_item = main_window.ui.imagesListWidget.currentItem()
        if not selected_item:
            main_window.log("[Paste Chart] No image selected")
            return
            
        selected_meta = selected_item.data(Qt.UserRole)
        
        # Skip group headers
        if selected_meta.get('is_group_header', False):
            main_window.log("[Paste Chart] Cannot paste chart to group header")
            return
            
        chart_path, group_name = main_window.copied_chart_assignment
        selected_group = selected_meta.get('group_name', 'All Images')
        
        # Remove any existing chart assignment for this group
        existing_chart = None
        for chart_info in main_window.chart_groups:
            if chart_info[1] == selected_group:  # Same group
                existing_chart = chart_info[0]
                break
        
        # Clear the existing chart flag from the previous chart in this group
        if existing_chart:
            for i in range(main_window.ui.imagesListWidget.count()):
                item = main_window.ui.imagesListWidget.item(i)
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
            main_window.chart_groups = [[c, g] for c, g in main_window.chart_groups if not (c == existing_chart and g == selected_group)]

        selected_meta['chart'] = True
        selected_item.setData(Qt.UserRole, selected_meta)
        selected_item.setBackground(QColor('#ADD8E6'))

        group_data = main_window.group_calibrations[group_name]
        main_window.group_calibrations[selected_group] = group_data

        main_window.chart_groups.append([chart_path, selected_group])
        
        # Update UI to show group context
        main_window.ui.chartPathLineEdit.setText(f"[{selected_group}] {os.path.basename(chart_path)}")
        main_window.log(f"[Paste Chart] Chart pasted to group '{selected_group}': {os.path.basename(chart_path)}")
        
        # Store group information for calibration
        main_window.current_chart_group = selected_group
    
    @staticmethod
    def use_selected_chart_for_all_groups(main_window):
        """
        Take the current chart assignment and assign it to all other groups.
        
        Args:
            main_window: Reference to the main application window
        """
        chart_path, source_type = ChartTools.get_current_chart_assignment(main_window)
        
        if not chart_path:
            main_window.log("[Use Chart For All] No chart assignment found for selected image or its group")
            return
            
        selected_item = main_window.ui.imagesListWidget.currentItem()
        if not selected_item:
            return
            
        current_group = selected_item.data(Qt.UserRole).get('group_name', 'All Images')
        
        # Get all unique groups from the image list
        all_groups = set()
        for i in range(main_window.ui.imagesListWidget.count()):
            item = main_window.ui.imagesListWidget.item(i)
            meta = item.data(Qt.UserRole)
            if not meta.get('is_group_header', False):
                group_name = meta.get('group_name', 'All Images')
                all_groups.add(group_name)
        
        # Remove chart assignments from all groups first
        for group_name in all_groups:
            if group_name == current_group:
                continue  # Skip current group
                
            # Find existing chart for this group and remove it
            existing_chart = None
            for chart_info in main_window.chart_groups:
                if chart_info[1] == group_name:
                    existing_chart = chart_info[0]
                    break
            
            if existing_chart:
                # Clear chart flags from items in this group
                for i in range(main_window.ui.imagesListWidget.count()):
                    item = main_window.ui.imagesListWidget.item(i)
                    meta = item.data(Qt.UserRole)
                    
                    if (meta.get('is_group_header', False) or 
                        meta.get('group_name', 'All Images') != group_name):
                        continue
                    
                    # Clear chart flag if this was the chart image
                    if meta.get('input_path') == existing_chart:
                        meta['chart'] = False
                        meta.pop('debug_images', None)
                        item.setData(Qt.UserRole, meta)
                        item.setBackground(QColor('white'))
                
                # Remove from chart_groups list
                main_window.chart_groups = [[c, g] for c, g in main_window.chart_groups if not (c == existing_chart and g == group_name)]
        
        # Now add the current chart to all other groups
        groups_updated = 0
        for group_name in all_groups:
            if group_name == current_group:
                continue  # Skip current group
                
            main_window.chart_groups.append([chart_path, group_name])
            groups_updated += 1
        
        main_window.log(f"[Use Chart For All] Applied chart '{os.path.basename(chart_path)}' to {groups_updated} other groups")
    
    @staticmethod
    def remove_selected_chart_assignment(main_window):
        """
        Remove chart assignment from the selected image and its group.
        
        Args:
            main_window: Reference to the main application window
        """
        selected_item = main_window.ui.imagesListWidget.currentItem()
        if not selected_item:
            main_window.log("[Remove Chart] No image selected")
            return
            
        selected_meta = selected_item.data(Qt.UserRole)
        
        # Skip group headers
        if selected_meta.get('is_group_header', False):
            main_window.log("[Remove Chart] Cannot remove chart from group header")
            return
            
        group_name = selected_meta.get('group_name', 'All Images')
        image_path = selected_meta['input_path']
        
        # Clear chart assignment from the selected image
        if selected_meta.get('chart', False):
            selected_meta['chart'] = False
            selected_meta.pop('debug_images', None)
            selected_item.setData(Qt.UserRole, selected_meta)
            selected_item.setBackground(QColor('white'))
            main_window.log(f"[Remove Chart] Removed chart flag from image: {os.path.basename(image_path)}")
        
        # Find and remove group chart assignment
        existing_chart = None
        for chart_info in main_window.chart_groups:
            if chart_info[1] == group_name:
                existing_chart = chart_info[0]
                break
        
        if existing_chart:
            # Remove from chart_groups list
            main_window.chart_groups = [[c, g] for c, g in main_window.chart_groups if not (c == existing_chart and g == group_name)]
            main_window.log(f"[Remove Chart] Removed group chart assignment for '{group_name}': {os.path.basename(existing_chart)}")
            
            # Clear chart path display if it was showing this group's chart
            current_chart_text = main_window.ui.chartPathLineEdit.text()
            if current_chart_text.startswith(f'[{group_name}]'):
                main_window.ui.chartPathLineEdit.clear()
        
        # Clear current chart group if it matches
        if hasattr(main_window, 'current_chart_group') and main_window.current_chart_group == group_name:
            main_window.current_chart_group = None