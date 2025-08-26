"""
Image Loader Module

This module contains all functions and classes related to loading images from various formats.
Extracted from the main image_space.py file for better code organization.

Classes and functions include:
- RAW image loading with background workers
- Thumbnail loading and caching utilities  
- Support for various RAW formats (NEF, CR2, CR3, DNG, ARW)
- PIL and rawpy fallback mechanisms
- Color correction integration for thumbnails
"""

import os
import io
import numpy as np
import rawpy
import cv2
from PIL import Image
from PySide6.QtCore import QObject, Signal, QRunnable, QEventLoop
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QGraphicsPixmapItem, QApplication
import colour
from colour import RGB_COLOURSPACES

# Supported file formats
RAW_EXTENSIONS = ('.nef', '.cr2', '.cr3', '.dng', '.arw', '.raw')
OUTPUT_FORMATS = ('.jpg', '.png', '.tiff', '.exr')

# Global reference swatches for color correction
reference = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
illuminant_XYZ = colour.xy_to_XYZ(reference.illuminant)

reference_swatches = colour.XYZ_to_RGB(
    colour.xyY_to_XYZ(list(reference.data.values())),
    RGB_COLOURSPACES['sRGB'],
    illuminant=reference.illuminant,           # a 2-tuple (x, y)
    chromatic_adaptation_transform="CAT02",
    apply_cctf_encoding=False,
)

def _initialize_qt_limits():
    """
    Initialize Qt limits to handle large images.
    Increases allocation limit from default 128MB to 512MB.
    """
    try:
        # Set the Qt image allocation limit environment variable
        import os
        os.environ['QT_IMAGEIO_MAXALLOC'] = '512'  # 512MB limit
        
        # Also try to set via QImageReader if available
        from PySide6.QtGui import QImageReader
        QImageReader.setAllocationLimit(512)  # 512MB limit
        return True
    except Exception:
        pass
    return False

# Initialize Qt limits on module import
_initialize_qt_limits()


class RawLoadSignals(QObject):
    """Qt signals for RawLoadWorker to communicate with main thread."""
    loaded = Signal(np.ndarray)  # Emits loaded float32 image array
    error = Signal(str)          # Emits error message on failure


class RawLoadWorker(QRunnable):
    """
    Loads a RAW file in a background thread and emits only the full-precision
    float32 image array.
    
    This worker runs in a QThreadPool and communicates back to the main thread
    via Qt signals when the RAW file is loaded or if an error occurs.
    """
    
    def __init__(self, path: str):
        """
        Initialize the RAW loader worker.
        
        Args:
            path: Absolute path to the RAW image file to load
        """
        super().__init__()
        self.path = path
        self.signals = RawLoadSignals()

    def run(self):
        """Execute the RAW loading in background thread."""
        try:
            with rawpy.imread(self.path) as raw:
                common = dict(
                    gamma=(1, 1),  # Linear output
                    no_auto_bright=True,
                    use_camera_wb=True,
                    output_color=rawpy.ColorSpace.sRGB
                )
                # Only full-precision pipeline
                rgb_full = raw.postprocess(output_bps=16, **common)
                
                # Validate the RGB array before processing
                if rgb_full is None or rgb_full.size == 0:
                    raise ValueError("RAW postprocessing returned empty array")
                
                if len(rgb_full.shape) != 3 or rgb_full.shape[2] != 3:
                    raise ValueError(f"Invalid RGB array shape: {rgb_full.shape}, expected (H, W, 3)")
                
                # Convert to float32 first, normalize, but keep as float32 for OpenCV compatibility
                full_fp = (rgb_full.astype(np.float32) / 65535.0)
                
                # Scale the image to 1mpx for faster preview processing
                height, width = full_fp.shape[:2]
                current_pixels = height * width
                target_pixels = 1_000_000  # 1 megapixel
                
                if current_pixels > target_pixels:
                    # Validate dimensions before resize
                    if height <= 0 or width <= 0:
                        raise ValueError(f"Invalid image dimensions: {width}x{height}")
                    
                    # Calculate scale factor to achieve 1mpx
                    scale_factor = np.sqrt(target_pixels / current_pixels)
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    
                    # Ensure minimum dimensions
                    new_width = max(1, new_width)
                    new_height = max(1, new_height)
                    
                    # Validate the array is contiguous and has valid data
                    if not full_fp.flags.c_contiguous:
                        full_fp = np.ascontiguousarray(full_fp)
                    
                    if np.any(np.isnan(full_fp)) or np.any(np.isinf(full_fp)):
                        raise ValueError("Array contains NaN or infinite values")
                    
                    # Use cv2.INTER_AREA for downscaling (best quality for size reduction)
                    full_fp = cv2.resize(full_fp, (new_width, new_height), interpolation=cv2.INTER_AREA)

            # Emit the float32 array
            self.signals.loaded.emit(full_fp)

        except Exception as e:
            self.signals.error.emit(str(e))


class ImageLoader:
    """
    Static utility class for loading images from various formats.
    
    Provides methods for:
    - Loading RAW images synchronously using background workers
    - Loading thumbnails with caching and fallback mechanisms
    - Creating pixmaps for Qt display
    - Color correction integration
    """
    
    @staticmethod
    def load_raw_image_sync(path: str, threadpool, log_callback=None) -> np.ndarray | None:
        """
        Load a RAW file off the main thread but return synchronously.
        
        This method schedules a RawLoadWorker on the provided threadpool,
        spins a QEventLoop until the worker emits loaded(fp_array) or error(msg),
        and returns the float32 array in [0,1], or None on failure.
        
        Args:
            path: Path to the RAW image file
            threadpool: QThreadPool instance for background processing
            log_callback: Optional logging function that takes a string message
            
        Returns:
            np.ndarray: Float32 image array in [0,1] range, or None on failure
        """
        loop = QEventLoop()
        result = {'fp': None}

        # 1) Create the worker
        worker = RawLoadWorker(path)

        # 2) Hook up signals to capture the result and quit the local loop
        def _on_loaded(fp):
            result['fp'] = fp
            loop.quit()

        def _on_error(msg):
            if log_callback:
                log_callback(f"[RAW Load Error] {msg}")
            loop.quit()

        worker.signals.loaded.connect(_on_loaded)
        worker.signals.error.connect(_on_error)

        # 3) Start the worker & wait
        threadpool.start(worker)
        loop.exec()  # <- yields to Qt event loop until quit()

        # 4) Return whatever we got
        return result['fp']
    
    @staticmethod
    def load_thumbnail_array(path, max_size=(512, 512), cache=None, chart_swatches=None, 
                           correct_thumbnails=False, log_callback=None):
        """
        Loads an image and creates a small thumbnail as a NumPy RGB array, for fast preview or stats.
        Supports fallback for RAW files using rawpy if PIL fails.
        
        Args:
            path: Path to the image file
            max_size: Maximum dimensions for thumbnail (width, height)
            cache: Optional cache dict to check/store results
            chart_swatches: Optional chart swatches for color correction
            correct_thumbnails: Whether to apply color correction to thumbnails
            log_callback: Optional logging function
            
        Returns:
            np.ndarray: RGB array (H, W, 3) float32, values in 0..1, or None on failure
        """
        # Check cache first
        if cache:
            cached = cache.get(path)
            if cached and 'array' in cached and cached['array'] is not None:
                return cached['array']

        ext = os.path.splitext(path)[1].lower()
        arr = None
        
        try:
            # Try PIL first for standard formats
            with Image.open(path) as img:
                img.thumbnail(max_size)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                
                # Apply color correction if requested
                if chart_swatches is not None and correct_thumbnails:
                    try:
                        corrected = colour.colour_correction(arr, chart_swatches, reference_swatches)
                        corrected = np.clip(corrected, 0, 1)
                        arr = np.uint8(255 * colour.cctf_encoding(corrected))
                        if log_callback:
                            log_callback("[Thumb] Applied colour correction to thumbnail.")
                    except Exception as e:
                        if log_callback:
                            log_callback(f"[Thumb] Colour correction on thumbnail failed: {e}")
                
                # Store in cache
                if cache:
                    cache[path] = cache.get(path, {})
                    cache[path]['array'] = arr
                    
                return arr
                
        except Exception as e:
            # RAW fallback for ARW, NEF, etc
            if ext in RAW_EXTENSIONS:
                try:
                    arr = ImageLoader._load_raw_thumbnail(path, max_size, log_callback)
                    if arr is not None and cache:
                        cache[path] = cache.get(path, {})
                        cache[path]['array'] = arr
                    return arr
                except Exception as e2:
                    if log_callback:
                        log_callback(f"[Thumb] RAW fallback failed: {e2}")
            
            if log_callback:
                log_callback(f"[Thumb] Could not load thumbnail: {e}")
        
        return None
    
    @staticmethod  
    def _load_raw_thumbnail(path, max_size, log_callback=None):
        """
        Load thumbnail from RAW file using rawpy.
        
        Args:
            path: Path to RAW file
            max_size: Maximum thumbnail size (width, height)
            log_callback: Optional logging function
            
        Returns:
            np.ndarray: RGB array or None on failure
        """
        with rawpy.imread(path) as raw:
            try:
                # Try to extract embedded thumbnail first
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    pil = Image.open(io.BytesIO(thumb.data))
                    pil.thumbnail(max_size)
                    arr = np.asarray(pil, dtype=np.float32) / 255.0
                    if log_callback:
                        log_callback("[Thumb] Loaded embedded JPEG thumbnail from RAW.")
                    return arr
                else:
                    if log_callback:
                        log_callback("[Thumb] No embedded JPEG, using raw postprocess.")
            except Exception:
                if log_callback:
                    log_callback("[Thumb] No embedded thumbnail, using raw postprocess.")

            # Fallback to full RAW postprocessing
            rgb = raw.postprocess(output_bps=8, no_auto_bright=True)
            arr = rgb.astype(np.float32) / 255.0
            
            # Downsample with cv2 if needed
            h, w, _ = arr.shape
            scale = min(max_size[0] / h, max_size[1] / w, 1.0)
            if scale < 1.0:
                arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
            if log_callback:
                log_callback("[Thumb] Loaded thumbnail from rawpy postprocess.")
            return arr
    
    @staticmethod
    def create_pixmap_from_path(path, max_size=(512, 512), cache=None, chart_swatches=None,
                               correct_thumbnails=False, log_callback=None):
        """
        Create a QPixmap from an image file path, with caching and RAW support.
        
        Args:
            path: Path to image file
            max_size: Maximum size for loading
            cache: Optional cache dict
            chart_swatches: Optional swatches for color correction
            correct_thumbnails: Whether to apply color correction
            log_callback: Optional logging function
            
        Returns:
            QPixmap: Loaded pixmap, or None on failure
        """
        if log_callback:
            log_callback(f"[ImageLoader] Loading pixmap from: {path}")
            log_callback(f"[ImageLoader] File exists: {os.path.exists(path)}")
            log_callback(f"[ImageLoader] Max size: {max_size}, Cache available: {cache is not None}")
        
        # Try cache first
        if cache:
            cached = cache.get(path, {})
            pixmap = cached.get('pixmap')
            if pixmap and not pixmap.isNull():
                if log_callback:
                    log_callback(f"[ImageLoader] ✅ Found in cache: {pixmap.width()}x{pixmap.height()}")
                return pixmap
            elif log_callback:
                log_callback(f"[ImageLoader] Not in cache or cache invalid")

        ext = os.path.splitext(path)[1].lower()
        if log_callback:
            log_callback(f"[ImageLoader] File extension: {ext}")
        pixmap = None
        
        # Handle standard image formats with Qt
        if ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'):
            if log_callback:
                log_callback(f"[ImageLoader] Loading standard image format: {ext}")
            
            # Normalize path - use os.path.normpath for proper platform handling
            normalized_path = os.path.normpath(path)
            
            if log_callback:
                log_callback(f"[ImageLoader] Original path: {path}")
                log_callback(f"[ImageLoader] Normalized path: {normalized_path}")
            
            # Double-check file exists with normalized path
            if not os.path.exists(normalized_path):
                if log_callback:
                    log_callback(f"[ImageLoader] ❌ File does not exist at normalized path: {normalized_path}")
                pixmap = None
            else:
                # Try loading with automatic downsampling for large images
                try:
                    # First, check image size with PIL to determine if downsampling is needed
                    with Image.open(normalized_path) as pil_img:
                        width, height = pil_img.size
                        total_pixels = width * height
                        target_pixels = 2_000_000  # 2 megapixels target for display
                        
                        # Check if image needs downsampling
                        if total_pixels > target_pixels:
                            if log_callback:
                                log_callback(f"[ImageLoader] Large image detected: {width}x{height} ({total_pixels/1e6:.1f}MP), downsampling to 2MP")
                            
                            # Calculate scale factor to achieve 2MP
                            scale_factor = np.sqrt(target_pixels / total_pixels)
                            new_width = int(width * scale_factor)
                            new_height = int(height * scale_factor)
                            
                            # Downsample using PIL's high-quality resampling
                            pil_img.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
                            
                            if log_callback:
                                log_callback(f"[ImageLoader] Downsampled to: {pil_img.width}x{pil_img.height}")
                        
                        # Convert PIL image to QPixmap
                        if pil_img.mode == 'RGB':
                            qformat = QImage.Format_RGB888
                        elif pil_img.mode == 'RGBA':
                            qformat = QImage.Format_RGBA8888
                        else:
                            # Convert to RGB if other format
                            pil_img = pil_img.convert('RGB')
                            qformat = QImage.Format_RGB888
                        
                        # Create QImage from PIL data
                        img_data = pil_img.tobytes()
                        qimg = QImage(img_data, pil_img.width, pil_img.height, pil_img.width * len(pil_img.mode), qformat)
                        pixmap = QPixmap.fromImage(qimg)
                        
                        if not pixmap.isNull():
                            if log_callback:
                                log_callback(f"[ImageLoader] ✅ Image loaded successfully: {pixmap.width()}x{pixmap.height()}")
                        else:
                            if log_callback:
                                log_callback(f"[ImageLoader] ❌ Failed to create QPixmap from image")
                            pixmap = None
                        
                except Exception as e:
                    if log_callback:
                        log_callback(f"[ImageLoader] ❌ Failed to load standard image: {normalized_path}, error: {str(e)}")
                    
                    # Try direct QPixmap load as last resort (may fail on very large images)
                    try:
                        pixmap = QPixmap(normalized_path)
                        if pixmap.isNull():
                            pixmap = None
                        elif log_callback:
                            log_callback(f"[ImageLoader] ✅ Loaded with QPixmap fallback: {pixmap.width()}x{pixmap.height()}")
                    except Exception as e2:
                        if log_callback:
                            log_callback(f"[ImageLoader] ❌ QPixmap fallback also failed: {str(e2)}")
                        pixmap = None

        # Handle RAW formats
        elif ext in RAW_EXTENSIONS:
            if log_callback:
                log_callback(f"[ImageLoader] Loading RAW format: {ext}")
            try:
                arr = ImageLoader.load_thumbnail_array(
                    path, max_size, cache, chart_swatches, correct_thumbnails, log_callback
                )
                if arr is not None:
                    pixmap = ImageLoader.array_to_pixmap(arr)
                    if log_callback:
                        log_callback(f"[ImageLoader] ✅ RAW image loaded: {pixmap.width()}x{pixmap.height()}")
                else:
                    if log_callback:
                        log_callback(f"[ImageLoader] ❌ RAW load returned None array")
            except Exception as e:
                if log_callback:
                    log_callback(f"[ImageLoader] ❌ RAW load failed: {e}")
                    import traceback
                    log_callback(f"[ImageLoader] RAW traceback: {traceback.format_exc()}")

        # Handle unsupported formats
        else:
            if log_callback:
                log_callback(f"[ImageLoader] ❌ Unsupported file format: {ext}")

        # Cache the result
        if pixmap and not pixmap.isNull() and cache:
            if path not in cache:
                cache[path] = {}
            cache[path]['pixmap'] = pixmap
            if log_callback:
                log_callback(f"[ImageLoader] ✅ Cached pixmap for {path}")
        elif log_callback:
            log_callback(f"[ImageLoader] ❌ Could not load or cache pixmap for {path}")

        if log_callback:
            log_callback(f"[ImageLoader] Returning pixmap: {pixmap is not None and not pixmap.isNull() if pixmap else False}")
        
        return pixmap
    
    @staticmethod
    def array_to_pixmap(arr):
        """
        Convert a numpy array to QPixmap.
        
        Args:
            arr: Numpy array (H, W, 3) with values 0-1 (float) or 0-255 (uint8)
            
        Returns:
            QPixmap: Converted pixmap
        """
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            arr_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        else:
            arr_uint8 = arr.astype(np.uint8)
            
        h, w, c = arr_uint8.shape
        img = QImage(arr_uint8.data, w, h, w * c, QImage.Format_RGB888)
        return QPixmap.fromImage(img)
    
    @staticmethod
    def adjust_pixmap_brightness(pixmap, factor):
        """
        Adjust the brightness of a QPixmap by a given factor.
        
        Args:
            pixmap: Input QPixmap
            factor: Brightness multiplier (1.0 = no change)
            
        Returns:
            QPixmap: Brightness-adjusted pixmap
        """
        if factor == 1.0:
            return pixmap
            
        # Convert to image, adjust, convert back
        img = pixmap.toImage()
        if img.isNull():
            return pixmap
            
        # Simple brightness adjustment - could be enhanced
        # This is a placeholder for the actual brightness adjustment logic
        # that was in _adjust_pixmap_brightness in the original code
        return pixmap  # TODO: Implement brightness adjustment


class ThumbnailManager:
    """
    Manager class for handling thumbnail operations and caching.
    
    Provides higher-level operations like preview display, cache management,
    and integration with Qt graphics scenes.
    """
    
    def __init__(self, cache_dict=None, log_callback=None):
        """
        Initialize thumbnail manager.
        
        Args:
            cache_dict: Optional existing cache dictionary
            log_callback: Optional logging callback function
        """
        self.cache = cache_dict if cache_dict is not None else {}
        self.log_callback = log_callback
    
    def preview_thumbnail(self, path, graphics_scene, graphics_view, chart_swatches=None, 
                         correct_thumbnails=False, average_exposure_map=None):
        """
        Load and display a thumbnail in a Qt graphics scene.
        
        Args:
            path: Path to image file
            graphics_scene: QGraphicsScene to display in
            graphics_view: QGraphicsView for scene operations
            chart_swatches: Optional color correction swatches
            correct_thumbnails: Whether to apply color correction
            average_exposure_map: Optional dict mapping paths to exposure factors
            
        Returns:
            bool: True if successfully displayed, False otherwise
        """
        pixmap = ImageLoader.create_pixmap_from_path(
            path, cache=self.cache, chart_swatches=chart_swatches,
            correct_thumbnails=correct_thumbnails, log_callback=self.log_callback
        )
        
        if not pixmap or pixmap.isNull():
            return False
        
        # Apply exposure brightness if available
        if average_exposure_map and path in average_exposure_map:
            factor = average_exposure_map[path].get('average_exposure', 1.0)
            if factor != 1.0:
                pixmap = ImageLoader.adjust_pixmap_brightness(pixmap, factor)
        
        # Show in preview scene
        graphics_scene.clear()
        graphics_scene.addItem(QGraphicsPixmapItem(pixmap))
        graphics_view.resetTransform()
        graphics_view.fitInView(graphics_scene.itemsBoundingRect())
        
        return True
    
    def clear_cache(self):
        """Clear all cached thumbnails and arrays."""
        self.cache.clear()
    
    def get_cache_size(self):
        """Get number of cached items."""
        return len(self.cache)
    
    def remove_from_cache(self, path):
        """Remove specific path from cache."""
        if path in self.cache:
            del self.cache[path]