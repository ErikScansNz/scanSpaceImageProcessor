import gc
import os
import time
import threading
from pathlib import Path
import platform

import cv2
import numpy as np
import rawpy
import imageio
import colour
import tempfile

# Windows-specific imports
if platform.system() == 'Windows':
    import win32wnet
from OpenImageIO import ImageBuf, ImageBufAlgo, ImageSpec, ImageOutput, TypeFloat, ColorConfig, ROI, ImageInput, \
    TypeDesc, TypeUInt8
from PySide6.QtCore import (
    QRunnable, Signal, QObject
)
from colour import RGB_COLOURSPACES
from contextlib import contextmanager
from tifffile import tifffile
from colour_checker_detection import (
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC,
    detect_colour_checkers_segmentation)

# Import file naming schema
from ImageProcessor.fileNamingSchema import apply_naming_schema
# Import EXIF copy module
from ImageProcessor.copyExif import ExifCopyManager
# Import image editing tools
from ImageProcessor.editingTools import apply_all_adjustments

# supported file formats
RAW_EXTENSIONS = ('.nef', '.cr2', '.cr3', '.dng', '.arw', '.raw')
OUTPUT_FORMATS = ('.jpg', '.png', '.tiff', '.exr')


def wait_for_file_ready(filepath: str, max_wait: float = 5.0, check_interval: float = 0.1) -> bool:
    """
    Wait for file to be fully written and available for reading.
    
    Args:
        filepath: Path to check
        max_wait: Maximum time to wait in seconds
        check_interval: Time between checks in seconds
    
    Returns:
        bool: True if file is ready, False if timeout
    """
    start_time = time.time()
    
    while (time.time() - start_time) < max_wait:
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                time.sleep(check_interval)
                continue
            
            # Check if file has non-zero size
            if os.path.getsize(filepath) == 0:
                time.sleep(check_interval)
                continue
            
            # Try to open for reading to ensure it's not locked
            with open(filepath, 'rb') as f:
                f.read(1)  # Try to read first byte
            
            # Additional check for OIIO compatibility
            return True
            
        except (OSError, IOError, PermissionError):
            time.sleep(check_interval)
            continue
    
    return False


def safe_oiio_open_with_retry(filepath: str, max_retries: int = 3, wait_time: float = 2.0):
    """
    Safely open file with OIIO ImageInput with retry logic.
    
    Args:
        filepath: File to open
        max_retries: Maximum retry attempts
        wait_time: Time to wait for file to be ready
    
    Returns:
        ImageInput object or None
    """
    for attempt in range(max_retries):
        try:
            # First wait for file to be ready
            if not wait_for_file_ready(filepath, max_wait=wait_time):
                print(f"[OIIO] File not ready after {wait_time}s: {filepath}")
                continue
            
            # Try to open with OIIO
            img_input = ImageInput.open(filepath)
            if img_input:
                return img_input
            else:
                print(f"[OIIO] ImageInput.open returned None for {filepath} (attempt {attempt + 1})")
                
        except Exception as e:
            print(f"[OIIO] Error opening {filepath} (attempt {attempt + 1}): {e}")
        
        if attempt < max_retries - 1:
            time.sleep(0.5 * (attempt + 1))  # Exponential backoff
    
    return None

reference = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
illuminant_XYZ = colour.xy_to_XYZ(reference.illuminant)

reference_swatches = colour.XYZ_to_RGB(
    colour.xyY_to_XYZ(list(reference.data.values())),
    RGB_COLOURSPACES['sRGB'],
    illuminant=reference.illuminant,           # a 2-tuple (x, y)
    chromatic_adaptation_transform="CAT02",
    apply_cctf_encoding=False,
)


def simple_unc_to_drive_conversion(unc_path: str) -> str:
    """
    Simple conversion of UNC path to drive letter if a mapping exists.
    Only converts if we can find an exact drive mapping, otherwise returns original.
    """
    if not unc_path or not unc_path.startswith('\\\\'):
        return unc_path
    
    # Only attempt conversion on Windows
    if platform.system() != 'Windows':
        return unc_path
    
    try:
        import win32api
        drives = win32api.GetLogicalDriveStrings().split('\000')[:-1]
        
        for drive in drives:
            try:
                # Get the UNC path for this drive
                drive_unc = win32wnet.WNetGetUniversalName(drive.rstrip('\\'))
                
                # Check if our UNC path starts with this drive's UNC path
                if unc_path.startswith(drive_unc):
                    # Replace UNC portion with drive letter
                    relative_part = unc_path[len(drive_unc):]
                    mapped_path = drive.rstrip('\\') + relative_part
                    
                    # Only return if the mapped path actually exists
                    if os.path.exists(mapped_path):
                        return mapped_path
                        
            except Exception:
                continue  # Skip this drive and try the next one
                
    except Exception:
        pass
    
    # If no mapping found, return original path
    return unc_path


class RawLoadSignals(QObject):
    loaded = Signal(np.ndarray)
    error  = Signal(str)

class RawLoadWorker(QRunnable):
    """
    Loads a RAW file in a background thread and emits only the full-precision
    float32 image array.
    """
    def __init__(self, path: str):
        super().__init__()
        self.path    = path
        self.signals = RawLoadSignals()

    def run(self):
        try:
            with rawpy.imread(self.path) as raw:
                common = dict(
                    gamma=(1,1),
                    no_auto_bright=True,
                    use_camera_wb=True,
                    output_color=rawpy.ColorSpace.sRGB
                )
                # Only full-precision pipeline
                rgb_full = raw.postprocess(output_bps=16, **common)
                full_fp  = np.array(rgb_full, dtype=np.float32) / 65535.0

            # Emit just the float32 array
            self.signals.loaded.emit(full_fp)

        except Exception as e:
            self.signals.error.emit(str(e))


class ImageCorrectionWorker(QRunnable):
    def __init__(self, images, swatches, output_folder, signals, jpeg_quality=100,
                 rename_map=None, name_base='', padding=0, export_masked=False,
                 output_format: str = '.jpg', tiff_bitdepth=8, exr_colorspace: str | None = None,
                 export_schema: str = "", use_export_schema: bool = True,
                 custom_name: str = "", root_folder: str = "", from_network=False,
                 network_output_path: str = "", group_name: str = "",
                 use_chart: bool = True, exposure_adj: float = 0.0,
                 shadow_adj: float = 0.0, highlight_adj: float = 0.0, 
                 white_balance_adj: int = 5500, denoise_strength: float = 0.0,
                 sharpen_amount: float = 0.0, downsample_percentage: float = 100.0,
                 use_white_balance: bool = False):
        super().__init__()
        self.images = images
        self.swatches = swatches
        self.output_folder = output_folder
        self.signals = signals
        self.jpeg_quality = jpeg_quality
        self.output_format = output_format
        self.tiff_bitdepth = tiff_bitdepth
        self.exr_colorspace = exr_colorspace
        # File renaming support
        self.rename_map = rename_map or {}
        self.name_base = name_base
        self.padding = padding
        self.export_masked = export_masked
        self.cancelled = False
        # Export schema support
        self.export_schema = export_schema
        self.use_export_schema = use_export_schema
        self.custom_name = custom_name
        self.root_folder = root_folder
        self.from_network = from_network
        self.network_output_path = network_output_path
        self.group_name = group_name
        
        # Image adjustment parameters
        self.use_chart = use_chart
        self.exposure_adj = exposure_adj
        self.shadow_adj = shadow_adj
        self.highlight_adj = highlight_adj
        self.white_balance_adj = white_balance_adj
        self.use_white_balance = use_white_balance
        self.denoise_strength = denoise_strength
        self.sharpen_amount = sharpen_amount
        self.downsample_percentage = downsample_percentage

        # EXIF copy manager for thread-safe metadata operations
        self.exif_manager = ExifCopyManager()
        self.exif_manager.log_message.connect(self.signals.log.emit)

    def cancel(self):
        self.cancelled = True

    def _create_fallback_spec(self, corrected, corrected_uint8, ext):
        """
        Create a fallback ImageSpec when we can't read the output file spec.

        Args:
            corrected: The corrected float image array
            corrected_uint8: The corrected uint8 image array
            ext: The output file extension

        Returns:
            ImageSpec: Fallback spec for the image
        """
        try:
            if ext == '.exr' or ext == '.tiff':
                h, w, c = corrected.shape
                return ImageSpec(w, h, c, TypeFloat)
            else:
                h, w = corrected_uint8.shape[:2]
                channels = corrected_uint8.shape[2] if corrected_uint8.ndim == 3 else 1
                return ImageSpec(w, h, channels, TypeUInt8)
        except Exception as e:
            self.signals.log.emit(f"[Fallback Spec Error] {e}")
            # Absolute fallback - basic RGB spec
            return ImageSpec(100, 100, 3, TypeUInt8)

    def load_image(self, img_path: str) -> np.ndarray:
        """
        Load RAW image and return float32 array.
        
        Args:
            img_path: Path to the RAW image file
            
        Returns:
            np.ndarray: Float32 image array (0-1 range)
            
        Raises:
            Exception: If image loading fails
        """
        self.signals.log.emit(f"[RAW] Using original path for rawpy: {img_path}")
        with rawpy.imread(img_path) as raw:
            rgb = raw.postprocess(
                output_bps=16,
                gamma=(1, 1),
                no_auto_bright=True,
                use_camera_wb=True,
                output_color=rawpy.ColorSpace.sRGB
            )
            raw_img_float = np.array(rgb, dtype=np.float32) / 65535.0
            del rgb  # Delete rgb array after conversion
            gc.collect()
            
            # Apply downsampling if percentage is less than 100%
            if self.downsample_percentage < 100.0:
                raw_img_float = self._apply_downsampling(raw_img_float, self.downsample_percentage)
                
            return raw_img_float

    def apply_colour_correction(self, img_array: np.ndarray) -> np.ndarray:
        """
        Apply colour correction using chart swatches.
        
        Args:
            img_array: Input image array
            
        Returns:
            np.ndarray: Colour corrected image array
        """
        corrected = colour.colour_correction(img_array, self.swatches, reference_swatches)
        return corrected

    def apply_image_adjustments(self, img_array: np.ndarray, img_path: str) -> np.ndarray:
        """
        Apply exposure, shadow, highlight, white balance, denoise, and sharpen adjustments.
        Only called when use_chart is False.
        
        Args:
            img_array: Input image array
            img_path: Path to image (for metadata lookup)
            
        Returns:
            np.ndarray: Adjusted image array
        """
        # Apply all adjustments using editing tools
        adjusted = apply_all_adjustments(
            img_array,
            exposure=self.exposure_adj,
            shadows=self.shadow_adj,
            highlights=self.highlight_adj,
            use_wb=self.use_white_balance,
            target_wb=self.white_balance_adj,
            wb_tint=0.0,
            denoise_strength=self.denoise_strength,
            sharpen_amount=self.sharpen_amount,
            sharpen_radius=1.0,  # Default radius
            sharpen_threshold=0.0  # Default threshold
        )
        
        return adjusted
    
    def _has_non_default_adjustments(self) -> bool:
        """
        Check if any adjustment parameters differ from their defaults.
        
        Returns:
            bool: True if any adjustments are non-default
        """
        has_adjustments = False
        
        # Check each adjustment parameter
        if self.exposure_adj != 0.0:
            has_adjustments = True
        if self.shadow_adj != 0.0:
            has_adjustments = True
        if self.highlight_adj != 0.0:
            has_adjustments = True
        if self.white_balance_adj != 5500:  # Default daylight temperature
            has_adjustments = True
        if hasattr(self, 'denoise_strength') and self.denoise_strength != 0.0:
            has_adjustments = True
        if hasattr(self, 'sharpen_amount') and self.sharpen_amount != 0.0:
            has_adjustments = True
            
        return has_adjustments

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
                    self.signals.log.emit(f"[WB] Found white balance from rawpy: {wb_temp}K")
                    return wb_temp
            
            # Method 2: Try to get white balance from EXIF using OpenImageIO
            wb_temp = self._get_wb_from_oiio_exif(img_path)
            if wb_temp is not None:
                self.signals.log.emit(f"[WB] Found white balance from EXIF: {wb_temp}K")
                return wb_temp
                
        except Exception as e:
            self.signals.log.emit(f"[WB] Error reading white balance from {img_path}: {e}")
        
        self.signals.log.emit(f"[WB] Using default white balance: {default_wb}K")
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
                    # This is a simplified conversion - more accurate methods exist
                    r_mult, g_mult, b_mult = wb_coeffs[0], wb_coeffs[1], wb_coeffs[2]
                    
                    # Normalize to green
                    if g_mult > 0:
                        r_ratio = r_mult / g_mult
                        b_ratio = b_mult / g_mult
                        
                        # Estimate temperature from red/blue ratio
                        # Warmer light = more red, less blue (lower temperature)
                        # Cooler light = less red, more blue (higher temperature)
                        rb_ratio = r_ratio / (b_ratio + 1e-10)
                        
                        # Empirical formula for RGB multipliers to temperature conversion
                        # Based on typical camera white balance behavior
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
            self.signals.log.emit(f"[WB] Error reading rawpy white balance: {e}")
        
        return None

    def _get_wb_from_oiio_exif(self, img_path: str) -> float:
        """
        Extract white balance from EXIF metadata using ExifRead.
        
        Args:
            img_path: Path to image file
            
        Returns:
            float: White balance temperature in Kelvin, or None if not available
        """
        try:
            import exifread
            
            # Open file and read EXIF data
            with open(img_path, 'rb') as f:
                tags = exifread.process_file(f, details=False)
            
            if not tags:
                return None
            
            # Try various EXIF white balance fields
            wb_fields = [
                'EXIF ColorTemperature',
                'EXIF WhiteBalance',
                'Image ColorTemperature',
                'Image WhiteBalance',
                'MakerNote ColorTemperature',
                'MakerNote WhiteBalance',
                # Canon specific
                'MakerNote ColorTemp',
                'MakerNote ColorTempAsShot',
                # Nikon specific
                'MakerNote WB_RBLevels',
                'MakerNote ColorBalance',
            ]
            
            for field in wb_fields:
                if field in tags:
                    tag_value = tags[field]
                    # ExifRead returns IfdTag objects, get their values
                    value_str = str(tag_value.values[0] if hasattr(tag_value, 'values') else tag_value)
                    
                    try:
                        # Try to extract numeric value
                        # Remove any non-numeric characters except decimal point
                        import re
                        numeric_match = re.search(r'[\d.]+', value_str)
                        if numeric_match:
                            temp = float(numeric_match.group())
                            if 1000 <= temp <= 12000:  # Reasonable temperature range
                                return temp
                    except (ValueError, TypeError):
                        pass
            
            # Try to calculate from RGB multipliers if available
            # This is common in RAW files
            for tag_name in ['EXIF WhitePoint', 'Image WhitePoint', 'MakerNote WB_RGBLevels']:
                if tag_name in tags:
                    try:
                        values = tags[tag_name].values
                        if len(values) >= 2:
                            # Convert RGB multipliers to approximate color temperature
                            # This is a rough approximation
                            r_over_b = float(values[0]) / float(values[1]) if float(values[1]) != 0 else 1
                            # Approximate mapping from R/B ratio to temperature
                            if 0.5 <= r_over_b <= 2.0:
                                temp = 3000 + (r_over_b - 0.5) * 4000
                                if 1000 <= temp <= 12000:
                                    return temp
                    except (ValueError, TypeError, IndexError, AttributeError):
                        pass
                        
        except Exception as e:
            self.signals.log.emit(f"[WB] Error reading ExifRead white balance: {e}")
        
        return None

    def _estimate_wb_from_filename(self, img_path: str) -> float:
        """
        Estimate white balance from filename patterns or directory structure.
        
        Args:
            img_path: Path to image file
            
        Returns:
            float: Estimated white balance temperature in Kelvin
        """
        filename = os.path.basename(img_path).lower()
        
        # Common photography lighting conditions
        wb_keywords = {
            'tungsten': 3200,
            'incandescent': 3200,
            'halogen': 3200,
            'fluorescent': 4000,
            'daylight': 5500,
            'sun': 5500,
            'cloudy': 6500,
            'shade': 7500,
            'flash': 5500,
            'studio': 5500
        }
        
        # Check filename for lighting keywords
        for keyword, temp in wb_keywords.items():
            if keyword in filename:
                return float(temp)
        
        # Check parent directory names for lighting conditions
        try:
            parent_dir = os.path.basename(os.path.dirname(img_path)).lower()
            for keyword, temp in wb_keywords.items():
                if keyword in parent_dir:
                    return float(temp)
        except:
            pass
        
        # Default to daylight
        return 5500.0

    def encode_image(self, img_array: np.ndarray, output_format: str) -> np.ndarray:
        """
        Encode image array for specific output format.
        
        Args:
            img_array: Float image array (0-1 range)
            output_format: Target output format
            
        Returns:
            np.ndarray: Encoded image array (uint8 or uint16)
        """
        if output_format in ('.png', '.jpg', '.jpeg'):
            # Always 8-bit for these formats
            encoded_corrected = colour.cctf_encoding(img_array)
            return np.uint8(255 * encoded_corrected)
        elif output_format == '.tiff':
            encoded_corrected = colour.cctf_encoding(img_array)
            if self.tiff_bitdepth == 16:
                return np.uint16(np.clip(encoded_corrected * 65535.0, 0, 65535))
            else:
                return np.uint8(np.clip(encoded_corrected * 255.0, 0, 255))
        else:
            # For EXR, return the float array directly
            return img_array

    def export_image_jpg(self, img_array: np.ndarray, output_path: str):
        """Export image as JPEG format."""
        imageio.imwrite(output_path, img_array, quality=self.jpeg_quality)

    def export_image_png(self, img_array: np.ndarray, output_path: str):
        """Export image as PNG format."""
        imageio.imwrite(output_path, img_array)

    def export_image_tiff(self, img_array: np.ndarray, output_path: str):
        """Export image as TIFF format."""
        with tifffile.TiffWriter(output_path) as tiff_writer:
            tiff_writer.write(img_array)

    def export_image_exr(self, img_array: np.ndarray, output_path: str) -> np.ndarray:
        """
        Export image as EXR format with color space conversion.
        
        Returns:
            np.ndarray: uint8 version for preview
        """
        # Write a temp EXR tagged as sRGB
        h, w, c = img_array.shape
        fd, temp_exr = tempfile.mkstemp(suffix=".exr")
        os.close(fd)

        spec = ImageSpec(w, h, 3, TypeFloat)
        spec.attribute("oiio:ColorSpace", "sRGB")
        out = ImageOutput.create(temp_exr)
        out.open(temp_exr, spec)

        # Flatten and write, then immediately delete flattened array
        flattened_corrected = img_array.flatten()
        out.write_image(flattened_corrected)
        del flattened_corrected
        gc.collect()

        out.close()
        del out, spec  # Clean up OIIO objects

        # Load it via ImageBuf & colorconvert
        buf_in = ImageBuf(temp_exr)
        buf_out = ImageBufAlgo.colorconvert(buf_in, "sRGB", self.exr_colorspace)
        del buf_in  # Delete input buffer
        gc.collect()

        # Write the final EXR in the target space
        buf_out.write(output_path)

        # Create uint8 version for preview
        preview_uint8 = np.uint8(np.clip(img_array * 255.0, 0, 255))
        
        # Remove the temp file
        try:
            os.remove(temp_exr)
        except OSError:
            pass
        del temp_exr  # Clean up temp file path

        return buf_out, preview_uint8

    def construct_export_path(self, img_path: str) -> str:
        """
        Construct the output path using export schema or default naming.
        
        Args:
            img_path: Input image path
            
        Returns:
            str: Output file path
        """
        
        if self.from_network:
            # For network processing, use the pre-calculated output path from the server
            if self.network_output_path:
                out_path = self.network_output_path
                print(f"[Network Worker] Using server-provided output path: {out_path}")
                return out_path
            else:
                print(f"[Network Worker] WARNING: network_output_path is empty, falling back to local construction")
                # Fallback: construct output path from input filename
                if getattr(self, "use_original_filenames", False):
                    out_fn = os.path.splitext(os.path.basename(img_path))[0] + self.output_format
                else:
                    # Use rename_map if available, otherwise use original filename
                    seq = self.rename_map.get(img_path, 1) if self.rename_map else 1
                    if self.name_base:
                        out_fn = f"{self.name_base}_{seq:0{self.padding}d}{self.output_format}"
                    else:
                        out_fn = os.path.splitext(os.path.basename(img_path))[0] + self.output_format
                    del seq  # Clean up sequence variable
                out_path = os.path.join(self.output_folder, out_fn)
                del out_fn  # Clean up filename variable
                return out_path
        else:
            if self.use_export_schema and self.export_schema:
                try:
                    seq = self.rename_map.get(img_path, 1)
                    
                    # Ensure all parameters are strings, not None
                    custom_name = self.custom_name if self.custom_name is not None else ""
                    root_folder = self.root_folder if self.root_folder is not None else ""
                    group_name = self.group_name if self.group_name is not None else ""
                    
                    out_path = apply_naming_schema(
                        schema=self.export_schema,
                        input_path=img_path,
                        output_base_dir=self.output_folder,
                        custom_name=custom_name,
                        image_number=seq,
                        output_extension=self.output_format,
                        root_folder=root_folder,
                        group_name=group_name
                    )
                    del seq, custom_name, root_folder, group_name  # Clean up variables
                    # Ensure directory exists
                    out_dir = os.path.dirname(out_path)
                    if out_dir and out_dir != self.output_folder:
                        os.makedirs(out_dir, exist_ok=True)
                    del out_dir  # Clean up directory variable
                    return out_path
                except Exception as e:
                    self.signals.log.emit(f"[Schema Error] Failed to apply naming schema for {img_path}: {e}")
                    # Fallback to default naming
                    out_fn = os.path.splitext(os.path.basename(img_path))[0] + self.output_format
                    out_path = os.path.join(self.output_folder, out_fn)
                    del out_fn  # Clean up filename variable
                    return out_path
            else:
                # Use legacy naming logic
                out_fn = None
                if getattr(self, "use_original_filenames", False):
                    # Use original filename with extension
                    out_fn = os.path.splitext(os.path.basename(img_path))[0] + self.output_format
                else:
                    seq = self.rename_map.get(img_path)
                    if seq is not None and self.name_base:
                        out_fn = f"{self.name_base}_{seq:0{self.padding}d}{self.output_format}"
                    else:
                        out_fn = os.path.splitext(os.path.basename(img_path))[0] + self.output_format
                    del seq  # Clean up sequence variable
                out_path = os.path.join(self.output_folder, out_fn)
                del out_fn  # Clean up filename variable
                return out_path

    def embed_metadata(self, img_path: str, out_path: str, corrected: np.ndarray, 
                      corrected_uint8: np.ndarray, buf_out=None):
        """
        Embed metadata from source image into output image.
        
        Args:
            img_path: Source image path
            out_path: Output image path
            corrected: Float corrected image array
            corrected_uint8: uint8 corrected image array
            buf_out: EXR buffer for EXR files
        """
        ext = self.output_format
        
        # Read source metadata
        in_img = ImageInput.open(img_path)
        if in_img:
            src_pvs = in_img.spec().extra_attribs  # ParamValueList of all tags
            in_img.close()
            del in_img  # Clean up input image object
        else:
            src_pvs = []

        # Read output spec
        out_reader = ImageInput.open(out_path)
        if out_reader:
            out_spec = out_reader.spec()
            out_reader.close()
            del out_reader  # Clean up output reader
        else:
            out_spec = self._create_fallback_spec(corrected, corrected_uint8, ext)

        # Copy metadata to output spec
        for pv in src_pvs:
            name = pv.name
            val = pv.value
            if isinstance(val, (int, float, str)):
                out_spec.attribute(name, val)
            else:
                out_spec.attribute(name, str(val))
            del name, val  # Clean up loop variables

        del src_pvs  # Clean up source attributes
        gc.collect()

        # Write with metadata
        writer = ImageOutput.create(out_path)
        if not writer:
            self.signals.log.emit(f"[Metadata ERROR] Could not open writer for {out_path}", flush=True)
        else:
            writer.open(out_path, out_spec)
            if ext == '.exr' and buf_out is not None:
                pixels = buf_out.get_pixels(TypeFloat)
            else:
                if corrected_uint8 is not None:
                    pixels = corrected_uint8.flatten()
                else:
                    # Fallback: use corrected array converted to uint8
                    encoded_corrected = colour.cctf_encoding(corrected)
                    temp_uint8 = np.uint8(255 * encoded_corrected)
                    pixels = temp_uint8.flatten()
                    del encoded_corrected, temp_uint8
            writer.write_image(pixels)
            writer.close()
            del writer, pixels  # Clean up writer and pixels
            gc.collect()

    def run(self):
        """
        Main processing loop using modular component functions.
        """
        ext = self.output_format

        for img_path in self.images:
            local_timer_start = time.time()
            out_path = None  # Initialize out_path to avoid UnboundLocalError

            try:
                # 1. Construct export path
                out_path = self.construct_export_path(img_path)
                if out_path is None:
                    # Fallback if construct_export_path returns None
                    out_path = os.path.join(self.output_folder, os.path.splitext(os.path.basename(img_path))[0] + self.output_format)

                # 2. Send start signal
                if self.from_network:
                    self.signals.status.emit('started')
                else:
                    self.signals.status.emit(img_path, 'started', (time.time() - local_timer_start), out_path)
                self.signals.log.emit(f"[Worker] Starting {os.path.basename(img_path)} -> {os.path.basename(out_path)}")

                # 3. Load RAW image
                try:
                    raw_img_float = self.load_image(img_path)
                except Exception as e:
                    self.signals.log.emit(f"[Image Load Error] {img_path}: {e}")
                    self.signals.status.emit(img_path, 'error', time.time() - local_timer_start, out_path)
                    continue

                # 4. Check for cancellation
                if self.cancelled:
                    self.signals.log.emit("[Worker] Cancelled by user. Exiting thread.")
                    self.signals.status.emit(img_path, 'cancelled', time.time() - local_timer_start, out_path)
                    del raw_img_float
                    gc.collect()
                    return

                # 5. Apply processing pipeline based on use_chart setting
                if self.use_chart:
                    # Traditional chart-based color correction
                    corrected = self.apply_colour_correction(raw_img_float)
                    del raw_img_float
                    gc.collect()

                    # Apply average brightness correction if metadata available
                    multiplier = 1.0
                    if hasattr(self, "image_metadata_map"):
                        meta = self.image_metadata_map.get(img_path)
                        if meta is not None:
                            value = meta.get('average_exposure')
                            try:
                                multiplier = float(value)
                            except (TypeError, ValueError):
                                multiplier = 1.0
                            del value
                        del meta

                    corrected = corrected * multiplier
                    corrected = np.clip(corrected, 0, 1)
                    
                    # Apply image adjustments AFTER chart correction if they're not at defaults
                    if self._has_non_default_adjustments():
                        # Get current white balance from metadata
                        
                        # Apply only non-default adjustments
                        corrected = apply_all_adjustments(
                            corrected,
                            exposure=self.exposure_adj if self.exposure_adj != 0.0 else 0.0,
                            shadows=self.shadow_adj if self.shadow_adj != 0.0 else 0.0,
                            highlights=self.highlight_adj if self.highlight_adj != 0.0 else 0.0,
                            use_wb=self.use_white_balance,
                            target_wb=self.white_balance_adj,
                            wb_tint=0.0,
                            denoise_strength=self.denoise_strength if hasattr(self, 'denoise_strength') and self.denoise_strength != 0.0 else 0.0
                        )
                        corrected = np.clip(corrected, 0, 1)
                else:
                    # New adjustment-based processing (no chart)
                    corrected = self.apply_image_adjustments(raw_img_float, img_path)
                    del raw_img_float
                    gc.collect()
                    corrected = np.clip(corrected, 0, 1)

                # 6. Handle masking if enabled
                if getattr(self, "export_masked", False):
                    corrected = self._apply_masking(corrected)

                # 7. Encode image for output format
                if ext == '.exr':
                    # Special handling for EXR
                    buf_out, corrected_uint8 = self.export_image_exr(corrected, out_path)
                else:
                    # Standard encoding
                    encoded_img = self.encode_image(corrected, ext)
                    
                    # Export using format-specific function
                    if ext in ('.jpg', '.jpeg'):
                        self.export_image_jpg(encoded_img, out_path)
                    elif ext == '.png':
                        self.export_image_png(encoded_img, out_path)
                    elif ext == '.tiff':
                        self.export_image_tiff(encoded_img, out_path)
                    
                    corrected_uint8 = encoded_img if encoded_img.dtype == np.uint8 else np.uint8(np.clip(corrected * 255.0, 0, 255))
                    buf_out = None

                self.signals.log.emit(f"[Saved] {out_path}")

                # 8. Embed metadata
                self.embed_metadata(img_path, out_path, corrected, corrected_uint8, buf_out)

                # 9. Clean up EXR-specific variables
                if ext == '.exr' and buf_out is not None:
                    del buf_out

                # 10. Send preview and completion signals
                # Create 2000px max resolution proxy for UI preview to avoid massive memory transfers
                # preview_proxy = self._create_preview_proxy(corrected_uint8)
                data = [out_path]
                self.signals.preview.emit(data)
                del data

                self.signals.status.emit(img_path, 'finished', (time.time() - local_timer_start), out_path)

                # Console output for headless/network mode
                if self.from_network:
                    processing_time = time.time() - local_timer_start
                    del processing_time

                # 11. Clean up major arrays
                del corrected, corrected_uint8
                gc.collect()

            except Exception as e:
                self.signals.log.emit(f"[Processing Error] {img_path}: {e}")
                # Ensure out_path has a fallback value for error reporting
                if out_path is None:
                    out_path = os.path.join(self.output_folder, os.path.splitext(os.path.basename(img_path))[0] + self.output_format)
                self.signals.status.emit(img_path, 'error', time.time() - local_timer_start, out_path)

                # Console output for headless/network mode
                if self.from_network:
                    print(f"[Network Worker] ✗ Error: {os.path.basename(img_path)} - {e}")
                    import traceback
                    print(f"[Network Worker] Traceback: {traceback.format_exc()}")

                # Clean up any remaining variables in case of error
                for var_name in ['corrected', 'corrected_uint8', 'raw_img_float', 'encoded_img', 'buf_out']:
                    try:
                        if var_name in locals():
                            del locals()[var_name]
                    except:
                        pass
                gc.collect()

            # Clean up per-iteration variables
            try:
                del img_path, out_path, local_timer_start
            except:
                pass
            gc.collect()

    def _create_preview_proxy(self, corrected_uint8: np.ndarray, max_size: int = 2000) -> np.ndarray:
        """
        Create a downsized proxy image for UI preview to reduce memory transfer.
        
        Args:
            corrected_uint8: Full resolution uint8 image array
            max_size: Maximum dimension (width or height) for proxy
            
        Returns:
            np.ndarray: Downsized proxy image array
        """
        h, w = corrected_uint8.shape[:2]
        
        # Calculate scale factor to keep max dimension under max_size
        scale = min(max_size / w, max_size / h)
        
        # Only downsize if image is larger than max_size
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            # Use OpenCV for high-quality resizing
            proxy = cv2.resize(corrected_uint8, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return proxy
        else:
            # Image is already small enough, return copy to avoid modifying original
            return corrected_uint8.copy()

    def _apply_masking(self, corrected: np.ndarray) -> np.ndarray:
        """
        Apply masking to remove shadow and highlight regions.
        
        Args:
            corrected: Input image array
            
        Returns:
            np.ndarray: Masked image array
        """
        lum = 0.2126 * corrected[:, :, 0] + 0.7152 * corrected[:, :, 1] + 0.0722 * corrected[:, :, 2]
        shadow_limit = getattr(self, "shadow_limit", 0.05)
        highlight_limit = getattr(self, "highlight_limit", 0.98)
        shadow_mask = lum <= shadow_limit
        highlight_mask = lum >= highlight_limit
        del lum

        mask = shadow_mask | highlight_mask
        del shadow_mask, highlight_mask
        gc.collect()

        # Find connected clusters in mask
        mask_uint8 = mask.astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        del mask_uint8

        min_area = 200  # minimum area threshold (in pixels)
        clean_mask = np.zeros_like(mask, dtype=bool)
        del mask
        gc.collect()

        for label in range(1, num_labels):  # 0 is background
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= min_area:
                clean_mask[labels == label] = True

        # Clean up connected components data
        del labels, stats, centroids
        gc.collect()

        masked_pixels_count = np.sum(clean_mask)
        corrected[clean_mask] = 0
        del clean_mask
        gc.collect()

        self.signals.log.emit(
            f"[Export Masked] Masked {masked_pixels_count} pixels (min area={min_area}) "
            f"with limits shadow<={shadow_limit:.2f}, highlight>={highlight_limit:.2f}"
        )
        del masked_pixels_count

        return corrected
    
    def _apply_downsampling(self, img_array: np.ndarray, percentage: float) -> np.ndarray:
        """
        Apply downsampling to the image based on percentage.
        
        Args:
            img_array: Input image array
            percentage: Downsample percentage (0-100)
            
        Returns:
            np.ndarray: Downsampled image array
        """
        if percentage >= 100.0:
            return img_array
            
        h, w = img_array.shape[:2]
        
        # Calculate linear scale factor from area percentage
        area_scale = percentage / 100.0
        linear_scale = area_scale ** 0.5  # Square root for linear dimension scaling
        
        # Calculate new dimensions
        new_w = max(1, int(w * linear_scale))
        new_h = max(1, int(h * linear_scale))
        
        # Use OpenCV for high-quality downsampling
        downsampled = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Log the downsampling operation
        original_mp = (w * h) / 1_000_000.0
        new_mp = (new_w * new_h) / 1_000_000.0
        self.signals.log.emit(
            f"[Downsample] {percentage:.1f}% - {w}x{h} ({original_mp:.1f}MP) → {new_w}x{new_h} ({new_mp:.1f}MP)"
        )
        
        return downsampled