"""
Image Editing Tools for Scan Space Image Processor

This module provides global image adjustment functions that can be applied
to the entire dataset during processing. Each function follows photography
standards and uses professional image processing techniques.

Performance Features:
    - Ultra-fast mode for images <0.5MP (subsampling + float16)
    - Fast mode for images <2MP (polynomial approximations) 
    - Accurate mode for large images (full Photoshop-style algorithms)
    - Optional GPU acceleration with CuPy for NVIDIA cards
    - Intelligent luminance subsampling for large images
    - Adaptive interpolation methods based on image size

Functions:
    - adjust_exposure: Apply exposure compensation using EV stops
    - adjust_shadows: Modify shadow regions with gradient falloff
    - adjust_highlights: Modify highlight regions with gradient falloff
    - adjust_white_balance: Apply white balance correction
    - sample_white_balance: Calculate WB from neutral point
"""

import numpy as np
from typing import Tuple, Optional, Union
import colour
import cv2
from PySide6.QtCore import Qt
from scipy import ndimage
import gc
import sys
from collections import OrderedDict
import time

# GPU acceleration support (optional)
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None

# Performance optimization flags - simplified for stability
USE_FAST_MODE = True  # Enable optimizations for real-time preview
FAST_MODE_MAX_PIXELS = 2_000_000  # 2MP threshold for fast mode

# =============================================================================
# MEMORY-MANAGED CACHING SYSTEM
# =============================================================================

# Cache configuration
MAX_CACHE_ENTRIES = 5  # Maximum number of cached images
MAX_CACHE_MEMORY_MB = 512  # Maximum cache memory in MB
CACHE_CLEANUP_THRESHOLD = 0.8  # Cleanup when cache reaches 80% of memory limit

class CacheEntry:
    """Represents a single cache entry with metadata."""
    
    def __init__(self, data: np.ndarray, cache_key: str):
        self.data = data
        self.cache_key = cache_key
        self.access_time = time.time()
        self.access_count = 1
        self.size_mb = self._calculate_size_mb(data)
    
    def _calculate_size_mb(self, data: np.ndarray) -> float:
        """Calculate memory size of numpy array in MB."""
        return data.nbytes / (1024 * 1024)
    
    def touch(self):
        """Update access time and count."""
        self.access_time = time.time()
        self.access_count += 1

class MemoryManagedCache:
    """
    LRU cache with memory limits and automatic cleanup.
    
    Features:
    - LRU eviction when memory limit exceeded
    - Automatic cleanup when cache gets too large
    - Memory usage monitoring
    - Cache statistics
    """
    
    def __init__(self, max_entries: int = MAX_CACHE_ENTRIES, max_memory_mb: float = MAX_CACHE_MEMORY_MB):
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb
        self.cache = OrderedDict()  # LRU cache
        self.total_memory_mb = 0.0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, cache_key: str) -> Optional[np.ndarray]:
        """Get item from cache, returns None if not found."""
        if cache_key in self.cache:
            # Move to end (most recently used)
            entry = self.cache.pop(cache_key)
            entry.touch()
            self.cache[cache_key] = entry
            self.hits += 1
            return entry.data.copy()  # Return copy to prevent modification
        else:
            self.misses += 1
            return None
    
    def put(self, cache_key: str, data: np.ndarray):
        """Add item to cache with automatic memory management."""
        entry = CacheEntry(data.copy(), cache_key)  # Store copy to prevent external modification
        
        # Check if key already exists
        if cache_key in self.cache:
            old_entry = self.cache.pop(cache_key)
            self.total_memory_mb -= old_entry.size_mb
        
        # Add new entry
        self.cache[cache_key] = entry
        self.total_memory_mb += entry.size_mb
        
        # Cleanup if needed
        self._cleanup_if_needed()
    
    def _cleanup_if_needed(self):
        """Perform cleanup if cache exceeds limits."""
        # Remove oldest entries if we exceed max entries
        while len(self.cache) > self.max_entries:
            self._evict_oldest()
        
        # Remove entries if we exceed memory limit
        while self.total_memory_mb > self.max_memory_mb:
            if not self.cache:  # Empty cache
                break
            self._evict_oldest()
        
        # Proactive cleanup when approaching limits
        if (self.total_memory_mb > self.max_memory_mb * CACHE_CLEANUP_THRESHOLD or 
            len(self.cache) > self.max_entries * CACHE_CLEANUP_THRESHOLD):
            self._cleanup_least_used()
    
    def _evict_oldest(self):
        """Remove the least recently used entry."""
        if self.cache:
            _, entry = self.cache.popitem(last=False)  # Remove oldest (first) item
            self.total_memory_mb -= entry.size_mb
            self.evictions += 1
    
    def _cleanup_least_used(self):
        """Remove entries with low access counts to free memory."""
        if len(self.cache) <= 2:  # Keep at least 2 entries
            return
        
        # Sort by access count (ascending) and remove least used
        entries_by_usage = sorted(self.cache.items(), key=lambda x: x[1].access_count)
        
        # Remove up to 25% of entries with lowest access counts
        to_remove = min(len(entries_by_usage) // 4, len(entries_by_usage) - 2)
        
        for i in range(to_remove):
            key, entry = entries_by_usage[i]
            if key in self.cache:  # Double-check it's still there
                self.cache.pop(key)
                self.total_memory_mb -= entry.size_mb
                self.evictions += 1
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.total_memory_mb = 0.0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        return {
            'entries': len(self.cache),
            'memory_mb': round(self.total_memory_mb, 2),
            'memory_percent': round((self.total_memory_mb / self.max_memory_mb) * 100, 1),
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': round(hit_rate * 100, 1)
        }
    
    def force_cleanup(self):
        """Force aggressive cleanup to free memory."""
        # Keep only the most recently used entry
        while len(self.cache) > 1:
            self._evict_oldest()
        
        # Force garbage collection
        gc.collect()
        
        # GPU cleanup if available
        if GPU_AVAILABLE and cp is not None:
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass  # Ignore GPU cleanup errors

# Global cache instance
_ADJUSTMENT_CACHE = MemoryManagedCache()
_CURRENT_IMAGE_ID = None

def clear_adjustment_cache():
    """Clear the adjustment cache when switching images."""
    global _ADJUSTMENT_CACHE, _CURRENT_IMAGE_ID
    _ADJUSTMENT_CACHE.clear()
    _CURRENT_IMAGE_ID = None

def set_current_image_id(image_id: str):
    """Set the current image ID for caching purposes."""
    global _CURRENT_IMAGE_ID, _ADJUSTMENT_CACHE
    if _CURRENT_IMAGE_ID != image_id:
        # When switching images, perform aggressive cleanup
        _ADJUSTMENT_CACHE.force_cleanup()
        _CURRENT_IMAGE_ID = image_id

def get_cache_stats() -> dict:
    """Get current cache statistics."""
    return _ADJUSTMENT_CACHE.get_stats()

def check_memory_usage():
    """
    Check current system memory usage and trigger cleanup if needed.
    
    Returns:
        bool: True if cleanup was performed
    """
    try:
        # Try to get memory info (requires psutil if available)
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            
            # Trigger cleanup if system memory is high
            if memory_percent > 85:  # 85% system memory usage
                _ADJUSTMENT_CACHE.force_cleanup()
                return True
            elif memory_percent > 75:  # 75% system memory usage
                _ADJUSTMENT_CACHE._cleanup_least_used()
                return True
        except ImportError:
            # Fallback: check cache memory only
            cache_stats = _ADJUSTMENT_CACHE.get_stats()
            if cache_stats['memory_percent'] > 90:
                _ADJUSTMENT_CACHE.force_cleanup()
                return True
    except Exception:
        # If anything fails, do basic cleanup
        _ADJUSTMENT_CACHE._cleanup_if_needed()
    
    return False
USE_ULTRA_FAST_MODE = True  # Enable ultra-fast approximations for very small images
ULTRA_FAST_MAX_PIXELS = 500_000  # 0.5MP threshold for ultra-fast mode

# LAB processing mode flag
USE_LAB_MODE = True  # Use LAB color space for better quality (RawTherapee-style)


def _should_use_fast_mode(image: np.ndarray) -> bool:
    """Determine if we should use fast mode for this image."""
    if not USE_FAST_MODE:
        return False
    
    total_pixels = image.shape[0] * image.shape[1]
    return total_pixels <= FAST_MODE_MAX_PIXELS


def _should_use_ultra_fast_mode(image: np.ndarray) -> bool:
    """Determine if we should use ultra-fast approximations for very small images."""
    if not USE_ULTRA_FAST_MODE:
        return False
    
    total_pixels = image.shape[0] * image.shape[1]
    return total_pixels <= ULTRA_FAST_MAX_PIXELS


# =============================================================================
# COLOR SPACE CONVERSION UTILITIES
# =============================================================================

def _rgb_to_lab(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert RGB image to LAB color space.
    
    Args:
        image: RGB image array (float32, 0-1 range)
        
    Returns:
        Tuple of (L, a, b) channels
    """
    # Convert sRGB to XYZ, then to LAB
    xyz = colour.sRGB_to_XYZ(image)
    lab = colour.XYZ_to_Lab(xyz)
    
    # Split into separate channels
    L = lab[..., 0]  # Lightness (0-100)
    a = lab[..., 1]  # Green-Red (-128 to 127)
    b = lab[..., 2]  # Blue-Yellow (-128 to 127)
    
    return L, a, b


def _lab_to_rgb(L: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Convert LAB channels back to RGB image.
    
    Args:
        L: Lightness channel (0-100)
        a: Green-Red channel (-128 to 127)
        b: Blue-Yellow channel (-128 to 127)
        
    Returns:
        RGB image array (float32, 0-1 range)
    """
    # Stack channels back together
    lab = np.stack([L, a, b], axis=-1)
    
    # Convert LAB to XYZ, then to sRGB
    xyz = colour.Lab_to_XYZ(lab)
    rgb = colour.XYZ_to_sRGB(xyz)
    
    return np.clip(rgb, 0, 1)


def _is_lab_image(image: np.ndarray) -> bool:
    """
    Check if the image is already in LAB format.
    
    Args:
        image: Input image array
        
    Returns:
        bool: True if image appears to be in LAB format
    """
    # LAB images have L in range 0-100, a/b in range -128 to 127
    # RGB images are typically 0-1 or 0-255
    if image.shape[-1] != 3:
        return False
    
    # Check if values suggest LAB (L channel should be 0-100)
    max_val = np.max(image[..., 0])
    min_a = np.min(image[..., 1])
    min_b = np.min(image[..., 2])
    
    # LAB heuristic: L channel > 1 and a/b channels can be negative
    return max_val > 1.5 and (min_a < -1 or min_b < -1)


def adjust_exposure(image: np.ndarray, exposure_value: float) -> np.ndarray:
    """
    Adjust image exposure using photography standard EV (Exposure Value) stops.
    
    In photography, each stop represents a doubling or halving of light.
    The formula used is: output = input * 2^(exposure_value)
    
    Args:
        image: Input image array (float32, range 0-1)
        exposure_value: Exposure adjustment in stops (-10 to +10)
                       Positive values brighten, negative values darken
    
    Returns:
        np.ndarray: Adjusted image with same shape and dtype
    
    Example:
        +1.0 EV = 2x brighter (one stop up)
        -1.0 EV = 0.5x brightness (one stop down)
        +2.0 EV = 4x brighter (two stops up)
    """
    if exposure_value == 0:
        return image
    
    # Calculate exposure multiplier using photography standard
    # Each stop is a power of 2
    multiplier = np.power(2.0, exposure_value)
    
    # Apply exposure adjustment
    adjusted = image * multiplier
    
    # Clip to valid range
    return np.clip(adjusted, 0, 1)


def adjust_shadows_highlights_combined(
    image: np.ndarray, 
    shadow_amount: float, 
    highlight_amount: float,
    falloff_power: float = 7.0
) -> np.ndarray:
    """
    Apply both shadow and highlight adjustments simultaneously with LAB processing and performance optimizations.
    
    This combined approach uses your original exponential algorithm but processes in LAB color space
    for perceptually uniform adjustments. Uses different algorithms based on image size for optimal performance.
    
    Args:
        image: Input image array (float32, range 0-1)
        shadow_amount: Shadow adjustment strength (-100 to +100)
        highlight_amount: Highlight adjustment strength (-100 to +100)
        falloff_power: Controls the exponential curve steepness (default 8.0)
                      Higher values = sharper falloff, more targeted adjustments
                      Lower values = gentler falloff, broader tonal range affected
                      Range: 2.0 (gentle) to 16.0 (aggressive)
    
    Returns:
        np.ndarray: Image with both shadow and highlight adjustments applied
    """
    if shadow_amount == 0 and highlight_amount == 0:
        return image
    
    # Check if input is LAB format
    is_lab_input = _is_lab_image(image)
    
    # Convert to LAB color space for perceptually uniform adjustments
    if not is_lab_input:
        L, a, b = _rgb_to_lab(image)
        # Normalize L to 0-1 range for calculations
        L_norm = L / 100.0
    else:
        L, a, b = image[..., 0], image[..., 1], image[..., 2]
        L_norm = L / 100.0
    
    # Convert amounts to appropriate scale
    shadows_scaled = shadow_amount * 0.05
    highlights_scaled = highlight_amount * 0.05
    
    ultra_fast = _should_use_ultra_fast_mode(image)
    fast_mode = _should_use_fast_mode(image)
    
    # Use L channel as luminance (already perceptually uniform)
    luminance = L_norm

    highlight_adj = highlights_scaled * (np.power(falloff_power, luminance) - 1.0) if highlights_scaled != 0 else 0
    shadow_adj = shadows_scaled * (np.power(falloff_power, 1.0 - luminance) - 1.0) if shadows_scaled != 0 else 0
    
    # Total adjustment for L channel (scale back to 0-100 range)
    total_adjustment = (highlight_adj + shadow_adj) * 100.0
    
    # Apply adjustment to L channel
    L_adjusted = L + total_adjustment
    L_adjusted = np.clip(L_adjusted, 0, 100)
    
    # Slight chrominance scaling in heavily adjusted regions (optional)
    if abs(shadow_amount) > 50 or abs(highlight_amount) > 50:
        adjustment_strength = np.abs(total_adjustment) / 20.0  # Normalize adjustment strength
        chroma_scale = 1.0 - np.clip(adjustment_strength * 0.05, 0, 0.2)  # Subtle saturation adjustment
        a_adjusted = a * chroma_scale
        b_adjusted = b * chroma_scale
    else:
        a_adjusted = a
        b_adjusted = b
    
    # Convert back to RGB if needed
    if not is_lab_input:
        result = _lab_to_rgb(L_adjusted, a_adjusted, b_adjusted)
        return np.clip(result, 0, 1)
    else:
        # Return as LAB
        return np.stack([L_adjusted, a_adjusted, b_adjusted], axis=-1)


def calculate_white_balance_multipliers(
    current_temp: float, 
    target_temp: float,
    tint: float = 0.0
) -> Tuple[float, float, float]:
    """
    Calculate RGB multipliers for white balance adjustment.
    
    Uses color temperature conversion to calculate the appropriate
    RGB channel multipliers for white balance correction.
    
    Args:
        current_temp: Current color temperature in Kelvin
        target_temp: Target color temperature in Kelvin
        tint: Green/Magenta tint adjustment (-100 to +100)
    
    Returns:
        Tuple of (r_mult, g_mult, b_mult) multipliers
    """
    # Convert temperatures to CIE xy chromaticity coordinates
    # Using Planckian locus for blackbody radiation
    current_xy = colour.temperature.CCT_to_xy(current_temp)
    target_xy = colour.temperature.CCT_to_xy(target_temp)
    
    # Convert to XYZ (normalized to Y=1)
    current_XYZ = colour.xy_to_XYZ(current_xy)
    target_XYZ = colour.xy_to_XYZ(target_xy)
    
    # Convert to RGB using sRGB primaries
    current_rgb = colour.XYZ_to_RGB(
        current_XYZ, 
        colour.RGB_COLOURSPACES['sRGB'],
        apply_cctf_encoding=False
    )
    target_rgb = colour.XYZ_to_RGB(
        target_XYZ,
        colour.RGB_COLOURSPACES['sRGB'],
        apply_cctf_encoding=False
    )
    
    # Calculate multipliers (inverse of current, multiply by target)
    # This compensates for the current WB and applies the target
    multipliers = target_rgb / (current_rgb + 1e-10)  # Avoid division by zero
    
    # Apply tint adjustment (green/magenta)
    if tint != 0:
        tint_factor = 1 + (tint / 100.0) * 0.1
        multipliers[1] *= tint_factor  # Adjust green channel for tint
    
    # Normalize multipliers so the minimum is 1.0
    # This prevents overall darkening
    min_mult = np.min(multipliers)
    if min_mult > 0:
        multipliers = multipliers / min_mult
    
    return tuple(multipliers)


def adjust_white_balance(
    image: np.ndarray,
    current_temp: float,
    target_temp: float,
    tint: float = 0.0
) -> np.ndarray:
    """
    Adjust the white balance of an image by color temperature.
    
    Applies color temperature correction by calculating and applying
    RGB channel multipliers based on blackbody radiation curves.
    
    Args:
        image: Input image array (float32, range 0-1)
        current_temp: Current white balance in Kelvin (from EXIF or estimated)
        target_temp: Target white balance in Kelvin (1000-12000)
        tint: Optional green/magenta tint adjustment (-100 to +100)
    
    Returns:
        np.ndarray: White balance adjusted image
    """
    if current_temp == target_temp and tint == 0:
        return image
    
    # Calculate RGB multipliers
    r_mult, g_mult, b_mult = calculate_white_balance_multipliers(
        current_temp, target_temp, tint
    )
    
    # Apply multipliers to each channel
    adjusted = image.copy()
    adjusted[..., 0] *= r_mult
    adjusted[..., 1] *= g_mult
    adjusted[..., 2] *= b_mult
    
    return np.clip(adjusted, 0, 1)


def sample_white_balance_from_point(
    image: np.ndarray,
    sample_x: int,
    sample_y: int,
    current_temp: float,
    sample_radius: int = 5
) -> float:
    """
    Calculate white balance from a neutral point in the image.
    
    Samples a region around the clicked point and calculates the
    color temperature needed to make that region neutral gray.
    
    Args:
        image: Input image array (float32, range 0-1)
        sample_x: X coordinate of sample point
        sample_y: Y coordinate of sample point
        current_temp: Current white balance temperature in Kelvin
        sample_radius: Radius around point to average (pixels)
    
    Returns:
        float: Calculated white balance temperature in Kelvin
    """
    # Extract sample region with bounds checking
    h, w = image.shape[:2]
    x_min = max(0, sample_x - sample_radius)
    x_max = min(w, sample_x + sample_radius + 1)
    y_min = max(0, sample_y - sample_radius)
    y_max = min(h, sample_y + sample_radius + 1)
    
    # Get average color of sampled region
    sample_region = image[y_min:y_max, x_min:x_max]
    avg_color = np.mean(sample_region, axis=(0, 1))
    
    # Avoid division by zero
    if avg_color[1] < 0.001:  # Green channel too dark
        return current_temp
    
    # Calculate how much we need to adjust each channel to make it neutral
    # Neutral means R=G=B, so we calculate ratios to green channel
    r_ratio = avg_color[0] / avg_color[1]
    b_ratio = avg_color[2] / avg_color[1]
    
    # Estimate color temperature adjustment based on R/B ratio
    # This is a simplified approach - more accurate would use full colorimetric calculations
    # 
    # Warmer (lower temp) = more red, less blue
    # Cooler (higher temp) = less red, more blue
    
    # Calculate temperature shift based on red/blue imbalance
    # Using logarithmic relationship between temperature and R/B ratio
    rb_ratio = r_ratio / (b_ratio + 1e-10)
    
    # Empirical formula for temperature estimation
    # Based on approximate Planckian locus behavior
    if rb_ratio > 1:
        # Image is too warm (reddish), need higher temperature
        temp_multiplier = 1 + np.log(rb_ratio) * 0.3
    else:
        # Image is too cool (bluish), need lower temperature
        temp_multiplier = 1 / (1 + np.log(1/rb_ratio) * 0.3)
    
    # Calculate new temperature
    new_temp = current_temp * temp_multiplier
    
    # Clamp to valid range
    new_temp = np.clip(new_temp, 1000, 12000)
    
    return float(new_temp)


def adjust_sharpen(image: np.ndarray, amount: float = 0.0, radius: float = 1.0, threshold: float = 0.0) -> np.ndarray:
    """
    Apply RawTherapee-style unsharp mask sharpening with LAB processing.
    
    This implementation processes in LAB color space to avoid color shifts
    and uses edge-aware sharpening with threshold control.
    
    Args:
        image: Input image array (float32, range 0-1)
        amount: Sharpening strength (0 to 100, where 0 = no sharpening)
        radius: Blur radius for unsharp mask (0.5 to 5.0 pixels)
        threshold: Threshold to prevent sharpening noise (0 to 100)
    
    Returns:
        np.ndarray: Sharpened image
    """
    if amount <= 0:
        return image
    
    # Check if input is LAB format
    is_lab_input = _is_lab_image(image)
    
    # Convert to LAB color space for luminance-only sharpening
    if not is_lab_input:
        L, a, b = _rgb_to_lab(image)
        # Normalize L to 0-1 range for processing
        L_norm = L / 100.0
    else:
        L, a, b = image[..., 0], image[..., 1], image[..., 2]
        L_norm = L / 100.0
    
    # Convert amount to appropriate scale (0-100 -> 0-2.0)
    amount_scaled = amount / 50.0
    
    # Convert threshold to 0-1 range
    threshold_scaled = threshold / 100.0
    
    # Determine blur radius based on image size for optimal performance
    sigma = radius
    
    # Apply Gaussian blur to create the mask
    # Use larger kernel size for better quality
    kernel_size = int(6 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create Gaussian blur using cv2 for performance
    L_blurred = cv2.GaussianBlur(L_norm.astype(np.float32), 
                                 (kernel_size, kernel_size), 
                                 sigma, 
                                 borderType=cv2.BORDER_REFLECT)
    
    # Calculate unsharp mask (difference between original and blurred)
    unsharp_mask = L_norm - L_blurred
    
    # Apply threshold to reduce noise amplification
    if threshold_scaled > 0:
        # Create edge mask - only sharpen where edges exceed threshold
        edge_strength = np.abs(unsharp_mask)
        edge_mask = np.where(edge_strength > threshold_scaled, 1.0, 
                           edge_strength / threshold_scaled)
        unsharp_mask = unsharp_mask * edge_mask
    
    # Apply RawTherapee-style adaptive sharpening
    # Reduce sharpening in very bright and very dark areas to prevent artifacts
    adaptive_mask = np.ones_like(L_norm)
    
    # Reduce sharpening in highlights (> 0.9) and shadows (< 0.1)
    highlight_mask = np.where(L_norm > 0.9, 
                             1.0 - (L_norm - 0.9) * 10.0,  # Linear falloff
                             1.0)
    shadow_mask = np.where(L_norm < 0.1,
                          L_norm * 10.0,  # Linear ramp up
                          1.0)
    
    adaptive_mask = highlight_mask * shadow_mask
    adaptive_mask = np.clip(adaptive_mask, 0.1, 1.0)  # Keep minimum 10% sharpening
    
    # Apply sharpening with adaptive scaling
    sharpened_L = L_norm + (unsharp_mask * amount_scaled * adaptive_mask)
    
    # Clamp to valid range and convert back to 0-100 scale
    sharpened_L = np.clip(sharpened_L, 0, 1) * 100.0
    
    # Optional: slight saturation enhancement in sharpened areas (RawTherapee feature)
    if amount > 30:  # Only for stronger sharpening
        edge_boost = np.abs(unsharp_mask) * (amount / 100.0) * 0.1
        saturation_boost = 1.0 + edge_boost
        
        # Apply subtle saturation enhancement
        a_enhanced = a * saturation_boost
        b_enhanced = b * saturation_boost
        
        # Clamp chrominance to prevent oversaturation
        a_enhanced = np.clip(a_enhanced, -127, 127)
        b_enhanced = np.clip(b_enhanced, -127, 127)
    else:
        a_enhanced = a
        b_enhanced = b
    
    # Convert back to RGB if needed
    if not is_lab_input:
        result = _lab_to_rgb(sharpened_L, a_enhanced, b_enhanced)
        return np.clip(result, 0, 1)
    else:
        # Return as LAB
        return np.stack([sharpened_L, a_enhanced, b_enhanced], axis=-1)


def apply_all_adjustments(
    image: np.ndarray,
    exposure: float = 0.0,
    shadows: float = 0.0,
    highlights: float = 0.0,
    current_wb: float = 5500.0,
    target_wb: float = 5500.0,
    wb_tint: float = 0.0,
    denoise_strength: float = 0.0,
    sharpen_amount: float = 0.0,
    sharpen_radius: float = 1.0,
    sharpen_threshold: float = 0.0
) -> np.ndarray:
    """
    Apply all editing adjustments in the correct order using optimized algorithms.
    
    Order of operations for best results:
    1. Denoising (should be done early to avoid amplifying noise)
    2. White balance (works on linear data)
    3. Exposure (multiplicative)
    4. Combined Shadows/Highlights (exponential algorithm with LAB processing)
    5. Sharpening (should be done last to avoid amplifying earlier adjustments)
    
    Args:
        image: Input image array (float32, range 0-1)
        exposure: Exposure adjustment in EV stops (-10 to +10)
        shadows: Shadow adjustment (-100 to +100)
        highlights: Highlight adjustment (-100 to +100)
        current_wb: Current white balance in Kelvin
        target_wb: Target white balance in Kelvin
        wb_tint: White balance tint adjustment (-100 to +100)
        denoise_strength: Denoising strength (0 to 100)
        sharpen_amount: Sharpening strength (0 to 100)
        sharpen_radius: Sharpening radius (0.5 to 5.0)
        sharpen_threshold: Sharpening threshold (0 to 100)
    
    Returns:
        np.ndarray: Fully adjusted image
    """

    shadows = shadows / 70
    exposure = exposure
    highlights = highlights * 1.3

    # Start with a copy to avoid modifying original
    result = image.copy()
    
    # 1. Denoising (should be done early to avoid amplifying noise with other adjustments)
    if denoise_strength > 0:
        # Create cache key for denoise operation (simplified for better cache hit rate)
        image_signature = f"{image.shape[0]}x{image.shape[1]}x{image.shape[2] if len(image.shape) > 2 else 1}"
        cache_key = f"denoise_{denoise_strength:.1f}_{image_signature}"
        
        # Check memory usage before caching expensive operations
        check_memory_usage()
        
        # Check if we have cached denoise result
        cached_result = _ADJUSTMENT_CACHE.get(cache_key)
        if cached_result is not None:
            result = cached_result
        else:
            # Apply denoise and cache the result (if not too large)
            result = adjust_denoise(result, denoise_strength)
            
            # Only cache if the result is not too large (prevent caching huge images)
            result_size_mb = result.nbytes / (1024 * 1024)
            if result_size_mb < 200:  # Don't cache images larger than 200MB
                _ADJUSTMENT_CACHE.put(cache_key, result)
    
    # 2. White balance (should be done on linear data)
    if current_wb != target_wb or wb_tint != 0:
        result = adjust_white_balance(result, current_wb, target_wb, wb_tint)
    
    # 3. Exposure adjustment (global multiplicative)
    if exposure != 0:
        result = adjust_exposure(result, exposure)
    
    # 4. Combined shadow/highlight adjustment using exponential algorithm with LAB processing
    # This is more efficient and produces better results than separate adjustments
    if shadows != 0 or highlights != 0:
        result = adjust_shadows_highlights_combined(result, shadows, highlights)
    
    # 5. Sharpening (should be done last to avoid amplifying earlier adjustments)
    if sharpen_amount > 0:
        result = adjust_sharpen(result, sharpen_amount, sharpen_radius, sharpen_threshold)
    
    return result


# Global adjustment values that will be used across the dataset
class GlobalAdjustments:
    """
    Container for global image adjustments applied to entire dataset.
    
    These values are set via UI controls and applied during processing.
    """
    
    def __init__(self):
        self.exposure = 0.0          # -10.0 to +10.0 EV stops (supports float precision)
        self.shadows = 0.0           # -100 to +100
        self.highlights = 0.0        # -100 to +100
        self.white_balance = None    # Target temperature in Kelvin
        self.wb_tint = 0.0          # -100 to +100 green/magenta
        self.denoise_strength = 0.0  # 0 to 100 denoising strength
        self.enabled = False         # Master enable/disable
    
    def reset(self):
        """Reset all adjustments to defaults."""
        self.exposure = 0.0
        self.shadows = 0.0
        self.highlights = 0.0
        self.white_balance = None
        self.wb_tint = 0.0
        self.denoise_strength = 0.0
        self.enabled = False
    
    def has_adjustments(self) -> bool:
        """Check if any adjustments are active."""
        return (
            self.enabled and (
                self.exposure != 0 or
                self.shadows != 0 or
                self.highlights != 0 or
                self.white_balance is not None or
                self.denoise_strength > 0
            )
        )
    
    def apply_to_image(self, image: np.ndarray, current_wb: float = 5500) -> np.ndarray:
        """
        Apply all active adjustments to an image using optimized algorithms.
        
        Args:
            image: Input image array
            current_wb: Current white balance of the image
        
        Returns:
            Adjusted image with all processing applied
        """
        if not self.enabled or not self.has_adjustments():
            return image
        
        target_wb = self.white_balance if self.white_balance else current_wb
        
        return apply_all_adjustments(
            image,
            exposure=self.exposure,
            shadows=self.shadows,
            highlights=self.highlights,
            current_wb=current_wb,
            target_wb=target_wb,
            wb_tint=self.wb_tint,
            denoise_strength=self.denoise_strength
        )


# Global instance for use across application
global_adjustments = GlobalAdjustments()


# =============================================================================
# IMAGE RESCALING FUNCTIONS
# =============================================================================

class RescalingCalculator:
    """
    Handles rescaling calculations for image downsample operations.
    
    Synchronizes between percentage, megapixel, and dimension inputs
    based on the original RAW image dimensions.
    """
    
    def __init__(self):
        self.original_width = 0
        self.original_height = 0
        self.original_megapixels = 0.0
        self._updating = False  # Prevent recursive updates
    
    def set_original_dimensions(self, width: int, height: int):
        """
        Set the original RAW image dimensions.
        
        Args:
            width: Original image width in pixels
            height: Original image height in pixels
        """
        self.original_width = width
        self.original_height = height
        self.original_megapixels = (width * height) / 1_000_000.0
    
    def calculate_from_percentage(self, percentage: float) -> tuple:
        """
        Calculate all rescaling values from percentage input.
        
        Args:
            percentage: Downsample percentage (0-100)
            
        Returns:
            tuple: (megapixels, width, height)
        """
        if self.original_megapixels == 0:
            return 0.0, 0, 0
            
        # Calculate scale factor
        scale_factor = percentage / 100.0
        
        # Calculate new values
        new_megapixels = self.original_megapixels * scale_factor
        new_width = int(self.original_width * (scale_factor ** 0.5))
        new_height = int(self.original_height * (scale_factor ** 0.5))
        
        return new_megapixels, new_width, new_height
    
    def calculate_from_megapixels(self, target_megapixels: float) -> tuple:
        """
        Calculate all rescaling values from megapixel input.
        
        Args:
            target_megapixels: Target megapixels
            
        Returns:
            tuple: (percentage, width, height)
        """
        if self.original_megapixels == 0:
            return 0.0, 0, 0
            
        # Calculate scale factor
        scale_factor = target_megapixels / self.original_megapixels
        scale_factor = max(0.01, min(1.0, scale_factor))  # Clamp between 1% and 100%
        
        # Calculate new values
        percentage = scale_factor * 100.0
        linear_scale = scale_factor ** 0.5
        new_width = int(self.original_width * linear_scale)
        new_height = int(self.original_height * linear_scale)
        
        return percentage, new_width, new_height
    
    def calculate_from_width(self, target_width: int) -> tuple:
        """
        Calculate all rescaling values from width input (maintaining aspect ratio).
        
        Args:
            target_width: Target width in pixels
            
        Returns:
            tuple: (percentage, megapixels, height)
        """
        if self.original_width == 0:
            return 0.0, 0.0, 0
            
        # Calculate scale factor from width
        linear_scale = target_width / self.original_width
        linear_scale = max(0.1, min(1.0, linear_scale))  # Clamp between 10% and 100%
        
        # Calculate new values maintaining aspect ratio
        new_height = int(self.original_height * linear_scale)
        scale_factor = linear_scale ** 2  # Area scale factor
        percentage = scale_factor * 100.0
        new_megapixels = self.original_megapixels * scale_factor
        
        return percentage, new_megapixels, new_height
    
    def calculate_from_height(self, target_height: int) -> tuple:
        """
        Calculate all rescaling values from height input (maintaining aspect ratio).
        
        Args:
            target_height: Target height in pixels
            
        Returns:
            tuple: (percentage, megapixels, width)
        """
        if self.original_height == 0:
            return 0.0, 0.0, 0
            
        # Calculate scale factor from height
        linear_scale = target_height / self.original_height
        linear_scale = max(0.1, min(1.0, linear_scale))  # Clamp between 10% and 100%
        
        # Calculate new values maintaining aspect ratio
        new_width = int(self.original_width * linear_scale)
        scale_factor = linear_scale ** 2  # Area scale factor
        percentage = scale_factor * 100.0
        new_megapixels = self.original_megapixels * scale_factor
        
        return percentage, new_megapixels, new_width


def setup_rescaling_controls(main_window):
    """
    Setup the rescaling controls and connect their signals.
    
    Args:
        main_window: Reference to the main window instance
    """
    # Create rescaling calculator instance
    main_window.rescaling_calculator = RescalingCalculator()
    
    # Connect the enableRescaleCheckBox
    if hasattr(main_window.ui, 'enableRescaleCheckBox'):
        main_window.ui.enableRescaleCheckBox.toggled.connect(
            lambda checked: _on_rescale_enabled_toggled(main_window, checked)
        )
    
    # Connect the input controls
    if hasattr(main_window.ui, 'targetDownsamplePercentDoubleSpinbox'):
        main_window.ui.targetDownsamplePercentDoubleSpinbox.valueChanged.connect(
            lambda value: _on_percentage_changed(main_window, value)
        )
    
    if hasattr(main_window.ui, 'targetDownsampleMpxDoubleSpinBox'):
        main_window.ui.targetDownsampleMpxDoubleSpinBox.valueChanged.connect(
            lambda value: _on_megapixels_changed(main_window, value)
        )
    
    if hasattr(main_window.ui, 'xDimensionOutputSpinbox'):
        main_window.ui.xDimensionOutputSpinbox.valueChanged.connect(
            lambda value: _on_width_changed(main_window, value)
        )
    
    if hasattr(main_window.ui, 'yDiemnsionOutputSpinbox'):
        main_window.ui.yDiemnsionOutputSpinbox.valueChanged.connect(
            lambda value: _on_height_changed(main_window, value)
        )


def _on_rescale_enabled_toggled(main_window, checked):
    """Handle rescale checkbox toggle."""
    # Enable/disable the rescaling controls
    controls = [
        'targetDownsamplePercentDoubleSpinbox',
        'targetDownsampleMpxDoubleSpinBox', 
        'xDimensionOutputSpinbox',
        'yDiemnsionOutputSpinbox'
    ]
    
    for control_name in controls:
        if hasattr(main_window.ui, control_name):
            control = getattr(main_window.ui, control_name)
            control.setEnabled(checked)
    
    if checked:
        # Initialize with current image dimensions
        _update_rescaling_from_current_image(main_window)


def _update_rescaling_from_current_image(main_window):
    """Update rescaling calculator with current image dimensions."""
    try:
        # Get currently selected image
        current_item = main_window.ui.imagesListWidget.currentItem()
        if not current_item:
            return
            
        metadata = current_item.data(Qt.UserRole)
        if metadata.get('is_group_header', False):
            return
            
        image_path = metadata.get('input_path')
        if not image_path:
            return
        
        # Try to get dimensions from RAW file metadata (you may need to implement this)
        # For now, we'll use some example dimensions - you should replace this with actual metadata reading
        width, height = _get_image_dimensions(image_path)
        
        if width > 0 and height > 0:
            main_window.rescaling_calculator.set_original_dimensions(width, height)
            
            # Initialize UI controls with current image info
            main_window.rescaling_calculator._updating = True
            try:
                # Set percentage to 100% (no downsampling)
                if hasattr(main_window.ui, 'targetDownsamplePercentDoubleSpinbox'):
                    main_window.ui.targetDownsamplePercentDoubleSpinbox.setValue(100.0)
                
                # Set megapixels to current image
                if hasattr(main_window.ui, 'targetDownsampleMpxDoubleSpinBox'):
                    main_window.ui.targetDownsampleMpxDoubleSpinBox.setValue(main_window.rescaling_calculator.original_megapixels)
                
                # Set dimensions to current image
                if hasattr(main_window.ui, 'xDimensionOutputSpinbox'):
                    main_window.ui.xDimensionOutputSpinbox.setValue(width)
                
                if hasattr(main_window.ui, 'yDiemnsionOutputSpinbox'):
                    main_window.ui.yDiemnsionOutputSpinbox.setValue(height)
                    
            finally:
                main_window.rescaling_calculator._updating = False
                

    except Exception as e:
        if hasattr(main_window, 'log_error'):
            main_window.log_error(f"[Rescaling] Error updating from current image: {e}")


def _get_image_dimensions(image_path: str) -> tuple:
    """
    Get image dimensions from file metadata.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (width, height) or (0, 0) if unable to read
    """
    try:
        # For RAW files, try to use rawpy to get dimensions
        if image_path.lower().endswith(('.nef', '.cr2', '.cr3', '.dng', '.arw', '.raw')):
            try:
                import rawpy
                with rawpy.imread(image_path) as raw:
                    return raw.sizes.raw_width, raw.sizes.raw_height
            except Exception:
                pass
        
        # For other image formats, try PIL
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                return img.size  # Returns (width, height)
        except Exception:
            pass
            
        # Try OpenCV as fallback
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is not None:
                h, w = img.shape[:2]
                return w, h
        except Exception:
            pass
            
    except Exception:
        pass
    
    # Return default if all methods fail
    return 0, 0


def _on_percentage_changed(main_window, value):
    """Handle percentage spinbox change."""
    if main_window.rescaling_calculator._updating:
        return
        
    main_window.rescaling_calculator._updating = True
    try:
        megapixels, width, height = main_window.rescaling_calculator.calculate_from_percentage(value)
        
        # Update other controls
        if hasattr(main_window.ui, 'targetDownsampleMpxDoubleSpinBox'):
            main_window.ui.targetDownsampleMpxDoubleSpinBox.setValue(megapixels)
        
        if hasattr(main_window.ui, 'xDimensionOutputSpinbox'):
            main_window.ui.xDimensionOutputSpinbox.setValue(width)
            
        if hasattr(main_window.ui, 'yDiemnsionOutputSpinbox'):
            main_window.ui.yDiemnsionOutputSpinbox.setValue(height)
            
    finally:
        main_window.rescaling_calculator._updating = False


def _on_megapixels_changed(main_window, value):
    """Handle megapixels spinbox change."""
    if main_window.rescaling_calculator._updating:
        return
        
    main_window.rescaling_calculator._updating = True
    try:
        percentage, width, height = main_window.rescaling_calculator.calculate_from_megapixels(value)
        
        # Update other controls
        if hasattr(main_window.ui, 'targetDownsamplePercentDoubleSpinbox'):
            main_window.ui.targetDownsamplePercentDoubleSpinbox.setValue(percentage)
        
        if hasattr(main_window.ui, 'xDimensionOutputSpinbox'):
            main_window.ui.xDimensionOutputSpinbox.setValue(width)
            
        if hasattr(main_window.ui, 'yDiemnsionOutputSpinbox'):
            main_window.ui.yDiemnsionOutputSpinbox.setValue(height)
            
    finally:
        main_window.rescaling_calculator._updating = False


def _on_width_changed(main_window, value):
    """Handle width spinbox change."""
    if main_window.rescaling_calculator._updating:
        return
        
    main_window.rescaling_calculator._updating = True
    try:
        percentage, megapixels, height = main_window.rescaling_calculator.calculate_from_width(value)
        
        # Update other controls
        if hasattr(main_window.ui, 'targetDownsamplePercentDoubleSpinbox'):
            main_window.ui.targetDownsamplePercentDoubleSpinbox.setValue(percentage)
        
        if hasattr(main_window.ui, 'targetDownsampleMpxDoubleSpinBox'):
            main_window.ui.targetDownsampleMpxDoubleSpinBox.setValue(megapixels)
            
        if hasattr(main_window.ui, 'yDiemnsionOutputSpinbox'):
            main_window.ui.yDiemnsionOutputSpinbox.setValue(height)
            
    finally:
        main_window.rescaling_calculator._updating = False


def _on_height_changed(main_window, value):
    """Handle height spinbox change."""
    if main_window.rescaling_calculator._updating:
        return
        
    main_window.rescaling_calculator._updating = True
    try:
        percentage, megapixels, width = main_window.rescaling_calculator.calculate_from_height(value)
        
        # Update other controls
        if hasattr(main_window.ui, 'targetDownsamplePercentDoubleSpinbox'):
            main_window.ui.targetDownsamplePercentDoubleSpinbox.setValue(percentage)
        
        if hasattr(main_window.ui, 'targetDownsampleMpxDoubleSpinBox'):
            main_window.ui.targetDownsampleMpxDoubleSpinBox.setValue(megapixels)
            
        if hasattr(main_window.ui, 'xDimensionOutputSpinbox'):
            main_window.ui.xDimensionOutputSpinbox.setValue(width)
            
    finally:
        main_window.rescaling_calculator._updating = False


# =============================================================================
# DENOISING FUNCTIONS
# =============================================================================
# Based on RawTherapee's impulse_denoise.cc algorithm
# Implements edge-preserving denoising with multi-scale processing

def _detect_impulse_noise(image: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Detect impulse noise using median filter comparison (RawTherapee approach).
    
    Args:
        image: Input image array (single channel)
        threshold: Noise detection threshold
        
    Returns:
        Binary mask of detected noise pixels
    """
    # Apply median filter to get noise-free reference
    median_filtered = cv2.medianBlur((image * 255).astype(np.uint8), 3) / 255.0
    
    # Calculate absolute difference
    diff = np.abs(image - median_filtered)
    
    # Detect pixels that deviate significantly from median
    noise_mask = diff > threshold
    
    return noise_mask


def _adaptive_bilateral_filter(image: np.ndarray, sigma_color: float, sigma_space: float) -> np.ndarray:
    """
    Apply bilateral filtering with adaptive parameters (RawTherapee style).
    
    Args:
        image: Input image array
        sigma_color: Color sigma for bilateral filter
        sigma_space: Space sigma for bilateral filter
        
    Returns:
        Filtered image
    """
    # Convert to 8-bit for cv2.bilateralFilter
    img_8bit = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    # Apply bilateral filter to each channel
    if len(image.shape) == 3:
        filtered_channels = []
        for c in range(image.shape[2]):
            filtered_c = cv2.bilateralFilter(img_8bit[..., c], -1, sigma_color * 255, sigma_space)
            filtered_channels.append(filtered_c)
        filtered = np.stack(filtered_channels, axis=2)
    else:
        filtered = cv2.bilateralFilter(img_8bit, -1, sigma_color * 255, sigma_space)
    
    return filtered.astype(np.float32) / 255.0


def _edge_preserving_denoise(image: np.ndarray, strength: float) -> np.ndarray:
    """
    Advanced edge-preserving denoising similar to RawTherapee's approach.
    
    Args:
        image: Input image array (float32, 0-1 range)
        strength: Denoising strength (0-100)
        
    Returns:
        Denoised image
    """
    if strength <= 0:
        return image
        
    # Calculate adaptive parameters based on strength
    strength_normalized = strength / 100.0
    
    # Multi-scale denoising approach
    result = image.copy()
    
    # Stage 1: Impulse noise removal (median filtering)
    if strength > 20:
        # Detect impulse noise
        if len(image.shape) == 3:
            # Process each channel for color images
            impulse_masks = []
            for c in range(3):
                mask = _detect_impulse_noise(image[..., c], threshold=0.05 * strength_normalized)
                impulse_masks.append(mask)
            
            # Apply median filter only to noisy pixels
            median_kernel_size = min(5, int(2 * strength_normalized * 3) + 1)
            if median_kernel_size % 2 == 0:
                median_kernel_size += 1
                
            median_filtered = cv2.medianBlur((result * 255).astype(np.uint8), median_kernel_size).astype(np.float32) / 255.0
            
            # Blend median result only where impulse noise was detected
            for c in range(3):
                noise_blend = strength_normalized * 0.8  # Reduce strength for impulse correction
                result[..., c] = np.where(impulse_masks[c], 
                                        (1 - noise_blend) * result[..., c] + noise_blend * median_filtered[..., c],
                                        result[..., c])
        else:
            # Grayscale processing
            impulse_mask = _detect_impulse_noise(image, threshold=0.05 * strength_normalized)
            median_kernel_size = min(5, int(2 * strength_normalized * 3) + 1)
            if median_kernel_size % 2 == 0:
                median_kernel_size += 1
            median_filtered = cv2.medianBlur((result * 255).astype(np.uint8), median_kernel_size).astype(np.float32) / 255.0
            noise_blend = strength_normalized * 0.8
            result = np.where(impulse_mask, 
                            (1 - noise_blend) * result + noise_blend * median_filtered,
                            result)
    
    # Stage 2: Gaussian noise reduction (bilateral filtering)
    if strength > 10:
        # Adaptive bilateral filter parameters
        sigma_color = 0.1 * strength_normalized  # Color similarity
        sigma_space = 2.0 * strength_normalized   # Spatial proximity
        
        bilateral_filtered = _adaptive_bilateral_filter(result, sigma_color, sigma_space)
        
        # Blend with original based on strength
        bilateral_blend = strength_normalized * 0.6  # Moderate blending for bilateral
        result = (1 - bilateral_blend) * result + bilateral_blend * bilateral_filtered
    
    # Stage 3: Fine detail preservation (edge detection and protection)
    if strength > 30:
        # Detect edges to preserve fine details
        if len(image.shape) == 3:
            # Convert to grayscale for edge detection
            gray = 0.299 * result[..., 0] + 0.587 * result[..., 1] + 0.114 * result[..., 2]
        else:
            gray = result
            
        # Sobel edge detection
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Normalize edge magnitude
        edge_magnitude = np.clip(edge_magnitude / (edge_magnitude.max() + 1e-8), 0, 1)
        
        # Create edge preservation mask (stronger preservation for high-contrast edges)
        edge_preservation = 1.0 - (edge_magnitude * 0.7)  # Preserve 70% of detail near edges
        
        # Apply additional gentle smoothing with edge protection
        if len(result.shape) == 3:
            edge_preservation = edge_preservation[..., np.newaxis]  # Broadcast to color channels
        
        # Gentle Gaussian blur for final smoothing
        gaussian_sigma = strength_normalized * 0.8
        gaussian_filtered = cv2.GaussianBlur(result, (0, 0), gaussian_sigma)
        
        # Blend with edge preservation
        final_blend = strength_normalized * 0.3 * edge_preservation  # Light final smoothing
        result = (1 - final_blend) * result + final_blend * gaussian_filtered
    
    return np.clip(result, 0, 1)


def adjust_denoise(image: np.ndarray, denoise_strength: float) -> np.ndarray:
    """
    Apply image denoising using RawTherapee-inspired algorithms with performance optimizations.
    
    Implements a multi-stage denoising approach:
    1. Impulse noise removal (median filtering)
    2. Gaussian noise reduction (bilateral filtering)
    3. Edge-preserving smoothing with detail protection
    
    Uses different algorithms based on image size:
    - Ultra-fast: <0.5MP - Simple bilateral filtering
    - Fast: <2MP - Two-stage denoising (impulse + bilateral)
    - Accurate: ≥2MP - Full three-stage processing
    
    Args:
        image: Input image array (float32, range 0-1)
        denoise_strength: Denoising strength (0 to 100)
                         0 = no denoising, 100 = maximum denoising
    
    Returns:
        np.ndarray: Denoised image
    """
    if denoise_strength <= 0:
        return image
        
    # Clamp strength to valid range
    denoise_strength = np.clip(denoise_strength, 0, 100)
    
    ultra_fast = _should_use_ultra_fast_mode(image)
    fast_mode = _should_use_fast_mode(image)
    
    if ultra_fast:
        # Ultra-fast mode: Simple bilateral filtering only
        strength_normalized = denoise_strength / 100.0
        sigma_color = 0.08 * strength_normalized
        sigma_space = 1.5 * strength_normalized
        
        result = _adaptive_bilateral_filter(image, sigma_color, sigma_space)
        
        # Blend with original
        blend_factor = strength_normalized * 0.7
        return (1 - blend_factor) * image + blend_factor * result
        
    elif fast_mode:
        # Fast mode: Two-stage processing (impulse + bilateral)
        strength_normalized = denoise_strength / 100.0
        result = image.copy()
        
        # Stage 1: Impulse noise (if strength > 25)
        if denoise_strength > 25:
            median_kernel = min(5, int(strength_normalized * 4) + 1)
            if median_kernel % 2 == 0:
                median_kernel += 1
            median_filtered = cv2.medianBlur((result * 255).astype(np.uint8), median_kernel).astype(np.float32) / 255.0
            impulse_blend = strength_normalized * 0.4
            result = (1 - impulse_blend) * result + impulse_blend * median_filtered
        
        # Stage 2: Bilateral filtering
        sigma_color = 0.1 * strength_normalized
        sigma_space = 2.0 * strength_normalized
        bilateral_result = _adaptive_bilateral_filter(result, sigma_color, sigma_space)
        
        bilateral_blend = strength_normalized * 0.6
        result = (1 - bilateral_blend) * result + bilateral_blend * bilateral_result
        
        return np.clip(result, 0, 1)
        
    else:
        # Accurate mode: Full three-stage processing
        return _edge_preserving_denoise(image, denoise_strength)