"""
miniwasi: vectorised WASI-type bio-optical model (deep water) with a batched
Levenberg-Marquardt inversion, and an ENVI image processor built on it.

    from miniwasi import MiniWasi, ImageProcessor
"""

from .MiniWASI import MiniWasi
from .ImageProcessor import ImageProcessor

__version__ = "0.1.0"
__all__ = ["MiniWasi", "ImageProcessor"]
