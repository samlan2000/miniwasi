"""
miniwasi: vectorised WASI-type bio-optical model (deep water) with a batched
Levenberg-Marquardt inversion, an ENVI image processor built on it, and a batch processor
for many images.

    from miniwasi import MiniWasi, ImageProcessor, BatchImageProcessor
"""

from .MiniWASI import MiniWasi
from .ImageProcessor import ImageProcessor
from .BatchImageProcessor import BatchImageProcessor

__version__ = "0.3.0"
__all__ = ["MiniWasi", "ImageProcessor", "BatchImageProcessor"]
