"""Model architectures for marine debris detection."""

from src.models.segformer import SegFormerMultispectral, create_model

__all__ = ["SegFormerMultispectral", "create_model"]
