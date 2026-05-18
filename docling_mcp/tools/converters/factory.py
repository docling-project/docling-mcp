"""Factory for creating document converters based on configuration."""

from typing import TYPE_CHECKING, Union

from docling_mcp.logger import setup_logger
from docling_mcp.settings.service_client import ConversionMode, settings

if TYPE_CHECKING:
    from .local import LocalDocumentConverter
    from .remote import RemoteDocumentConverter

logger = setup_logger()


def get_converter() -> Union["RemoteDocumentConverter", "LocalDocumentConverter"]:
    """Get the appropriate converter based on settings."""
    # Import converters lazily to avoid importing DocumentConverter when not needed
    from .remote import RemoteDocumentConverter

    logger.info(f"Selecting converter for mode: {settings.conversion_mode}")

    if settings.conversion_mode == ConversionMode.REMOTE:
        try:
            converter = RemoteDocumentConverter()
            if converter.is_available():
                logger.info("Using remote converter")
                return converter
            else:
                logger.warning("Remote service is not available")
                # If service is not available but no fallback, still return the converter
                # It will fail when trying to convert, but that's expected
                if not settings.fallback_to_local:
                    return converter
        except Exception as e:
            logger.error(f"Failed to initialize remote converter: {e}")

            # Only import local converter if fallback is enabled
            if settings.fallback_to_local:
                from .local import LOCAL_CONVERSION_AVAILABLE, LocalDocumentConverter

                if LOCAL_CONVERSION_AVAILABLE:
                    logger.info("Falling back to local converter")
                    return LocalDocumentConverter()
            raise

        # If we get here, fallback is enabled but remote is not available
        if settings.fallback_to_local:
            from .local import LOCAL_CONVERSION_AVAILABLE, LocalDocumentConverter

            if LOCAL_CONVERSION_AVAILABLE:
                logger.info("Falling back to local converter (service unavailable)")
                return LocalDocumentConverter()
            else:
                raise ImportError(
                    "Remote service unavailable and local fallback requires docling-mcp[local] extra. "
                    "Install with: pip install docling-mcp[local]"
                )

        # No fallback, return the unavailable converter
        return converter

    elif settings.conversion_mode == ConversionMode.LOCAL:
        # Import local converter only when needed
        from .local import LOCAL_CONVERSION_AVAILABLE, LocalDocumentConverter

        if not LOCAL_CONVERSION_AVAILABLE:
            raise ImportError(
                "Local conversion mode requires docling-mcp[local] extra. "
                "Install with: pip install docling-mcp[local]"
            )
        logger.info("Using local converter")
        return LocalDocumentConverter()

    raise ValueError(f"Unknown conversion mode: {settings.conversion_mode}")
