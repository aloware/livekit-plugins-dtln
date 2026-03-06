from livekit.agents import Plugin
import logging

from .noise_suppressor import DTLNNoiseSuppressor

logger = logging.getLogger(__name__)


class DTLNPlugin(Plugin):
    def __init__(self):
        super().__init__(
            title="DTLN",
            version="0.1.0",
            package="livekit-plugins-dtln",
            logger=logger,
        )

    def download_files(self):
        from .noise_suppressor import download_models
        download_models()


def noise_suppression(**kwargs) -> DTLNNoiseSuppressor:
    """Create a DTLNNoiseSuppressor instance.

    Pass to AudioInputOptions(noise_cancellation=dtln.noise_suppression()).
    """
    return DTLNNoiseSuppressor(**kwargs)


Plugin.register_plugin(DTLNPlugin())

__all__ = ["DTLNNoiseSuppressor", "noise_suppression"]
