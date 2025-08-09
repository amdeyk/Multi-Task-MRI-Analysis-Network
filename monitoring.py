import logging
from typing import Optional

# Custom log level for medical data events
MEDDATA_LEVEL = logging.INFO + 5
logging.addLevelName(MEDDATA_LEVEL, "MEDDATA")


def meddata(self: logging.Logger, message: str, *args, **kwargs) -> None:
    """Log medical dataset events at MEDDATA level."""
    if self.isEnabledFor(MEDDATA_LEVEL):
        self._log(MEDDATA_LEVEL, message, args, **kwargs)


logging.Logger.meddata = meddata


def setup_logging(level: int = logging.INFO, filename: Optional[str] = None) -> logging.Logger:
    """Configure structured logging for the application.

    Parameters
    ----------
    level: int
        Logging level for the root logger.
    filename: Optional[str]
        Optional log file path. If provided, logs are written to the file in
        addition to standard output.
    """
    handlers = [logging.StreamHandler()]
    if filename:
        handlers.append(logging.FileHandler(filename))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
    )
    logger = logging.getLogger("mri_kan")
    logger.meddata("Logging initialised")
    return logger
