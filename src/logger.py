from pathlib import Path
from datetime import datetime
import logging.config

BOLD_SEQ = '\033[1m'
COLOR_SEQ = "\033[1;%dm"
RESET_SEQ = "\033[0m"

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s %(name)-12s %(module)-12s %(funcName)-12s %(levelname)-8s %(message)s",
            "datefmt": '%Y-%m-%d %H:%M:%S'
        },
    },
    "handlers": {
        "file": {
            "level": "DEBUG",
            "formatter": "default",
            "class": "logging.FileHandler",
            "filename": ''.join([
                (Path().cwd() / 'log').absolute().as_posix(),
                '/',
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                '.log'
            ]),
            "encoding": "utf-8",
        },
        "console": {
            "level": "INFO",
            "formatter": "default",
            "class": "logging.StreamHandler",
        },
    },
    "loggers": {
        "": {
            "handlers": [
                "file",
                "console"
            ],
            "level": "DEBUG",
            "propagate": False
        },
    },
}



class DicLogger:
    def __init__(self, log_settings: dict):
        logger = logging.getLogger(__name__)
        logging.config.dictConfig(log_settings)
        self.log = logger