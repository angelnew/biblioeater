import logging
from logging.config import fileConfig
import os

global nlp_logger

if 'nlp_logger' not in globals():
    log_fn = os.path.join(os.path.dirname(__file__), "conf/logging.ini")
    fileConfig(log_fn, disable_existing_loggers=False)
    nlp_logger = logging.getLogger("nlp")
