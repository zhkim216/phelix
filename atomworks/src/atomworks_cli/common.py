import logging


def enable_logging(verbose: bool = False) -> None:
    # Set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)

    # Make sure all AtomWorks loggers are set to the desired level
    for name in logging.root.manager.loggerDict:
        if name.startswith("atomworks"):
            logging.getLogger(name).setLevel(log_level)
