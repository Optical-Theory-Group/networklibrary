import logging
import os


def setup_logging():
    log_file = '{}/simulation.log'.format(os.path.dirname(os.path.abspath(__file__)))
    log_formatter = logging.Formatter('%(asctime)s - %(module)s:%(lineno)s - %(levelname)s - %(message)s',
                                      datefmt='%Y-%m-%d %H:%M:%S')
    level = logging.ERROR

    # Create a logger object for the  current module/script
    logger = logging.getLogger()
    logger.setLevel(level)

    # Create a file handler for the log file
    file_handler = logging.FileHandler(log_file,'w')
    file_handler.setLevel(level)
    file_handler.setFormatter(log_formatter)

    # Create a console handler for logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(log_formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

