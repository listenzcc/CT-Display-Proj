'''
FileName: onstart.py
Author: Chuncheng
Version: V0.0
Purpose:
'''

# %%
import os
import logging
from pathlib import Path

# %%
CONFIG = dict(
    short_name='CT A.D.',
    app_name='基于影像组学的脑出血预后智能评估系统 V-3.0',
    templates_folder=Path(__file__).joinpath('../../templates'),
    subjects_folder=Path(__file__).joinpath('../../subjects'),
    log_folder=Path(__file__).joinpath('../../log'),
    assets_folder=Path('assets')
    # assets_folder=Path(__file__).joinpath('../assets').relative_to(Path.cwd())
)

CONFIG['acknowledge'] = open(
    Path(__file__).joinpath('../acknowledge.md'), 'rb').read().decode('utf-8')

print(CONFIG['acknowledge'], type(CONFIG['acknowledge']))

# %%
logger_kwargs = dict(
    level_file=logging.DEBUG,
    level_console=logging.DEBUG,
    format_file='%(asctime)s %(name)s %(levelname)-8s %(message)-40s {{%(filename)s:%(lineno)s:%(module)s:%(funcName)s}}',
    format_console='%(asctime)s %(name)s %(levelname)-8s %(message)-40s {{%(filename)s:%(lineno)s}}'
)


def generate_logger(name, filepath, level_file, level_console, format_file, format_console):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(filepath)
    file_handler.setFormatter(logging.Formatter(format_file))
    file_handler.setLevel(level_file)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(format_console))
    console_handler.setLevel(level_console)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = generate_logger('CTDisplay', CONFIG['log_folder'].joinpath(
    'CTDisplay.log'), **logger_kwargs)
logger.info(
    '--------------------------------------------------------------------------------')
logger.info(
    '---- New Session is started ----------------------------------------------------')

# %%
