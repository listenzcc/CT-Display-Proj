# %%
import time
import numpy as np
import pandas as pd
import plotly.express as px

from pathlib import Path

from onstart import CONFIG, logger

# %%
# Find all files
folder = CONFIG['templates_folder']

if not folder.is_dir():
    logger.error('Templates folder not found.')

files = dict()
for e in folder.iterdir():
    p = e.as_posix()
    n = e.name
    files[n] = p

files

logger.debug('Templates folder contains {}'.format(files))

# %%
try:
    alff_data = pd.read_csv(files['alff.csv'])
except Exception as e:
    logger.error('Failed read alff {}'.format(e))

alff_data

# %%
ALFF_DATA = dict(
    data_a0_good=alff_data.query('prediction=="好" & area=="视觉区"')[
        'score'].values,
    data_a0_bad=alff_data.query('prediction=="差" & area=="视觉区"')[
        'score'].values,
    data_a1_good=alff_data.query('prediction=="好" & area=="辅助运动区"')[
        'score'].values,
    data_a1_bad=alff_data.query('prediction=="差" & area=="辅助运动区"')[
        'score'].values,
)

for k, e in ALFF_DATA.items():
    logger.debug(
        'ALFF data is read as name: {}, mean: {:0.2f}, count: {}'.format(k, np.mean(e), len(e)))


ALFF_DATA
# %%
