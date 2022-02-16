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
# weights
try:
    weights = pd.read_csv(files['weights.csv'])
except Exception as e:
    logger.error('Failed read weights: {}'.format(e))

weights = weights.set_index('combine', drop=False)
WEIGHTS = weights

logger.debug('Weights file contains {}'.format(weights))

# %%
# patients
try:
    patients_data = pd.read_csv(files['patientsData.csv'])
except Exception as e:
    logger.error('Failed read patients_data {}'.format(e))

new_columns = ['{}_{}_{}'.format(a, b, c.split('.')[0])
               for a, b, c in zip(patients_data.iloc[1],
                                  patients_data.iloc[0],
                                  patients_data.columns)]

# Change names to match with weights
for j, e in enumerate(new_columns):
    if e == 'original_shape_LeastAxis':
        new_columns[j] = 'original_shape_LeastAxisLength'

    if e == 'original_shape_MinorAxis':
        new_columns[j] = 'original_shape_MinorAxisLength'

patients_data_1 = patients_data.copy().iloc[2:]

patients_data_1.columns = new_columns
patients_data_1.index = range(len(patients_data_1))
patients_data_1

logger.debug('Patients data has {} entries'.format(len(patients_data_1)))

# %%
# Compute distribution of features
data = []

select_positive = patients_data_1['custom_custom_预后'] == '好'
select_negative = patients_data_1['custom_custom_预后'] == '差'

for j in weights.index:
    n = weights.loc[j, 'combine']
    d = np.array(patients_data_1[n].map(float))
    w = weights.loc[j, 'weight']

    max_pos = np.max(d[select_positive])
    min_pos = np.min(d[select_positive])
    max_neg = np.max(d[select_negative])
    min_neg = np.min(d[select_negative])

    data.append((n, d, (max_pos, min_pos, max_neg, min_neg), w))

data

logger.debug('Parsed patients data for {} positive and {} negative entries'.format(
    np.count_nonzero(select_positive),
    np.count_nonzero(select_negative)
))

# %%
# Compute range table

range_table = pd.DataFrame()
range_table['weightName'] = weights['combine']
range_table['weightValue'] = weights['weight']
range_table['maxPositive'] = [e[2][0] for e in data]
range_table['minPositive'] = [e[2][1] for e in data]
range_table['maxNegative'] = [e[2][2] for e in data]
range_table['minNegative'] = [e[2][3] for e in data]
range_table.set_index('weightName', inplace=True)
range_table

RANGE_TABLE = range_table

logger.debug('The ranges are {}'.format(range_table))

# %%
# Compute offset
values = data[0][1] * 0
for n, d, _, w in data:
    values += d * w

table = pd.DataFrame()
table['label'] = patients_data_1['custom_custom_预后']
table['value'] = values

values_positive = values[select_positive]
values_negative = values[select_negative]

counts = []
for y in values:
    c1 = np.count_nonzero(values_positive > y)
    c2 = np.count_nonzero(values_negative < y)
    counts.append((y, c1 + c2))

counts = sorted(counts, key=lambda e: e[1])

OFFSET = counts[0][0]

# Whether save the html as logs
if False:
    title = 'Offset Estimation - {} - {}'.format(OFFSET, time.time())
    fig = px.box(table, color='label', y='value', title=title)
    fig.write_html(CONFIG['log_folder'].joinpath('{}.html'.format(title)))

    _df = pd.DataFrame(counts, columns=['offset', 'countIncorrect'])
    title = 'Incorrect Count by OFFSET - {} - {}'.format(OFFSET, time.time())
    fig = px.scatter(_df, x='offset', y='countIncorrect', title=title)
    fig.write_html(CONFIG['log_folder'].joinpath('{}.html'.format(title)))

logger.debug('The offset value is estimated as {}'.format(OFFSET))

# %%
# Success information
logger.info('Templates are loaded.')

# %%
