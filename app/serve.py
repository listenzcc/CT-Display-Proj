# %%
import json
import flask
from pathlib import Path
import base64
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from skimage import measure
from tqdm.auto import tqdm

import threading

from werkzeug.serving import run_simple

from onstart import CONFIG, logger

from load_templates import WEIGHTS, RANGE_TABLE, OFFSET
from load_alff import ALFF_DATA
from load_subjects import SUBJECT_MANAGER

from gui import gui

# %%
external_stylesheets = [
    CONFIG['assets_folder'].joinpath('debug-style.css').as_posix(),
    CONFIG['assets_folder'].joinpath('basic-style.css').as_posix(),
]

meta_tags = [
    # {'name': 'Cache-Control', 'content': 'no-cache, no-store, must-revalidate'},
    # {'name': 'Pragma', 'content': 'no-cache'},
    # {'name': 'Expires', 'content': '0'}
]


app = dash.Dash(
    'mainTopic',  # CONFIG['app_name'],
    external_stylesheets=external_stylesheets,
    meta_tags=meta_tags,
    # prevent_initial_callbacks=True,
)

# %%
dynamic_data = dict(
    slice_distance=10,
    pixel_resolution=1.0,
)

# %%


def mk_threshold_options():
    options = []
    for j in range(20, 60, 2):
        options.append(dict(
            label=j,
            value=j
        ))
    return options


def subject_onselect(subject):

    dynamic_data['subject'] = subject

    # mk_features_table(subject)
    # mk_figures(subject)

    mk_figure_3d(subject)
    mk_figures_slices(subject, use_dynamic_data=False)

    logger.debug('The dynamic_data is updated, the keys are: {}.'.format(
        [e for e in dynamic_data]))

    return


def slice_view_recompute(threshold, use_dynamic_data):
    subject = dynamic_data['subject']
    mk_figures_slices(subject, threshold, shrink=True,
                      use_dynamic_data=use_dynamic_data)
    return


def mk_ALFF_table(value0, value1):
    # Basically, the good values are larger
    values0_good = ALFF_DATA['data_a0_good']
    values0_bad = ALFF_DATA['data_a0_bad']
    values1_good = ALFF_DATA['data_a1_good']
    values1_bad = ALFF_DATA['data_a1_bad']

    p0_good = np.count_nonzero(values0_good < value0) / len(values0_good)
    p0_bad = np.count_nonzero(values0_bad > value0) / len(values0_bad)

    p1_good = np.count_nonzero(values1_good < value1) / len(values1_good)
    p1_bad = np.count_nonzero(values1_bad > value1) / len(values1_bad)

    logger.debug('ALFF probabilities is computed as {}, {}, {}, {}'.format(
        p0_good, p0_bad, p1_good, p1_bad))

    data = [
        ('?????????', '???', '{:0.2f}'.format(p0_good)),
        ('?????????', '???', '{:0.2f}'.format(p0_bad)),
        ('???????????????', '???', '{:0.2f}'.format(p1_good)),
        ('???????????????', '???', '{:0.2f}'.format(p1_bad)),
    ]

    columns = ['??????', '??????', '??????']
    df = pd.DataFrame(data, columns=columns)

    data = df.to_dict('records')

    _columns = [{'name': e, 'id': e} for e in columns]

    table_obj = dash_table.DataTable(
        columns=_columns,
        data=data
    )

    score = p0_good * 0.5 + p1_good * 0.5

    logger.debug('The ALFF score is {}, {}, {}'.format(
        score, p0_good, p1_good))

    return table_obj, score


def mk_features_table(subject):
    # Try get_features only use the subject,
    # Or will try to compute the features.

    # Not using it
    # try:
    #     df = SUBJECT_MANAGER.get_features(subject)
    # except AssertionError:
    #     img_array = SUBJECT_MANAGER.get_array(subject)
    #     img_contour = SUBJECT_MANAGER.compute_contour(img_array)
    #     df = SUBJECT_MANAGER.get_features(subject, img_array, img_contour)

    logger.debug('Computing features will cost about 1 - 5 minutes.')
    img_array = dynamic_data['img_array']
    img_contour = dynamic_data['img_contour'].copy()
    img_contour[img_contour >= 200] = 0
    df = SUBJECT_MANAGER.get_features(subject, img_array, img_contour)

    # N.A. refers we found no ROI,
    # so we do nothing for it.
    score = 'N.A.'

    if df.iloc[0]['name'] != 'N.A.':

        def _filter(name):
            b = name in [e for e in WEIGHTS['combine']]
            return b

        df = df[df['name'].map(_filter)]

        columns = ['maxPositive', 'minPositive', 'maxNegative', 'minNegative']
        for col in columns:
            df[col] = df['name'].map(lambda e: RANGE_TABLE.loc[e, col])

        def _range_check(se):
            values = [se[e] for e in columns]
            b = all([se['value'] < np.max(values),
                     se['value'] > np.min(values)])
            if b:
                return 'Valid'
            else:
                return 'Warn'

        df['Status'] = df.apply(_range_check, axis=1)
        df['Weight'] = df['name'].map(lambda e: WEIGHTS.loc[e, 'weight'])

        score = np.sum(np.array(df['Weight']) * np.array(df['value'])) - OFFSET

        def _format(e):
            return '{:0.4f}'.format(e)

        df['value'] = df['value'].map(_format)
        for col in columns:
            df[col] = df[col].map(_format)

    df['threshold'] = dynamic_data['threshold']
    columns = [{"name": i, "id": i} for i in df.columns]
    data = df.to_dict('records')

    table_obj = dash_table.DataTable(
        columns=columns,
        data=data
    )

    dynamic_data['table_obj'] = table_obj
    dynamic_data['score'] = score

    return table_obj, score


def mk_figure_3d(subject):
    img_array = SUBJECT_MANAGER.get_array(subject)
    img_contour = SUBJECT_MANAGER.compute_contour(img_array)

    # --------------------------------------------------------------------------------
    # The fig_contour is the 3D view of the contour surface
    data = []

    # Skull
    color = 'grey'
    verts, faces, normals, values = measure.marching_cubes(img_array,
                                                           500,
                                                           step_size=3)
    x, y, z = verts.T
    i, j, k = faces.T

    logger.debug('Using color: "{}" rendering vertex with shape {}'.format(
        color, [e.shape for e in [x, y, z, i, j, k]]))

    data.append(
        go.Mesh3d(x=x, y=y, z=z, color=color, opacity=0.2, i=i, j=j, k=k)
    )

    # Target
    if np.count_nonzero(img_contour > 50):
        color = 'purple'
        verts, faces, normals, values = measure.marching_cubes(img_contour,
                                                               50,
                                                               step_size=3)
        x, y, z = verts.T
        i, j, k = faces.T

        logger.debug('Using color: "{}" rendering vertex with shape {}'.format(
            color, [e.shape for e in [x, y, z, i, j, k]]))

        data.append(
            go.Mesh3d(x=x, y=y, z=z, color=color, opacity=0.3, i=i, j=j, k=k)
        )

    shape = img_array.shape
    _x = shape[0] * dynamic_data['slice_distance'] / 300
    _y = shape[1] * dynamic_data['pixel_resolution'] / 300
    _z = shape[2] * dynamic_data['pixel_resolution'] / 300
    layout = dict(scene={'aspectratio': {'x': _x, 'y': _y, 'z': _z}},
                  # scene={'aspectmode': 'data'},
                  width=600,
                  title=subject)
    fig_contour = go.Figure(data, layout=layout)

    dynamic_data['fig_contour'] = fig_contour
    dynamic_data['img_array'] = img_array
    dynamic_data['img_contour'] = img_contour
    logger.debug('The fig of 3D is updated.')

    return fig_contour


def mk_figures_slices(subject, threshold=None, shrink=False, use_dynamic_data=False):
    img_array = SUBJECT_MANAGER.get_array(subject)
    if use_dynamic_data:
        img_contour = dynamic_data['img_contour']
    else:
        img_contour = SUBJECT_MANAGER.compute_contour(
            img_array, threshold, shrink)

    if threshold is None:
        threshold = 50

    a = img_contour > threshold
    b = img_contour < 300
    count_ROI = np.count_nonzero(a * b)
    logger.debug('The ROI count is {}'.format(count_ROI))

    dynamic_data['threshold'] = threshold

    # --------------------------------------------------------------------------------
    # The figs is a list of slice views
    figs_slices = []
    # range_color = (-1000, 2000)
    range_color = (0, 200)
    for j in tqdm(range(len(img_array)), 'Prepare Slices'):
        # Two-layers will be generated.
        # The background is the gray-scaled brain slice view.
        fig = px.imshow(img_array[j],
                        range_color=range_color,
                        color_continuous_scale='gray')

        # The upper layer is the contour of values between start=50 and end=100,
        # it is designed to be the detected object
        fig.add_trace(go.Contour(z=img_contour[j],
                                 showscale=False,
                                 hoverinfo='skip',
                                 line_width=2,
                                 contours=dict(
                                     start=threshold,
                                     end=101,
                                     size=25,
                                     coloring='lines',
                                     showlabels=False,
                                     labelfont=dict(size=12, color='white'))))

        _vol = dynamic_data['slice_distance'] * \
            dynamic_data['pixel_resolution'] * \
            dynamic_data['pixel_resolution'] / 1000
        fig.update_layout({'title': 'Subj:{} CutBy:{} Slice:{} Volume:{:0.2f} ml'.format(subject, threshold, j, count_ROI * _vol),
                           'dragmode': 'drawrect',
                           'width': 580,
                           'newshape.line.width': 1,
                           'newshape.line.color': 'cyan'})
        figs_slices.append(fig)
        pass

    dynamic_data['figs_slices'] = figs_slices
    # If the dynamic_data['img_contour'] is not used,
    # it should be updated.
    if not use_dynamic_data:
        dynamic_data['img_contour'] = img_contour

    logger.debug('The figs_slices is updated.')

    return figs_slices


def mk_figures(subject):
    raise DeprecationWarning('The mk_figures method is deprecated.')
    return 'Invalid Method'
    img_array = SUBJECT_MANAGER.get_array(subject)
    img_contour = SUBJECT_MANAGER.compute_contour(img_array)

    # --------------------------------------------------------------------------------
    # The fig_contour is the 3D view of the contour surface
    data = []

    # Skull
    color = 'grey'
    verts, faces, normals, values = measure.marching_cubes(img_array,
                                                           500,
                                                           step_size=3)
    x, y, z = verts.T
    i, j, k = faces.T

    logger.debug('Using color: "{}" rendering vertex with shape {}'.format(
        color, [e.shape for e in [x, y, z, i, j, k]]))

    data.append(
        go.Mesh3d(x=x, y=y, z=z, color=color, opacity=0.2, i=i, j=j, k=k)
    )

    # Target
    if np.count_nonzero(img_contour > 50):
        color = 'purple'
        verts, faces, normals, values = measure.marching_cubes(img_contour,
                                                               50,
                                                               step_size=3)
        x, y, z = verts.T
        i, j, k = faces.T

        logger.debug('Using color: "{}" rendering vertex with shape {}'.format(
            color, [e.shape for e in [x, y, z, i, j, k]]))

        data.append(
            go.Mesh3d(x=x, y=y, z=z, color=color, opacity=0.3, i=i, j=j, k=k)
        )

    layout = dict(scene={'aspectmode': 'data'},
                  width=600,
                  title=subject)
    fig_contour = go.Figure(data, layout=layout)

    dynamic_data['fig_contour'] = fig_contour
    logger.debug('The fig of 3D is updated.')

    # --------------------------------------------------------------------------------
    # The figs is a list of slice views
    figs_slices = []
    range_color = (-1000, 2000)
    range_color = (0, 200)
    for j in tqdm(range(len(img_array)), 'Prepare Slices'):
        # Two-layers will be generated.
        # The background is the gray-scaled brain slice view.
        fig = px.imshow(img_array[j],
                        range_color=range_color,
                        color_continuous_scale='gray')

        # The upper layer is the contour of values between start=50 and end=100,
        # it is designed to be the detected object
        fig.add_trace(go.Contour(z=img_contour[j],
                                 showscale=False,
                                 hoverinfo='skip',
                                 line_width=2,
                                 contours=dict(
                                     start=45,
                                     end=101,
                                     size=25,
                                     coloring='lines',
                                     showlabels=True,
                                     labelfont=dict(size=12, color='white'))))

        fig.update_layout({'title': '{} Slice: {}'.format(subject, j),
                           #    'dragmode': 'drawclosedpath',
                           'width': 580,
                           #    'newshape.line.color': 'cyan',
                           })
        figs_slices.append(fig)
        pass

    logger.debug('The figs_slices is updated.')

    dynamic_data['figs_slices'] = figs_slices

    return fig_contour, figs_slices

# %%


_subjects = [e for e in SUBJECT_MANAGER.subjects]
subject_selector_options = [{'label': e, 'value': e} for e in _subjects]
subject = _subjects[0]

subject_onselect(subject)

# %%
className = 'allow-debug'

_local_style = {
    'display': 'flex',
    'flex-direction': 'row',
    'flex-flow': 'wrap',
    'justify-content': 'space-between',
    'margin-top': '5px',
    'margin-bottom': '10px',
    'margin-left': '10px',
    'margin-right': '10px',
    'background-color': 'white',
}

_local_labelStyle = {'min-width': '100px'}

alff_div_children = [
    # --------------------------------------------------------------------------------
    # Alff subject
    html.Div(
        className=className,
        style={'background-image': 'url("/20.png")',
               'color': 'cornsilk'},
        children=[html.H2('Alff Score')],
    ),
    html.Div(
        id='ALFF-score',
        className='{} {}'.format(className, 'score'),
        children='N.A.'
    ),

    html.Div(
        className=className,
        children=[
            html.Label('????????? ALFF ???'),
            dcc.Input(
                id="ALFF-Area-0",
                type='number',
                placeholder="ALFF of Visual Area",
                value=0,
                required=True,
            ),
            html.Label('??????????????? ALFF ???'),
            dcc.Input(
                id="ALFF-Area-1",
                type='number',
                placeholder="ALFF of Sup-Motion Area",
                value=0,
                required=True,
            ),
        ]
    ),

    html.Div(
        className=className,
        children=[
            # html.Label('ALFF Detail'),
            dcc.Loading(html.Div(
                id='ALFF-table',
                className=className,
                children='Table of ALFF'
            ))
        ]
    )
]

behavior_div_children = [
    # --------------------------------------------------------------------------------
    # Subject information
    html.Div(
        className=className,
        style={'background-image': 'url("/20.png")',
               'color': 'cornsilk'},
        children=[html.H2('Subject Score')],
    ),
    html.Div(
        id='subject-score',
        className='{} {}'.format(className, 'score'),
        children='N.A.'
    ),

    html.Div(
        className=className,
        children=[
            # --------------------------------------------------------------------------------
            html.Label('Age'),
            dcc.RadioItems(
                id='behavior-age',
                className=className,
                style=_local_style,
                labelStyle=_local_labelStyle,
                options=[
                    dict(label='?????? 80', value=2),
                    dict(label='50 ~ 80', value=1),
                    dict(label='?????? 50', value=0),
                ]
            ),

            # --------------------------------------------------------------------------------
            html.Label('Gender'),
            dcc.RadioItems(
                id='behavior-gender',
                className=className,
                style=_local_style,
                labelStyle=_local_labelStyle,
                options=[
                    dict(label='???', value=2),
                    dict(label='???', value=1),
                ]
            ),

            # --------------------------------------------------------------------------------
            html.Label('Habit'),
            dcc.Checklist(
                id='behavior-habit',
                className=className,
                style=_local_style,
                labelStyle=_local_labelStyle,
                options=[
                    dict(label='??????', value=1.00001),
                    dict(label='??????', value=1.00002),
                    dict(label='???', value=0)
                ]
            ),

            # --------------------------------------------------------------------------------
            html.Label('Case'),
            dcc.Checklist(
                id='behavior-case',
                className=className,
                style=_local_style,
                labelStyle=_local_labelStyle,
                options=[
                    dict(label='?????????', value=1.00001),
                    dict(label='?????????', value=1.00002),
                    dict(label='????????????', value=1.00003),
                    dict(label='?????????', value=1.00004),
                    dict(label='???', value=0),
                ]
            ),

            # --------------------------------------------------------------------------------
            html.Label('Medicine'),
            dcc.Checklist(
                id='behavior-medicine',
                className=className,
                style=_local_style,
                labelStyle=_local_labelStyle,
                options=[
                    dict(label='????????????', value=1.00001),
                    dict(label='??????????????????', value=1.00002),
                    dict(label='???', value=0),
                ]
            ),

            # --------------------------------------------------------------------------------
            html.Label('GCS'),
            dcc.RadioItems(
                id='behavior-GCS',
                className=className,
                style=_local_style,
                labelStyle=_local_labelStyle,
                options=[
                    dict(label='3 ~ 6', value=2),
                    dict(label='6 ~ 12', value=1),
                    dict(label='12 ~ 15', value=0),
                ]
            ),

            # --------------------------------------------------------------------------------
            html.Label('NIHSS'),
            dcc.RadioItems(
                id='behavior-NIHSS',
                className=className,
                style=_local_style,
                labelStyle=_local_labelStyle,
                options=[
                    dict(label='?????? 20', value=4),
                    dict(label='14 ~ 20', value=3),
                    dict(label='11 ~ 15', value=2),
                    dict(label='6 ~ 10', value=1),
                    dict(label='0 ~ 5', value=0),
                ]
            ),

            # --------------------------------------------------------------------------------
            html.Label('Volume'),
            dcc.RadioItems(
                id='behavior-volume',
                className=className,
                style=_local_style,
                labelStyle=_local_labelStyle,
                options=[
                    dict(label='?????? 40', value=1),
                    dict(label='?????? 40', value=0),
                ]
            ),

            # --------------------------------------------------------------------------------
            html.Label('Ponding'),
            dcc.RadioItems(
                id='behavior-ponding',
                className=className,
                style=_local_style,
                labelStyle=_local_labelStyle,
                options=[
                    dict(label='???', value=2),
                    dict(label='???', value=0),
                ]
            ),

            # --------------------------------------------------------------------------------
            html.Label('Hemi'),
            dcc.RadioItems(
                id='behavior-hemi',
                className=className,
                style=_local_style,
                labelStyle=_local_labelStyle,
                options=[
                    dict(label='??????', value=2),
                    dict(label='??????', value=1),
                ]
            ),

            # --------------------------------------------------------------------------------
            html.Label('Complication'),
            dcc.Checklist(
                id='behavior-complication',
                className=className,
                style=_local_style,
                labelStyle=_local_labelStyle,
                options=[
                    dict(label='????????????', value=1.00001),
                    dict(label='????????????', value=1.00002),
                    dict(label='???????????????', value=1.00003),
                    dict(label='???????????????', value=1.00004),
                    dict(label='???????????????', value=1.00005),
                    dict(label='???', value=0),
                ]
            ),

        ]
    ),

    html.Br(),
    dcc.Textarea(id='subject-score-detail', value='----'),
]

features_div_children = [
    html.Div(
        className=className,
        style={'background-image': 'url("/20.png")',
               'color': 'cornsilk'},
        children=[html.H2('Features Score')],
    ),
    html.Div(
        id='features-score',
        className='{} {}'.format(className, 'score'),
        children='--'
    ),

    # --------------------------------------------------------------------------------
    # Figures
    html.Div(
        id='figures-div',
        className=className,
        style={'display': 'flex', 'flex-direction': 'row', 'width': '100%'},
        children=[
            html.Div(
                className=className,
                children=[
                    html.H2('Volume View'),
                    html.Div(
                        className=className,
                        children=[html.Label('The 3D view of all candidates')]),
                    dcc.Loading(dcc.Graph(
                        id='graph-1',
                        figure=dynamic_data['fig_contour']
                    ))
                ]
            ),
            html.Div(
                className=className,
                style={'min-width': '400px'},
                children=[
                    html.H2('Slice View'),
                    html.Div(
                        className=className,
                        style={'display': 'flex', 'flex-direction': 'row',
                               'padding-left': '10px'},
                        children=[
                            html.Label('Change the threshold:'),
                            html.Pre(
                                id='annotations-data',
                                style={'display': 'none'},
                                children=['annotations-data']),
                            dcc.Dropdown(
                                id='Threshold-selector',
                                clearable=False,
                                options=mk_threshold_options(),
                                value=50),
                            html.Button(
                                id='Threshold-apply',
                                style={'width': '150px'},
                                children=['Apply']),
                            dcc.Checklist(
                                id='annotation-mode',
                                className=className,
                                options=[
                                    dict(label='Rough', value=1),
                                ]
                            ),
                        ]
                    ),
                    dcc.Loading(dcc.Graph(
                        id='graph-2',
                        figure=dynamic_data['figs_slices'][int(
                            len(dynamic_data['figs_slices']) / 2)]
                    )),
                    dcc.Slider(
                        id='slider-1',
                        min=0,
                        max=len(dynamic_data['figs_slices']),
                        step=1,
                        marks={i: 'Slice {}'.format(i) if i == 0 else str(i)
                               for i in range(0, len(dynamic_data['figs_slices']))},
                        value=int(len(dynamic_data['figs_slices']) / 2),
                    ),
                ]
            ),
        ]
    ),

    # --------------------------------------------------------------------------------
    # Features table
    html.Div(
        id='features-table-div',
        className=className,
        children=[
            html.H2('Features Value'),
            html.Div(
                className=className,
                children=[
                    html.Button(id='Compute-features',
                                children=['Compute Features in 1 - 5 minutes']),
                    dcc.Loading(html.Div(
                        id='features-table',
                        className=className,
                        children='Table of Features'
                    )),
                ]
            )
        ]
    ),

]

app.layout = html.Div(
    id='main-window',
    className=className,
    # children=children_level_1,
    children=[
        html.Div(style={'display': 'none'}, id='blank-output'),
        html.Div(style={'display': 'none'}, id='blank-output-2'),
        html.Div(style={'display': 'none'}, id='blank-output-3'),
        # --------------------------------------------------------------------------------
        # Title
        html.Div(
            className=className,
            style={'display': 'flex', 'flex-direction': 'row',
                   'padding': '10px'},
            children=[
                html.Div(
                    id='app-title-div',
                    className=className,
                    style={'color': 'cornsilk',
                           'background-image': 'url("/10.jpg")',
                           'background-size': 'cover'},
                    children=[
                        html.H1(id='app-title',
                                children=CONFIG['app_name']),
                        html.Div(
                            style={'display': 'flex',
                                   'flex-direction': 'row', 'padding': '5px'},
                            children=[
                                html.Div(
                                    className=className,
                                    style={'background-image': 'url("/logo.jpg")',
                                           'height': '80px',
                                           'width': '80px',
                                           'color': 'cornsilk',
                                           'background-size': 'cover'},
                                ),
                                html.Div(
                                    className=className,
                                    style={'background-image': 'url("/logo-2.jpg")',
                                           'height': '80px',
                                           'width': '80px',
                                           'color': 'cornsilk',
                                           'background-size': 'cover'}
                                ), ]
                        )
                    ]
                ),
                dcc.Markdown(CONFIG['acknowledge'], style={
                             'margin-left': '20px',
                             'max-height': '180px',
                             'overflow-y': 'scroll'}),
            ]
        ),
        # --------------------------------------------------------------------------------
        # Subject selector
        html.Div(
            id='subject-selector-div',
            className=className,
            children=[
                html.H2('Subject Selector'),
                dcc.Dropdown(
                    id='CT-Subject-selector',
                    clearable=False,
                    options=subject_selector_options,
                    value=subject
                ),
                html.Div(
                    className=className,
                    style={'display': 'flex', 'flex-direction': 'row'},
                    children=[
                        html.Label('Slice Distance (mm)'),
                        dcc.Input(
                            id="Slice_distance",
                            type='number',
                            placeholder="Distance between slices",
                            min=1,
                            max=20,
                            value=dynamic_data['slice_distance'],
                            required=True,
                        ),
                        html.Label('Pixel Resolution (mm)'),
                        dcc.Input(
                            id="Pixel_resolution",
                            type='number',
                            placeholder="Pixel resolution",
                            min=0.1,
                            max=2.0,
                            value=dynamic_data['pixel_resolution'],
                            required=True,
                        ),
                        html.Label('Other Issues'),
                        html.Textarea(
                            id='CT-Subject-textarea',
                            children='[Place Holder]'
                        )
                    ]
                ),
            ]
        ),

        html.Div(
            id='three-column-div',
            className=className,
            style={'display': 'flex', 'flex-direction': 'row'},
            children=[
                html.Div(
                    id='features-div',
                    className=className,
                    children=features_div_children
                ),
                html.Div(
                    id='behavior-div',
                    className=className,
                    children=behavior_div_children
                ),
                html.Div(
                    id='alff-div',
                    className=className,
                    children=alff_div_children
                )
            ]
        )

    ]
)


# %%


# Display the #graph-2 if #slider-1 is changed
app.clientside_callback(
    """
    (e) => {
        console.log(e);
        let dom = document.getElementById('graph-2')
        dom.style.display = ''

        return ['a', 'b']
    }
    """,
    [
        Output('blank-output-2', 'children')
    ],
    [
        # Input('slider-1', 'value'),
        Input('Threshold-selector', 'value'),
        Input('Threshold-apply', 'n_clicks')
    ],
)

# Reset all things on subject selection
app.clientside_callback(
    """
    (e) => {
        console.log(e);

        let dom = document.getElementById('subject-score')
        dom.textContent = 'N.A.'

        dom = document.getElementById('features-score')
        dom.textContent = 'N.A.'

        dom = document.getElementById('ALFF-score')
        dom.textContent = 'N.A.'

        dom = document.getElementById('graph-2')
        dom.style.display = 'none'

        let behaviors = [
            'behavior-age',
            'behavior-gender',
            'behavior-habit',
            'behavior-case',
            'behavior-medicine',
            'behavior-GCS',
            'behavior-NIHSS',
            'behavior-volume',
            'behavior-ponding',
            'behavior-hemi',
            'behavior-complication'
            ]

        for (let b=0; b < behaviors.length; b++) {
            dom = document.getElementById(behaviors[b]);
            for (let i = 0; i < dom.childElementCount; i++) {
                dom.children[i].children[0].checked = false;
            }
        }

        return ['a', 'b']
    }
    """,
    [
        Output('blank-output-1', 'children')
    ],
    [
        Input('CT-Subject-selector', 'value')
    ],
    prevent_initialization=True,
)


@app.callback(
    [
        Output('ALFF-score', 'children'),
        Output('ALFF-table', 'children')
    ],
    [
        Input('ALFF-Area-0', 'value'),
        Input('ALFF-Area-1', 'value'),
    ]
)
def callback_ALFF_on_change(value0, value1):
    # --------------------------------------------------------------------------------
    # Which Input is inputted
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    logger.debug(
        'The callback_ALFF_on_change receives the event: {}'.format(cbcontext))

    table_obj, score = mk_ALFF_table(value0, value1)

    return '{:0.2f}'.format(score), table_obj


@app.callback(
    [
        Output('blank-output-3', 'children')
    ],
    [
        Input('Slice_distance', 'value'),
        Input('Pixel_resolution', 'value')
    ],
)
def callback_geometry_on_change(slice_distance, pixel_resolution):
    # --------------------------------------------------------------------------------
    # Which Input is inputted
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    logger.debug(
        'The callback_geometry_on_change receives the event: {}'.format(cbcontext))

    dynamic_data['slice_distance'] = slice_distance
    dynamic_data['pixel_resolution'] = pixel_resolution
    logger.debug('Geometry of the CT image is changed {}, {}'.format(
        slice_distance, pixel_resolution))

    return '{}, {}'.format(slice_distance, pixel_resolution),


@app.callback(
    [
        Output('features-table', 'children'),
        Output('features-score', 'children'),
    ],
    [
        Input('Compute-features', 'n_clicks')
    ],
    prevent_initialization=True,
)
def callback_compute_features(n_clicks):
    # --------------------------------------------------------------------------------
    # Which Input is inputted
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    logger.debug(
        'The callback_compute_features receives the event: {}, {}'.format(cbcontext, n_clicks))

    if not n_clicks:
        logger.warning('The n_clicks parameter is required')
        return 'Not Computed Yet', 'N.A.'

    subject = dynamic_data['subject']

    mk_features_table(subject)

    table_obj = dynamic_data['table_obj']
    score = dynamic_data['score']

    if isinstance(score, float):
        score = '{:0.2f}'.format(score)

    logger.debug('Update the table and score {}'.format(score))

    return table_obj, score


@app.callback(
    [
        # Output('features-table', 'children'),
        # Output('features-score', 'children'),
        Output('graph-1', 'figure'),
        Output('slider-1', 'marks'),
        Output('slider-1', 'min'),
        Output('slider-1', 'max'),
        Output('slider-1', 'value'),
        Output('CT-Subject-textarea', 'children')
        # Output('graph-2', 'figure'),
    ],
    [
        Input('CT-Subject-selector', 'value')
    ]
)
def callback_subject_selection(subject):
    '''
    Update the features table.
    '''
    # --------------------------------------------------------------------------------
    # Which Input is inputted
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    logger.debug(
        'The callback_subject_selection receives the event: {}'.format(cbcontext))

    subject_onselect(subject)

    fig_contour = dynamic_data['fig_contour']
    figs_slices = dynamic_data['figs_slices']

    num = len(figs_slices)
    marks = {i: 'Slice {}'.format(i) if i == 0 else str(i)
             for i in range(0, num, 5)}
    _min = 0
    _max = num
    _slice = int(num / 2)

    # Output('features-table', 'children'),
    # Output('features-score', 'children'),
    # Output('graph-1', 'figure'),
    # Output('slider-1', 'marks'),
    # Output('slider-1', 'min'),
    # Output('slider-1', 'max'),
    # Output('slider-1', 'value'),
    # Output('graph-2', 'figure'),

    output = (
        fig_contour,
        marks,
        _min,
        _max,
        _slice,
        '[Place Holder]',
        # figs_slices[_slice],
    )

    return output


@app.callback(
    Output("annotations-data", "children"),
    Input("graph-2", "relayoutData"),
    [
        Input('slider-1', 'value'),
        Input('annotation-mode', 'value'),
    ],
    prevent_initial_call=True,
)
def callback_on_new_annotation(relayout_data, slice_idx, value):
    # --------------------------------------------------------------------------------
    # Which Input is inputted
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    logger.debug(
        'The callback_on_new_annotation receives the event: {}'.format(cbcontext))

    logger.debug('Annotating the slice of {}'.format(slice_idx))

    if value is None:
        return dash.no_update

    if len(value) == 0:
        value.append(None)

    slice_idx = int(slice_idx)

    if cbcontext.startswith('slider-1'):
        return dash.no_update

    if cbcontext.startswith('annotation-mode'):
        return dash.no_update

    # [
    #   {
    #     "editable": true,
    #     "xref": "x",
    #     "yref": "y",
    #     "layer": "above",
    #     "opacity": 1,
    #     "line": {
    #       "color": "cyan",
    #       "width": 1,
    #       "dash": "solid"
    #     },
    #     "fillcolor": "rgba(0,0,0,0)",
    #     "fillrule": "evenodd",
    #     "type": "rect",
    #     "x0": 231.55161290322582,
    #     "y0": 330.958064516129,
    #     "x1": 343.86129032258066,
    #     "y1": 185.61612903225807
    #   }
    # ]
    if relayout_data is None:
        return dash.no_update

    if "shapes" in relayout_data:
        obj = relayout_data["shapes"]

        for dct in obj:
            x0 = int(np.min([dct['x0'], dct['x1']]))
            x1 = int(np.max([dct['x0'], dct['x1']]))
            y0 = int(np.min([dct['y0'], dct['y1']]))
            y1 = int(np.max([dct['y0'], dct['y1']]))
            # x1 = dct['x1']
            # y0 = dct['y0']
            # y1 = dct['y1']

            if value[0] == 1:
                logger.debug(
                    'Cutting values of {} in rough mode'.format((x0, x1, y0, y1)))
                dynamic_data['img_contour'][:, 0:y0] = 500
                dynamic_data['img_contour'][:, y1:-1] = 500
                dynamic_data['img_contour'][:, :, 0:x0] = 500
                dynamic_data['img_contour'][:, :, x1:-1] = 500

            else:
                logger.debug('Cutting values of {} on layer {}'.format((x0, x1, y0, y1),
                                                                       slice_idx))
                dynamic_data['img_contour'][slice_idx, y0:y1, x0:x1] = 300

        # Consume all the cuts
        # relayout_data['shapes'] = []
        return obj
    else:
        return dash.no_update


@app.callback(
    [
        Output('graph-2', 'figure'),
    ],
    [
        Input('slider-1', 'value'),
        Input('Threshold-selector', 'value'),
        Input('Threshold-apply', 'n_clicks'),
        Input('annotation-mode', 'value'),
    ],
)
def callback_slice_by_slider_1(slice, threshold, n_clicks, value):
    # --------------------------------------------------------------------------------
    # Which Input is inputted
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    logger.debug(
        'The callback_slider_1 receives the event: {}'.format(cbcontext))

    if cbcontext.startswith('Threshold-apply'):
        slice_view_recompute(threshold, use_dynamic_data=True)

    if value is None:
        value = [None]
    else:
        if len(value) == 0:
            value.append(None)

    if cbcontext.startswith('annotation-mode') and value[0] == 1:
        slice_view_recompute(threshold, use_dynamic_data=False)

    if cbcontext.startswith('annotation-mode') and value[0] != 1:
        slice_view_recompute(threshold, use_dynamic_data=True)

    if cbcontext.startswith('Threshold-selector'):
        slice_view_recompute(threshold, use_dynamic_data=False)

    if 'figs_slices' in dynamic_data:
        fig = dynamic_data['figs_slices'][int(slice)]

        img_contour = dynamic_data['img_contour'].copy()

        if value[0] == 1:
            img_slice = np.sum(img_contour, axis=0)
            # img_slice[img_slice > 300] = 0
            fig.add_trace(go.Contour(z=img_slice,
                                     showscale=False,
                                     hoverinfo='skip',
                                     line_width=2,
                                     contours=dict(
                                         start=25,
                                         end=101,
                                         size=25,
                                         coloring='lines',
                                         showlabels=False,
                                         labelfont=dict(size=12, color='white'))))

        else:
            img_slice = img_contour[int(slice)].copy()
            img_slice[img_slice < 300] = 0
            fig.add_trace(go.Contour(z=img_slice,
                                     showscale=False,
                                     hoverinfo='skip',
                                     line_width=2,
                                     contours=dict(
                                         start=25,
                                         end=101,
                                         size=25,
                                         coloring='lines',
                                         showlabels=False,
                                         labelfont=dict(size=12, color='white'))))

    else:
        fig = px.scatter([1, 2, 3])

    return fig,


@app.callback(
    [
        Output('subject-score-detail', 'value'),
        Output('subject-score', 'children'),
    ],
    [
        Input('behavior-age', 'value'),
        Input('behavior-gender', 'value'),
        Input('behavior-habit', 'value'),
        Input('behavior-case', 'value'),
        Input('behavior-medicine', 'value'),
        Input('behavior-GCS', 'value'),
        Input('behavior-NIHSS', 'value'),
        Input('behavior-volume', 'value'),
        Input('behavior-ponding', 'value'),
        Input('behavior-hemi', 'value'),
        Input('behavior-complication', 'value'),
    ],
)
def callback_behaviors_1(age, gender, habit, case, medicine, GCS, NIHSS, volume, ponding, hemi, complication):
    # --------------------------------------------------------------------------------
    # Which Input is inputted
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    logger.debug(
        'The callback_behaviors_1 receives the event: {}'.format(cbcontext))

    dct = dict(
        age=age,
        gender=gender,
        habit=habit,
        case=case,
        medicine=medicine,
        GCS=GCS,
        NIHSS=NIHSS,
        volume=volume,
        ponding=ponding,
        hemi=hemi,
        complication=complication
    )

    logger.debug('The behavior values are: {}'.format(dct))

    res = ['']
    score = 0
    for key, value in dct.items():
        res.append('    {}: {}'.format(key, value))

        if value is None:
            continue

        if isinstance(value, list):
            for v in value:
                score += v
        else:
            score += value

    score = int(score)

    res[0] = 'Score: {}'.format(score)

    res = '\n'.join(res)

    return res, score


# Add a static image route that serves images from desktop
# Be *very* careful here - you don't want to serve arbitrary files
# from your computer or server


@app.server.route('/<asset_name>')
def serve_image(asset_name):
    image_directory = Path(__file__).joinpath('../assets')
    if asset_name not in [e.name for e in image_directory.iterdir() if e.is_file()]:
        logger.error('Can not find asset: {}'.format(asset_name))
        raise Exception('Can not find asset: {}'.format(asset_name))
    logger.debug('Got asset {}'.format(asset_name))
    return flask.send_from_directory(image_directory, asset_name)


# %%
if __name__ == '__main__':
    port = 8693
    logger.info('Server is estimated in {}:{}'.format(
        'http://localhost', port))

    t = threading.Thread(target=run_simple, args=(
        'localhost', port, app.server))
    t.setDaemon(True)
    t.start()

    gui.mainloop()

    print('Bye Bye.')

# %%
