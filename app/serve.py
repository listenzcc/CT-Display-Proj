# %%
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

from onstart import CONFIG, logger

from load_templates import WEIGHTS, RANGE_TABLE, OFFSET
from load_subjects import SUBJECT_MANAGER

# %%
external_stylesheets = [
    CONFIG['assets_folder'].joinpath('debug-style.css').as_posix(),
    CONFIG['assets_folder'].joinpath('basic-style.css').as_posix(),
]

app = dash.Dash(CONFIG['app_name'], external_stylesheets=external_stylesheets)

# %%
dynamic_data = dict()

# %%
className = 'allow-debug'

_local_style = {
    'display': 'flex',
    'flex-direction': 'row',
    'flex-flow': 'wrap',
    'justify-content': 'space-between',
}

_local_labelStyle = {'min-width': '100px'}

behavior_div_children = [
    # --------------------------------------------------------------------------------
    # Subject information
    html.Div(
        id='subject-information-div',
        className=className,
        children=[
            html.Label('Subject Information'),
            html.Br(),
            dcc.Textarea(id='Subject-Information', value='----')
        ]
    ),

    # --------------------------------------------------------------------------------
    html.Label('Age'),
    dcc.RadioItems(
        id='behavior-age',
        className=className,
        style=_local_style,
        labelStyle=_local_labelStyle,
        options=[
            dict(label='高于 80', value=2),
            dict(label='50 ~ 80', value=1),
            dict(label='低于 50', value=0),
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
            dict(label='男', value=2),
            dict(label='女', value=1),
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
            dict(label='吸烟', value=1.00001),
            dict(label='饮酒', value=1.00002),
            dict(label='无', value=0)
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
            dict(label='高血压', value=1.00001),
            dict(label='糖尿病', value=1.00002),
            dict(label='高脂血症', value=1.00003),
            dict(label='脑卒中', value=1.00004),
            dict(label='无', value=0),
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
            dict(label='抗凝药物', value=1.00001),
            dict(label='抗血小板药物', value=1.00002),
            dict(label='无', value=0),
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
            dict(label='1 ~ 12', value=1),
            dict(label='12 ~ 6???', value=0),
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
            dict(label='大于 20', value=4),
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
            dict(label='大于 40', value=1),
            dict(label='小于 40', value=0),
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
            dict(label='是', value=2),
            dict(label='否', value=0),
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
            dict(label='左侧', value=2),
            dict(label='右侧', value=1),
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
            dict(label='再发出血', value=1.00001),
            dict(label='颅内感染', value=1.00002),
            dict(label='呼吸道感染', value=1.00003),
            dict(label='泌尿系感染', value=1.00004),
            dict(label='深静脉血栓', value=1.00005),
            dict(label='无', value=0),
        ]
    ),
]

features_div_children = [

    # --------------------------------------------------------------------------------
    # Features table
    dcc.Loading(
        html.Div(
            id='features-table-div',
            className=className,
            children=[
                html.Label('Score: '),
                html.Div(
                    id='features-score',
                    className=className,
                    children='--'
                ),
                html.Div(
                    className=className,
                    children=[
                        html.Div(
                            id='features-table',
                            className=className,
                            children='Table of Features'
                        )
                    ]
                )
            ]
        )
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
                    html.Div(
                        id='slide-2-container-div',
                        className=className,
                        children=dcc.Graph(
                            id='graph-2',
                        )
                    ),
                    dcc.Slider(
                        id='slider-1',
                        min=0,
                        max=10,
                        marks={i: 'Slice {}'.format(i) if i == 0 else str(i)
                               for i in range(0, 10)},
                        value=5,
                    ),
                ]
            ),
            dcc.Loading(html.Div(
                className=className,
                children=[
                    dcc.Graph(
                        id='graph-1',
                    )
                ]
            )),
        ]
    )
]

app.layout = html.Div(
    id='main-window',
    className=className,
    # children=children_level_1,
    children=[
        # --------------------------------------------------------------------------------
        # Title
        html.Div(
            id='app-title-div',
            className=className,
            children=[
                html.H1(id='app-title',
                        children='No Title')
            ]
        ),

        # --------------------------------------------------------------------------------
        # Subject selector
        html.Div(
            id='subject-selector-div',
            className=className,
            children=[
                html.Label('Subject'),
                dcc.Dropdown(
                    id='CT-Subject-selector',
                    clearable=False,
                    options=[{'label': e, 'value': e}
                             for e in SUBJECT_MANAGER.subjects],
                    value=[e for e in SUBJECT_MANAGER.subjects][1]
                )
            ]
        ),

        html.Div(
            id='two-column-div',
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
                )
            ]
        )

    ]
)

# %%


def mk_features_table(subject):
    # Try get_features only use the subject,
    # Or will try to compute the features.
    try:
        df = SUBJECT_MANAGER.get_features(subject)
    except AssertionError:
        img_array = SUBJECT_MANAGER.get_array(subject)
        img_contour = SUBJECT_MANAGER.compute_contour(img_array)
        df = SUBJECT_MANAGER.get_features(subject, img_array, img_contour)

    # N.A. refers we found no ROI,
    # so we do nothing for it.
    score = np.inf

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

    columns = [{"name": i, "id": i} for i in df.columns]
    data = df.to_dict('records')

    table_obj = dash_table.DataTable(
        columns=columns,
        data=data
    )

    return table_obj, score


def mk_figures(subject):
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

    logger.debug('The fig of 3D is updated.')

    # --------------------------------------------------------------------------------
    # The figs is a list of slice views
    figs_slices = []
    range_color = (-1000, 2000)
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
                                     start=50,
                                     end=100,
                                     size=25,
                                     coloring='lines',
                                     showlabels=True,
                                     labelfont=dict(size=12, color='white'))))

        fig.update_layout({'title': '{} Slice: {}'.format(subject, j),
                           'dragmode': 'drawclosedpath',
                           'width': 580,
                           'newshape.line.color': 'cyan'})
        figs_slices.append(fig)
        pass

    logger.debug('The figs_slices is updated.')

    dynamic_data['figs_slices'] = figs_slices
    logger.debug('The dynamic_data is updated: {}.'.format(
        [e for e in dynamic_data]))

    return fig_contour, figs_slices

# %%


@app.callback(
    [
        Output('features-table', 'children'),
        Output('features-score', 'children')
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

    table_obj, score = mk_features_table(subject)

    return table_obj, score


@app.callback(
    [
        Output('graph-1', 'figure'),
        Output('slider-1', 'marks'),
        Output('slider-1', 'min'),
        Output('slider-1', 'max'),
        Output('slider-1', 'value'),
        Output('graph-2', 'figure'),
    ],
    [
        Input('CT-Subject-selector', 'value')
    ]
)
def callback_subject_selection_1(subject):
    '''
    Update the figures.
    '''
    # --------------------------------------------------------------------------------
    # Which Input is inputted
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    logger.debug(
        'The callback_subject_selection_1 receives the event: {}'.format(cbcontext))

    fig, figs_slices = mk_figures(subject)

    num = len(figs_slices)
    marks = {i: 'Slice {}'.format(i) if i == 0 else str(i)
             for i in range(0, num)}
    _min = 0
    _max = num
    slice = int(num / 2)

    return fig, marks, _min, _max, slice, figs_slices[slice]


@app.callback(
    [
        Output('graph-2', 'figure')
    ],
    [
        Input('slider-1', 'value')
    ]
)
def callback_slider_1(slice_idx):
    '''
    Change Slice.
    '''
    # --------------------------------------------------------------------------------
    # Which Input is inputted
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    logger.debug(
        'The callback_slider_1 receives the event: {}'.format(cbcontext))

    figs = dynamic_data.get('figs_slices', None)

    if figs is None:
        return px.scatter(x=[1, 2, 3], y=[4, 5, 6]),

    fig = figs[slice_idx]

    return fig,


@app.callback(
    [
        Output('Subject-Information', 'value'),
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
    ]
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

    return res,


# %%
if __name__ == '__main__':
    logger.info('Server is estimated in {}'.format('http://127.0.0.1:8050'))
    app.run_server(debug=True)

# %%
