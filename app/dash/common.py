from datetime import date
from dash.dependencies import Input, Output
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px

from .dash import Dash
from .utils import read_claims


claims = read_claims(path="app/data/claims_sample_data__cleaned.csv")
available_payers = claims.sort_values("PAID_AMOUNT")["PAYER"].unique()
available_serv_categories = claims.sort_values("PAID_AMOUNT")["SERVICE_CATEGORY"].unique()


app_layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("EDA"),
                        html.H5("Select Payer and Category to analyse"),
                    ],
                    width=True,
                ),
            ],
            align="end",
        ),
        html.Hr(),
        html.Div(
            className="row", children=[
                html.Div(className='date-values', children=[
                    dcc.DatePickerRange(
                        id='date-value',
                        initial_visible_month=date(2018, 1, 1),
                        end_date=date(2020, 7, 31)
                    )
                ], style=dict(width='24%', height="50%", marginBottom=10)),
                html.Div(className='payer-values', children=[
                    dcc.Dropdown(
                        id="payer-value",
                        options=[{"label": col, "value": col} for col in available_payers],
                        value=None,
                        clearable=False,
                        searchable=False,
                        multi=True,
                    )], style=dict(width='38%', height="50%", marginBottom=10)),
                html.Div(className='category-values', children=[
                    dcc.Dropdown(
                        id="category-value",
                        options=[{"label": col, "value": col} for col in available_serv_categories],
                        value=None,
                        clearable=False,
                        searchable=False,
                        multi=True,
                    )], style=dict(width='38%', height="50%", marginBottom=10))
            ], style=dict(display='flex')
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="payer-pie"), md=5),
                dbc.Col(dcc.Graph(id="category-pie"), md=5),
            ],
            align="center",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="payer-area"), md=5),
                dbc.Col(dcc.Graph(id="category-area"), md=5),
            ],
            align="center",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="payer-area-stacked"), md=5),
                dbc.Col(dcc.Graph(id="category-area-stacked"), md=5),
            ],
            align="center",
        ),
    ],
    fluid=True,
)


def init_dash(server):
    dash_app = Dash(server=server, routes_pathname_prefix="/common/",)
    dash_app.layout = app_layout
    init_callbacks(dash_app)
    return dash_app.server


def init_callbacks(dash_app):
    dash_app.callback(
        Output("payer-pie", "figure"),
        [
            Input("payer-value", "value"),
            Input("category-value", "value"),
            Input("date-value", "start_date"),
            Input("date-value", "end_date"),
        ],
    )(update_payer_pie)

    dash_app.callback(
        Output("category-pie", "figure"),
        [
            Input("payer-value", "value"),
            Input("category-value", "value"),
            Input("date-value", "start_date"),
            Input("date-value", "end_date"),
        ],
    )(update_category_pie)

    dash_app.callback(
        Output('payer-value', 'options'),
        [
            Input("category-value", "value"),
            Input("date-value", "start_date"),
            Input("date-value", "end_date"),
        ],
    )(update_payer_dropdown)

    dash_app.callback(
        Output('category-value', 'options'),
        [
            Input("payer-value", "value"),
            Input("date-value", "start_date"),
            Input("date-value", "end_date"),
        ],
    )(update_category_dropdown)

    dash_app.callback(
        Output("payer-area", "figure"),
        [
            Input("payer-value", "value"),
            Input("category-value", "value"),
            Input("date-value", "start_date"),
            Input("date-value", "end_date"),
        ],
    )(update_payer_area)

    dash_app.callback(
        Output("category-area", "figure"),
        [
            Input("payer-value", "value"),
            Input("category-value", "value"),
            Input("date-value", "start_date"),
            Input("date-value", "end_date"),
        ],
    )(update_category_area)

    dash_app.callback(
        Output("category-area-stacked", "figure"),
        [
            Input("payer-value", "value"),
            Input("category-value", "value"),
            Input("date-value", "start_date"),
            Input("date-value", "end_date"),
        ],
    )(update_category_area_stacked)

    dash_app.callback(
        Output("payer-area-stacked", "figure"),
        [
            Input("payer-value", "value"),
            Input("category-value", "value"),
            Input("date-value", "start_date"),
            Input("date-value", "end_date"),
        ],
    )(update_payer_area_stacked)

    return dash_app


def update_payer_pie(payer_value, category_value, start_date, end_date):

    filtered_data = filter_claims(payer_value, category_value, start_date, end_date)

    payer_data = (
        filtered_data
        .groupby("PAYER")["PAID_AMOUNT"]
        .sum()
        .reset_index()
    )

    data = go.Pie(
        labels=payer_data.PAYER,
        values=payer_data.PAID_AMOUNT,
        hole=.3)
    
    layout = go.Layout(
        margin={"l": 80, "b": 60, "t": 60, "r": 20},
        height=450,
        title="By Payer",
        hovermode="closest",
    )
    
    payer_pie = go.Figure(data=[data], layout=layout)

    return payer_pie


def filter_claims(payer_value=None, category_value=None, start_date=None, end_date=None):
    filtered_data = (
        claims[claims["PAYER"].isin(payer_value)]
        if payer_value else claims
    )
    filtered_data = (
        filtered_data[filtered_data["SERVICE_CATEGORY"].isin(category_value)]
        if category_value else filtered_data
    )
    if start_date:
        filtered_data = filtered_data[filtered_data["MONTH_DATE"] >= date.fromisoformat(start_date)]
    if end_date:
        filtered_data = filtered_data[filtered_data["MONTH_DATE"] <= date.fromisoformat(end_date)]
    return filtered_data


def update_category_pie(payer_value, category_value, start_date, end_date):

    filtered_data = filter_claims(payer_value, category_value, start_date, end_date)

    cat_data = (
        filtered_data
        .groupby("SERVICE_CATEGORY")["PAID_AMOUNT"]
        .sum()
        .reset_index()
    )

    data = go.Pie(
        labels=cat_data.SERVICE_CATEGORY,
        values=cat_data.PAID_AMOUNT,
        hole=.3)
    
    layout = go.Layout(
        margin={"l": 80, "b": 60, "t": 60, "r": 20},
        height=450,
        title="By Service Category",
        hovermode="closest",
    )
    
    cat_pie = go.Figure(data=[data], layout=layout)

    return cat_pie


def update_payer_dropdown(category_value, start_date, end_date):
    filtered_data = filter_claims(
            category_value=category_value,
            start_date=start_date,
            end_date=end_date
        )
    if filtered_data.empty:
        return [{'label': None, 'value': None}]
    return [
        {'label': payer, 'value': payer}
        for payer in filtered_data.sort_values("PAID_AMOUNT")["PAYER"].unique()
    ]


def update_category_dropdown(payer_value, start_date, end_date):
    filtered_data = filter_claims(
        payer_value=payer_value,
        start_date=start_date,
        end_date=end_date
    )
    if filtered_data.empty:
        return [{'label': None, 'value': None}]
    return [
        {'label': cat, 'value': cat}
        for cat in filtered_data.sort_values("PAID_AMOUNT")["SERVICE_CATEGORY"].unique()
    ]


def update_payer_area(payer_value, category_value, start_date, end_date):
    filtered_data = filter_claims(payer_value, category_value, start_date, end_date)

    plot_data = (
        filtered_data
        .groupby(["MONTH_DATE", "PAYER"])["PAID_AMOUNT"]
        .sum()
        .reset_index()
    )

    fig = px.area(
        plot_data,
        x="MONTH_DATE",
        y="PAID_AMOUNT",
        color="PAYER"
    )
    fig.update_layout(
        width=640,
        height=450,
        yaxis_title="Paid Amount",
        xaxis_title="Month"
    )
    return fig


def update_category_area(payer_value, category_value, start_date, end_date):

    filtered_data = filter_claims(payer_value, category_value, start_date, end_date)

    plot_data = (
        filtered_data
        .groupby(["MONTH_DATE", "SERVICE_CATEGORY"])["PAID_AMOUNT"]
        .sum()
        .reset_index()
    )

    fig = px.area(
        plot_data,
        x="MONTH_DATE",
        y="PAID_AMOUNT",
        color="SERVICE_CATEGORY"
    )
    fig.update_layout(
        width=640,
        height=450,
        margin={"l": 80, "b": 60, "t": 60, "r": 20},
        yaxis_title="Paid Amount",
        xaxis_title="Month"
    )
    return fig


def update_payer_area_stacked(payer_value, category_value, start_date, end_date):

    filtered_data = filter_claims(payer_value, category_value, start_date, end_date)

    plot_data = filtered_data.pivot_table(
        index="MONTH_DATE",
        values="PAID_AMOUNT",
        columns="PAYER",
        aggfunc="sum"
    )

    fig = go.Figure()

    for col in plot_data.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data[col],
                name=col,
                stackgroup='one',
                groupnorm='percent'
            )
        )
    fig.update_layout(
        width=640,
        height=450,
        margin={"l": 80, "b": 60, "t": 60, "r": 20},
        yaxis_title="Paid Amount Percent",
        xaxis_title="Month"
    )
    return fig


def update_category_area_stacked(payer_value, category_value, start_date, end_date):

    filtered_data = filter_claims(payer_value, category_value, start_date, end_date)

    plot_data = filtered_data.pivot_table(
        index="MONTH_DATE",
        values="PAID_AMOUNT",
        columns="SERVICE_CATEGORY",
        aggfunc="sum"
    )

    fig = go.Figure()

    for col in plot_data.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data[col],
                name=col,
                stackgroup='one',
                groupnorm='percent'
            )
        )
    fig.update_layout(
        width=640,
        height=450,
        margin={"l": 80, "b": 60, "t": 60, "r": 20},
        yaxis_title="Paid Amount Percent",
        xaxis_title="Month"
    )
    return fig