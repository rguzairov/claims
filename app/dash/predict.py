from datetime import date
from dash.dependencies import Input, Output
from dash import dcc
import dash_daq as daq
from dash import html
import dash_bootstrap_components as dbc
from dash import dash_table
from dateutil.relativedelta import relativedelta
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing   
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA

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
                        html.H2("Timeseries predictions."),
                        html.H5("1. Select Payer and Category to predict."),
                        html.H5("2.Select upcoming N months"),
                        html.H5("3. Select model.")
                    ],
                    width=True,
                ),
            ],
            align="end",
        ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H5("Filters"),
                                html.Div(className='date-values', children=[
                                    html.P("Date range:"),
                                    dcc.DatePickerRange(
                                        id='date-value',
                                        initial_visible_month=date(2018, 1, 1),
                                        end_date=date(2020, 7, 31)
                                    )
                                ], style={'marginBottom': 10, 'marginTop': 20}),
                                html.Div(className='payer-values', children=[
                                    html.P("Payer:"),
                                    dcc.Dropdown(
                                        id="payer-value",
                                        options=[{"label": col, "value": col} for col in available_payers],
                                        value=None,
                                        clearable=False,
                                        searchable=False,
                                        multi=True,
                                    )], style={'marginBottom': 10, 'marginTop': 10}),
                                html.Div(className='category-values', children=[
                                    html.P("Category:"),
                                    dcc.Dropdown(
                                        id="category-value",
                                        options=[{"label": col, "value": col} for col in available_serv_categories],
                                        value=None,
                                        clearable=False,
                                        searchable=False,
                                        multi=True,
                                    )], style={'marginBottom': 10, 'marginTop': 10}),
                            ]
                        ),
                        html.Hr(),
                        html.Div(
                            [
                                html.H5("Models"),
                                html.Div(children=[
                                    html.P("Steps:"),
                                    dcc.Input(id="n-steps", value=6.0, type="number")],
                                    style={'marginBottom': 10, 'marginTop': 10}
                                ),
                                html.Div(className='', children=[
                                    html.P("Models:"),
                                    dcc.Dropdown(
                                        id="models",
                                        options=[
                                            {"label": "Median", "value": "median"},
                                            {"label": "Holtwinters", "value": "holtwinters"},
                                            {"label": "ARIMA", "value": "ARIMA"},
                                        ],
                                        value="median",
                                        clearable=False,
                                        searchable=False,
                                        multi=True,
                                    )], style={'marginBottom': 10, 'marginTop': 10}
                                ),
                                html.Div([
                                    html.P("With fitted data:"),
                                    daq.BooleanSwitch(id="with-fitted", on=False, color="blue"),
                                    ]
                                ),
                            ]
                        ),
                        html.Hr(),
                        html.Div(
                            [
                                html.H5("Parameters"),
                                html.P("Holtwinters model"),
                            ], style={'marginBottom': 10, 'marginTop': 10}
                        ),
                    ],
                    width=2,
                ),
                dbc.Col(
                    [
                        dbc.Spinner(
                            dcc.Graph(id="predict", style={"height": "80vh"}),
                            color="primary",
                        )
                    ], width=True,
                ),
                dbc.Col(
                    [
                        dash_table.DataTable(
                            id="metrics",
                            data=[],
                            columns=[
                                {"id": "Predict", "name": "Predict"},
                                {"id": "MAPE", "name": "MAPE"}],
                        )
                    ], width=True, style={'marginBottom': 10, 'marginTop': 60}
                ),
            ]
        ),
    ],
    fluid=True,
)


def init_dash(server):
    dash_app = Dash(server=server, routes_pathname_prefix="/predict/",)
    dash_app.layout = app_layout
    init_callbacks(dash_app)
    return dash_app.server


def init_callbacks(dash_app):
    dash_app.callback([
            Output("predict", "figure"),
            Output("metrics", "data"),
        ],
        [
            Input("payer-value", "value"),
            Input("category-value", "value"),
            Input("date-value", "start_date"),
            Input("date-value", "end_date"),
            Input("n-steps", "value"),
            Input("models", "value"),
            Input("with-fitted", "on"),
        ],
    )(update_predict)

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

    return dash_app


def update_predict(payer_value, category_value, start_date, end_date, n_steps, models, with_fitted):

    filtered_data = filter_claims(payer_value, category_value, start_date, end_date)
    prepared_data = prepare_data(filtered_data, n_steps)
    if "median" in models:
        prepared_data = median_predict(prepared_data, n_steps, with_fitted)
    if "holtwinters" in models:
        prepared_data = holtwinters_predict(prepared_data, n_steps, with_fitted)
    if "ARIMA" in models:
        prepared_data = arima_predict(prepared_data, n_steps)

    real_data = filter_claims(
        payer_value,
        category_value,
        start_date=date.fromisoformat(end_date).replace(day=1) + relativedelta(months=1)
    )
    real_data = prepare_data(real_data, n_steps=0)
    
    plot_data = (
        prepared_data
        .merge(
            (
                real_data[["PAID_AMOUNT", "MONTH_DATE"]]
                .rename(columns={"PAID_AMOUNT": "PAID_AMOUNT__REAL"})
            ),
            on="MONTH_DATE",
            how="left"
        )
        .set_index("MONTH_DATE")
    )

    metrics = count_mape(plot_data)

    fig = go.Figure()

    for col in plot_data.columns:
        fig.add_trace(
            go.Scatter(
                x=plot_data.index,
                y=plot_data[col],
                name=col
            )
        )
    fig.update_layout(
        width=900,
        height=600,
        margin={"l": 10, "b": 60, "t": 60, "r": 10},
        yaxis_title="Paid Amount",
        xaxis_title="Month"
    )
    return fig, metrics


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
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        filtered_data = filtered_data[filtered_data["MONTH_DATE"] >= start_date]
    if end_date:
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)
        filtered_data = filtered_data[filtered_data["MONTH_DATE"] <= end_date]
    return filtered_data


def prepare_data(data, n_steps=6):
    if data.empty:
        return pd.DataFrame(columns=["MONTH_DATE", "PAID_AMOUNT"])
    data_to_join = (
        data
        .groupby("MONTH_DATE")["PAID_AMOUNT"]
        .sum()
        .reset_index()
    )
    data_to_join["PAID_AMOUNT__EWM"] = data_to_join["PAID_AMOUNT"].ewm(2).mean()
    dates = pd.date_range(
        data_to_join["MONTH_DATE"].min(),
        data_to_join["MONTH_DATE"].max() + relativedelta(months=n_steps + 1),
        freq="M"
    )
    result = pd.DataFrame({"MONTH_DATE": [dt.date().replace(day=1) for dt in dates]})
    result = result.merge(data_to_join, on="MONTH_DATE", how="left")
    return result


def median_predict(data, n_steps, with_fit=False):
    result = data.sort_values("MONTH_DATE")
    to_fit = data[:-n_steps]
    if with_fit:
        result["PAID_AMOUNT__MEDIAN"] = to_fit["PAID_AMOUNT"].median()
    else:
        result["PAID_AMOUNT__MEDIAN"] = (
            [np.nan] * len(to_fit) + [to_fit["PAID_AMOUNT"].median()] * n_steps
        )
    return result


def holtwinters_predict(data, n_steps, with_fit=False, alpha=1/(2*12), amount_col="PAID_AMOUNT__EWM"):
    data = data.sort_values("MONTH_DATE")
    to_fit = data[:-n_steps]
    to_predict = data[-n_steps:]

    simple_hwes_model = SimpleExpSmoothing(to_fit[amount_col].to_list()).fit(
        smoothing_level=alpha,
        optimized=False,
        use_brute=True
    )
    fittedvalues = simple_hwes_model.fittedvalues if with_fit else [np.nan] * len(to_fit)
    data["PAID_AMOUNT__HWES_SIMPLE"] = (
        list(fittedvalues) +
        list(simple_hwes_model.forecast(n_steps))
    )

    hwes_add_model = ExponentialSmoothing(to_fit[amount_col], trend="add").fit()
    fittedvalues = hwes_add_model.fittedvalues if with_fit else [np.nan] * len(to_fit)
    data["PAID_AMOUNT__HWES_ADD"] = (
        list(fittedvalues) +
        list(hwes_add_model.forecast(n_steps))
    )

    hwes_mul_model = ExponentialSmoothing(to_fit[amount_col], trend="mul").fit()
    fittedvalues = hwes_mul_model.fittedvalues if with_fit else [np.nan] * len(to_fit)
    data["PAID_AMOUNT__HWES_MUL"] = (
        list(fittedvalues) +
        list(hwes_mul_model.forecast(n_steps))
    )
    return data


def arima_predict(data, n_steps):
    data = data.sort_values("MONTH_DATE")
    to_fit = data[:-n_steps]

    arima_model = ARIMA(to_fit["PAID_AMOUNT"].to_list()).fit()
    data["PAID_AMOUNT__ARIMA"] = (
        list([np.nan] * len(to_fit)) +
        list(arima_model.predict(start=1, end=n_steps))
    )
    return data


def count_mape(plot_data):
    metrics = []
    metrics_data = plot_data[
        ~plot_data["PAID_AMOUNT__REAL"].isnull()
    ]
    if metrics_data.empty:
        return []
    for col in metrics_data.columns:
        if col in {"PAID_AMOUNT__REAL", "PAID_AMOUNT"}:
            continue
        mape = mean_absolute_percentage_error(
            metrics_data["PAID_AMOUNT__REAL"],
            metrics_data[col].fillna(1)
        )
        metrics.append({"Predict": col, "MAPE": round(mape, 2)})
    return metrics


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

