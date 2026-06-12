"""
Visualization utilities for report chart generation.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List

from evalscope.constants import DEFAULT_BAR_WIDTH, PLOTLY_THEME, DataCollection
from evalscope.report import Report, ReportKey, get_data_frame
from evalscope.utils.logger import get_logger

logger = get_logger()


def plot_single_report_scores(df: pd.DataFrame):
    if df is None:
        return None
    logger.debug(f'df: \n{df}')
    plot = px.bar(df, x=df[ReportKey.dataset_name], y=df[ReportKey.score], text=df[ReportKey.score])

    width = DEFAULT_BAR_WIDTH if len(df[ReportKey.dataset_name]) <= 5 else None
    plot.update_traces(width=width, texttemplate='%{text:.2f}', textposition='outside')
    plot.update_layout(uniformtext_minsize=12, uniformtext_mode='hide', yaxis=dict(range=[0, 1]), template=PLOTLY_THEME)
    return plot


def plot_single_report_sunburst(report_list: List[Report]):
    if report_list[0].name == DataCollection.NAME:
        df = get_data_frame(report_list=report_list)
        categories = sorted([i for i in df.columns if i.startswith(ReportKey.category_prefix)])
        path = categories + [ReportKey.subset_name]
    else:
        df = get_data_frame(report_list=report_list, flatten_metrics=False)
        categories = sorted([i for i in df.columns if i.startswith(ReportKey.category_prefix)])
        path = [ReportKey.dataset_name] + categories + [ReportKey.subset_name]
    logger.debug(f'df: \n{df}')
    df[categories] = df[categories].fillna('default')  # NOTE: fillna for empty categories
    df = df[df[ReportKey.num] > 0]  # NOTE: filter out zero-num rows to avoid ZeroDivisionError in plotly
    if df.empty:
        return None

    plot = px.sunburst(
        df,
        path=path,
        values=ReportKey.num,
        color=ReportKey.score,
        color_continuous_scale='RdYlGn',  # see https://plotly.com/python/builtin-colorscales/
        color_continuous_midpoint=np.average(df[ReportKey.score], weights=df[ReportKey.num])
        if df[ReportKey.num].sum() > 0 else df[ReportKey.score].mean(),
        template=PLOTLY_THEME,
        maxdepth=4
    )
    plot.update_traces(insidetextorientation='radial')
    plot.update_layout(margin=dict(t=10, l=10, r=10, b=10), coloraxis=dict(cmin=0, cmax=1), height=600)
    return plot


def plot_single_dataset_scores(df: pd.DataFrame):
    plot = px.bar(
        df,
        x=df[ReportKey.metric_name],
        y=df[ReportKey.score],
        color=df[ReportKey.subset_name],
        text=df[ReportKey.score],
        barmode='group'
    )

    width = 0.2 if len(df[ReportKey.subset_name]) <= 3 else None
    plot.update_traces(width=width, texttemplate='%{text:.2f}', textposition='outside')
    plot.update_layout(uniformtext_minsize=12, uniformtext_mode='hide', yaxis=dict(range=[0, 1]), template=PLOTLY_THEME)
    return plot


def plot_multi_report_radar(df: pd.DataFrame):
    dataset_colors = px.colors.qualitative.Bold + px.colors.qualitative.Vivid

    fig = go.Figure()

    grouped = df.groupby(ReportKey.model_name)
    common_datasets = sorted(set.intersection(*[set(group[ReportKey.dataset_name]) for _, group in grouped]))

    ds_color_map = {ds: dataset_colors[i % len(dataset_colors)] for i, ds in enumerate(common_datasets)}

    for model_name, group in grouped:
        group = group.set_index(ReportKey.dataset_name)
        for ds in common_datasets:
            color = ds_color_map[ds]
            fig.add_trace(
                go.Scatterpolar(
                    r=[0, group.loc[ds, ReportKey.score]],
                    theta=[ds, ds],
                    name=f'{model_name} — {ds}',
                    mode='lines+markers+text',
                    text=['', ds],
                    textposition='top center',
                    textfont=dict(size=11, color='#444'),
                    line=dict(color=color, width=2.5),
                    marker=dict(size=10, color=color, line=dict(color='white', width=1.5)),
                )
            )

    fig.update_layout(
        template=PLOTLY_THEME,
        polar=dict(radialaxis=dict(visible=True, range=[0, 1]), angularaxis=dict(showticklabels=False)),
        margin=dict(t=20, l=20, r=20, b=20)
    )
    return fig
