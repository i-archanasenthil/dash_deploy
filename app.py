import pandas as pd
import numpy as np
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# Load data
df = pd.read_csv(r'data\synthetic_data_stats_competition_2025_final(in).csv')
df = df.drop(columns=["patient_id"])
corr_df = pd.read_csv(r'data\df_target_correlation.csv')
corr_df['correlation'] = abs(corr_df['correlation'])
corr_df = corr_df[corr_df['feature_2'] != 'outcome_afib_aflutter_new_post']
corr_df_cat = pd.read_csv(r'data\cat_correlation_strength.csv')

for col in df.columns:
    unique_vals = df[col].dropna().unique()
    if len(unique_vals) <= 2:
        df[col] = df[col].astype('category')

#numerical_cols = ['demographics_age_index_ecg','ecg_resting_hr','ecg_resting_pr','ecg_resting_qrs','ecg_resting_qtc','hgb_peri','hct_peri','rdw_peri','wbc_peri','plt_peri','inr_peri','ptt_peri','esr_peri','crp_high_sensitive_peri','albumin_peri','alkaline_phophatase_peri','alanine_transaminase_peri','aspartate_transaminase_peri','bilirubin_total_peri']
df_numerical = df.select_dtypes(include=['number'])
df_numerical['outcome_afib_aflutter_new_post'] = df['outcome_afib_aflutter_new_post']
df_categorical = df.select_dtypes(include=['object','category','bool'])

df_processed_1 = pd.read_csv(r'data\pda_data.csv')

X1 = df_processed_1.drop(columns=['outcome_afib_aflutter_new_post'])
y1 = df_processed_1['outcome_afib_aflutter_new_post']

scaler = StandardScaler()
scaled_data_1 = scaler.fit_transform(X1)

pca = PCA(n_components=2)
principal_component_1 = pca.fit_transform(scaled_data_1)

pca_df_1 = pd.DataFrame(principal_component_1, columns=['PC1', 'PC2'])
pca_df_1['Outcome'] = y1.astype(str)

custom_colors = {'0.0': "#f5b3ac", '1.0': "#db6459"}

columns_for_correlation = df_processed_1.columns
filtered_df = corr_df[corr_df['feature_2'].isin(columns_for_correlation)]
filtered_df = filtered_df[filtered_df['feature_2'] != 'outcome_afib_aflutter_new_post']
filtered_df['correlation'] = abs(filtered_df['correlation'])

app = dash.Dash(__name__)
server = app.server
app.title = "Interactive Exploratory Data Analysis"

app.layout = html.Div([
    html.H2("Exploratory Data Analysis", style={'textAlign': 'center'}),
    dcc.Tabs(id = 'tabs', value = 'tab1', children = [
        #Tab 1 for all the variables
        dcc.Tab(label='Correlation Analysis for the entire data', value = "tab1" ,children = [
            html.Div([
        html.Div([
            # Add tabs here:   -> dcc.Tab(label='Tab one', children=[
            html.H3("Correlation with Target for Numerical Features", style ={'textAlign': 'center'}),
            html.Div("The Pearson correlation coefficient measures the linear relationship between two numerical variables. "
                    "Values close to +1 or -1 indicate strong relationships, while values near 0 suggest weak or no linear correlation."
                    "The graph is plotted for the absolute value of correlation to understand if there is any data leakage"),
            dcc.Graph(
                id='correlation-bar-tab1',
                figure=px.bar(
                    data_frame=corr_df[corr_df['feature_2'].isin(df_numerical.columns)].sort_values(by= 'correlation', ascending=True),
                    y='feature_2',
                    x='correlation',
                    hover_data='correlation',
                    color='correlation',
                    color_continuous_scale=['#fbe3e0', '#f5b3ac', '#db6459', '#b4423a', '#7d2b23'],
                    labels={'correlation': 'Absolute Correlation Value', 'feature_2': 'Features'},
                    width=900,
                    height=1380
                ).update_layout(coloraxis_showscale=False,  plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray'))
            )
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'height': '1450px'}),

            html.Div([
                html.H3("Correlation with Target for Categorical Features", style ={'textAlign': 'center'}),
                html.Div("The cross-tabulation shows the distribution of the outcome across the different categories of categorical features."
                         "The positive rate/signal strength indicates the percentage of values within each category that falls into a positive outcome for Atrial Fibrilation."),
                dcc.Graph(
                    id='correlation-bar-cat-tab1',
                    figure=px.bar(
                        data_frame=corr_df_cat[corr_df_cat['feature'].isin(df_categorical.columns)].sort_values(by= 'signal_strength', ascending=True),
                        y='feature',
                        x='signal_strength',
                        hover_data='signal_strength',
                        color='signal_strength',
                        color_continuous_scale=['#fbe3e0', '#f5b3ac', '#db6459', '#b4423a', '#7d2b23'],
                        labels={'signal_strength': 'Absolute Correlation Value/ Positive Rate', 'feature': 'Features'},
                        width=900,
                        height=1380
                    ).update_layout(coloraxis_showscale=False,  plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray'))
                )
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'height': '1450px'}),
        ]),


        html.Div(id='selected-feature'),

        html.Div([
            html.Div(
                dcc.Graph(id='box-plot-tab1'),
                style={'width': '48%', 'display': 'inline-block'}
            ),
            html.Div(
                dcc.Graph(id='histogram-tab1'),
                style={'width': '48%', 'display': 'inline-block', 'padding-left': '4%'}
            )
        ], style={'display': 'flex', 'justify-content': 'space-between'})
    ]),

        dcc.Tab(label = 'Analysis on the model features', value = "tab2" ,children = [
            html.Div([
        html.Div([
            html.H3("Correlation of Key Features with Target", style ={'textAlign': 'center'}),
            html.Div("The Pearson correlation coefficient measures the linear relationship between two numerical variables. "
                    "Values close to +1 or -1 indicate strong relationships, while values near 0 suggest weak or no linear correlation."
                    "The graph is plotted for the absolute value of correlation to understand if there is any data leakage"),
            dcc.Graph(
                id='correlation-bar-tab2',
                figure=px.bar(
                    data_frame=filtered_df.sort_values(by= 'correlation', ascending=True),
                    y='feature_2',
                    x='correlation',
                    hover_data='correlation',
                    color='correlation',
                    color_continuous_scale=['#fbe3e0', '#f5b3ac', '#db6459', '#b4423a', '#7d2b23'],
                    labels={'correlation': 'Absolute Correlation Value', 'feature_2': 'Features'},
                    width=900,
                    height=680
                ).update_layout(coloraxis_showscale=False,  plot_bgcolor='white', xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray'))
                )
            ], style={'width': '50%', 'display': 'inline-block', 'verticalAlign': 'top', 'height': '750px'}),

            html.Div([
                html.H3("PCA: Classification of Data with the key features", style ={'textAlign': 'center'}),
                html.Div("Principal Component Analysis (PCA) is a dimensionality reduction technique that transforms the dataset into a set of orthogonal components. "
                        "The goal is to reduce the number of variables while preserving as much variance as possible."),
                dcc.Graph(
                    id='pca-plot-tab2',
                    figure = px.scatter(
                        pca_df_1,
                        x='PC1',
                        y='PC2',
                        color='Outcome',
                        color_discrete_map=custom_colors,
                        hover_data=['PC1', 'PC2','Outcome'],
                        width=900,
                        height=680,
                        opacity=0.6
                    ).update_layout(coloraxis_showscale=False, plot_bgcolor='white',  xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray'))
                )
            ], style={'width': '48%', 'display': 'inline-block', 'padding-left': '2%', 'height': '700px'})
        ]),


        html.Div(id='selected-feature-tab2'),

        html.Div([
            html.Div(
                dcc.Graph(id='box-plot-tab2'),
                style={'width': '48%', 'display': 'inline-block', 'padding-left': '4%'}
            ),
            html.Div(
                dcc.Graph(id='histogram-tab2'),
                style={'width': '48%', 'display': 'inline-block', 'padding-left': '4%'}
            )
        ], style={'display': 'flex', 'justify-content': 'space-between'})
    ])
            ])
            ])

 

@app.callback(
    Output('box-plot-tab1', 'figure'),
    Output('histogram-tab1', 'figure'),
    Output('selected-feature', 'children'),
    Input('correlation-bar-tab1', 'hoverData'),
    Input('correlation-bar-cat-tab1','hoverData')
)
def update_tab1_plots(num_hover, cat_hover):
    ctx = dash.callback_context
    empty_fig = go.Figure()
    empty_fig.add_annotation(
        text = "Box Plot not available for categorical columns", 
        showarrow=False
    )
    empty_fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, plot_bgcolor='white')

    if not ctx.triggered:
        return {},{}, "Hover over a bar to see the distribution"

    prop_id = ctx.triggered[0]['prop_id']
    if prop_id == 'correlation-bar-tab1.hoverData' and num_hover:
        hoverData = num_hover
        hovered_feature = hoverData['points'][0]['y']
        box_fig = px.box(
            data_frame=df_numerical,
            y='outcome_afib_aflutter_new_post',
            x=hovered_feature,
            orientation='h',
            color='outcome_afib_aflutter_new_post',
            color_discrete_sequence=['#f5b3ac', '#db6459']
        )
        box_fig.update_layout(width=800,  plot_bgcolor='white', title={'text': f'Distribution of {hovered_feature} by Target','x': 0.5,'xanchor': 'center'},  xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray')),

        hist_fig = px.histogram(
            data_frame=df,
            x=hovered_feature,
            color='outcome_afib_aflutter_new_post',
            color_discrete_map={0: "#f5b3ac", 1: "#db6459"},
            title=f"Histogram of {hovered_feature}",
            nbins=50
        )
        hist_fig.update_layout(width=800, plot_bgcolor='white', title={'text': f'Distribution of {hovered_feature}','x': 0.5,'xanchor': 'center'},  xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray'))

        return box_fig, hist_fig, f"Selected Feature: {hovered_feature}"
    
    elif prop_id == 'correlation-bar-cat-tab1.hoverData' and cat_hover:
        hoverData = cat_hover
        hovered_feature = hoverData['points'][0]['y']

        box_fig = empty_fig
        hist_fig = px.histogram(
            data_frame=df,
            x=hovered_feature,
            color='outcome_afib_aflutter_new_post',
            color_discrete_map={0: "#f5b3ac", 1: "#db6459"},
            title=f"Histogram of {hovered_feature}",
            nbins=50
        )
        hist_fig.update_layout(width=800, plot_bgcolor='white', title={'text': f'Histogram of {hovered_feature}','x': 0.5,'xanchor': 'center'},  xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray'))

        return box_fig, hist_fig, f"Selected Feature: {hovered_feature}"

    else:
        return {},{},"Hover over a bar to see the distribution"

@app.callback(
    Output('box-plot-tab2', 'figure'),
    Output('histogram-tab2', 'figure'),
    Output('selected-feature-tab2', 'children'),
    Input('correlation-bar-tab2', 'hoverData')
)
def update_tab2_plots(hoverData):
    if hoverData:
        hovered_feature = hoverData['points'][0]['y']
        box_fig = px.box(
            data_frame=df,
            y='outcome_afib_aflutter_new_post',
            x=hovered_feature,
            orientation='h',
            color='outcome_afib_aflutter_new_post',
            color_discrete_sequence=['#f5b3ac', '#db6459']
        )
        box_fig.update_layout(width=800,  plot_bgcolor='white', title={'text': f'Distribution of {hovered_feature} by Target','x': 0.5,'xanchor': 'center'},  xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray')),

        hist_fig = px.histogram(
            data_frame=df,
            x=hovered_feature,
            color='outcome_afib_aflutter_new_post',
            color_discrete_map={0: "#f5b3ac", 1: "#db6459"},
            title=f"Histogram of {hovered_feature}",
            nbins=50
        )
        hist_fig.update_layout(width=800, plot_bgcolor='white', title={'text': f'Distribution of {hovered_feature}','x': 0.5,'xanchor': 'center'},  xaxis=dict(showgrid=True, gridcolor='lightgray'), yaxis=dict(showgrid=True, gridcolor='lightgray'))

        return box_fig, hist_fig, f"Selected Feature: {hovered_feature}"
    else:
        return {}, {}, "Hover over a bar to see the distribution"
    

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
