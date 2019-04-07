import dash
import dash_core_components as dcc
import dash_html_components as html
import colorlover as cl
import plotly.graph_objs as go


def two_model_feature_importance_app(model1, model2, feature_names, port=8050):
    """
    This function generate and start a simple Dash application to compare feature importance for two models
    of the same Data.

    First two parameters are models that use Scikit Learn interface, they should be trained on the same features.
    They should have two attributes set:
    .feature_importances_: list attribute set, that sum-up to 1 - like sklearn tree-based models.
    .name: model identifier


    :param model1:
    :param model2:
    :param feature_names: list of names of the features used to train models. Order is important!
    :param port: port where the application would be available
    :return: Dash applilcation
    """

    # first we define color palette
    try:
        colors = cl.scales[str(len(feature_names))]['qual']['Set2']
    except KeyError:
        # When the number of features is grater than available in palette, we need to interpolate some colors:
        colors = cl.interp(cl.scales['7']['qual']['Set2'], len(feature_names))

    # then we wrap-up all data into dictionary cosy dictionary with feature names as keys:
    feature_dict={}
    for i in range(len(feature_names)):
        feature_dict[feature_names[i]]={
            'm1_importance': model1.feature_importances_[i],
            'm2_importance': model2.feature_importances_[i],
            'color': colors[i]
        }

    # We also need to establish order of the data to show. Let's make it average of the importance for both models.
    feature_ordered = list(feature_dict.keys())
    feature_ordered.sort(key = lambda x: +feature_dict[x]['m1_importance']+feature_dict[x]['m2_importance'])

    # Now we build traces for the bar chart.
    traces = []
    for feature in feature_ordered:
        trace = {
            'x': [feature_dict[feature]['m1_importance'], feature_dict[feature]['m2_importance']],
            'y': [model1.name, model2.name],
            'type': 'bar',
            'orientation': 'h',
            'name': feature,
            'marker': {'color':feature_dict [feature]['color']},
        }
        traces.append(trace)

    first_graph = dcc.Graph(
        id='bar_chart',
        figure={
            'data': traces,
            'layout': {
                'barmode': 'stack',
                'title': 'Features importances for two models',
                'legend': {'orientation': 'h',
                           'traceorder': 'normal'}
            }
        }
    )

    # And for Scatter Chart - althrough it need single trace only.
    scatter_traces = []
    scatter_trace = go.Scatter(
        x=[(feature_dict[name]['m1_importance'] + feature_dict[name]['m2_importance'])/2 for name in feature_ordered],
        y=[feature_dict[name]['m2_importance'] - feature_dict[name]['m1_importance'] for name in feature_ordered],
       mode='markers+text',
       marker=dict(size=20,
                   color=[feature_dict[name]['color'] for name in feature_ordered]),
       text=feature_ordered,
       textposition='bottom center'
    )
    scatter_traces.append(scatter_trace)

    second_graph = dcc.Graph(
        id='scatter_plot',
        figure={
            'data': scatter_traces,
            'layout': {
                'title': 'Averages(x) and Differences(y) of feature importance for model {}(up) and model {}(down)'.format(model2.name, model1.name)
            }
        }
    )

    # Finally we can define application layout...
    app = dash.Dash(__name__)
    app.layout = html.Div(children=[
        html.H1(children='Hello!'),

        html.Div(children='This is an application to compare Feature importance between two models'),
        first_graph,
        second_graph
    ])

    # and start it on the return
    return app.run_server(port=port)
