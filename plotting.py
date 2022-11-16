from sklearn.preprocessing import MinMaxScaler, StandardScaler
from plotly.colors import sample_colorscale
import math
import pandas as pd
import numpy as np
from wordcloud import WordCloud
from plotly.subplots import make_subplots
from itertools import combinations
from plotly.graph_objects import Scatter3d, Volume
import os

def wordcloud(loadings):
    """
    This function generates a wordcloud from the loadings of a topic model.
    Args:
        loadings (pandas.DataFrame): The loadings of a topic model.
    Returns:
        im (PIL.Image.Image): The wordcloud image.
    """
    unscaledloadings = loadings.sort_values(ascending=False)
    scaledloadings = MinMaxScaler().fit_transform(unscaledloadings.values.reshape(-1, 1)).flatten().round(4)
    scaledloadings_col = (
        MinMaxScaler(feature_range=(-10, 10)).fit_transform(unscaledloadings.values.reshape(-1, 1)).round(4)
    )
    scaledloadings_col = (
        pd.DataFrame(scaledloadings_col.T)
        .apply(lambda x: 1 / (1 + math.exp(-x)))
        .to_numpy()
    )
    colours = sample_colorscale("RdBu", samplepoints=scaledloadings_col)
    colour_dict = {x.split("_")[0]: y for x, y in zip(unscaledloadings.index, colours)}
    absolutescaledloadings = np.where(
        scaledloadings < 0.5, 1 - scaledloadings, scaledloadings
    )
    rescaledloadings = (
        MinMaxScaler().fit_transform(absolutescaledloadings.reshape(-1, 1)).flatten()
    )
    freq_dict = {
        x.split("_")[0]: y for x, y in zip(unscaledloadings.index, rescaledloadings)
    }

    def color_func(
        word, *args, **kwargs
    ):  # colour function to supply to wordcloud function.. don't ask !
        return colour_dict[word]

    wc = WordCloud(
        background_color="white",
        color_func=color_func,
        width=400,
        height=400,
        prefer_horizontal=1,
        min_font_size=8,
        max_font_size=200,
    )
    # generate wordcloud from loadings in frequency dict
    wc = wc.generate_from_frequencies(freq_dict)
    im = wc.to_image()
    return im



def to_html(X, y, loadings, Tasklabels, path="./"):
    
    """
    This function creates two html files with the estimated mean and standard deviation of the predicted loadings.
    Parameters
    ----------
    X : numpy.ndarray
        The matrix of the gradients.
    y : numpy.ndarray
        The matrix of the loadings.
    loadings : pandas.DataFrame
        The dataframe of the loadings.
    Tasklabels : list
        The list of the task labels.
    path : str
        The path to the directory where the html files will be saved.
    Returns
    -------
    None
    """
    from defs import gpr
    fig_estimated_mean = make_subplots(
        rows=4,
        cols=2,
        column_widths=[0.7, 0.25],
        vertical_spacing=0,
        horizontal_spacing=0.2,
        subplot_titles=(
            [
                "Component 1",
                "Component 1 Wordcloud",
                "Component 2",
                "Component 2 Wordcloud",
                "Component 3",
                "Component 3 Wordcloud",
                "Component 4",
                "Component 4 Wordcloud",
            ]
        ),
        specs=[
            [{"type": "surface"}, {"type": "xy"}],
            [{"type": "surface"}, {"type": "xy"}],
            [{"type": "surface"}, {"type": "xy"}],
            [{"type": "surface"}, {"type": "xy"}],
        ],
    )
    fig_standard_deviation = make_subplots(
        rows=4,
        cols=2,
        column_widths=[0.7, 0.25],
        vertical_spacing=0,
        horizontal_spacing=0.2,
        subplot_titles=(
            [
                "Component 1",
                "Component 1 Wordcloud",
                "Component 2",
                "Component 2 Wordcloud",
                "Component 3",
                "Component 3 Wordcloud",
                "Component 4",
                "Component 4 Wordcloud",
            ]
        ),
        specs=[
            [{"type": "surface"}, {"type": "xy"}],
            [{"type": "surface"}, {"type": "xy"}],
            [{"type": "surface"}, {"type": "xy"}],
            [{"type": "surface"}, {"type": "xy"}],
        ],
    )
    combs = list(combinations(range(X.shape[1]), 3))
    lim = 0.6
    res = 30
    for i in range(y.shape[1]):
        for c in combs:
            X_ = X[:, c]
            standardscaler = StandardScaler()
            y_ = standardscaler.fit_transform(y[:, i].reshape(-1, 1))
            gprmodel = gpr()
            gprmodel.fit(X_, y_)
            lin = np.linspace(-lim, lim, res)
            x1, x2, x3 = np.meshgrid(lin, lin, lin)
            xx = np.vstack((x1.flatten(), x2.flatten(), x3.flatten())).T
            y_mean, y_sd = gprmodel.predict(xx, return_std=True)
            fig_estimated_mean.add_trace(
                Volume(
                    x=pd.Series(x1.flatten(), name="Gradient 1"),
                    y=pd.Series(x2.flatten(), name="Gradient 2"),
                    z=pd.Series(x3.flatten(), name="Gradient 3"),
                    value=y_mean,
                    hoverinfo="skip",
                    opacityscale=[[0, 0.8], [0.35, 0], [0.65, 0], [1, 0.8]],
                    surface_count=25,
                    showlegend=False,
                    colorscale="RdBu",
                    colorbar={
                        "tickmode": "array",
                        "tickvals": [min(y_mean), max(y_mean)],
                        "ticktext": ["Predicted low loading", "Predicted high loading"],
                    },
                ),
                i + 1,
                1,
            )
            fig_estimated_mean.update(
                layout_scene=dict(
                    xaxis_title="Gradient 1",
                    yaxis_title="Gradient 2",
                    zaxis_title="Gradient 3",
                    aspectmode="data",
                ),
            )
            fig_estimated_mean.add_trace(
                Scatter3d(
                    x=X_[:, 0],
                    y=X_[:, 1],
                    z=X_[:, 2],
                    marker_color=y[:, i],
                    marker_colorscale="RdBu",
                    text=Tasklabels,
                    mode="markers+text",
                    showlegend=False,
                ),
                i + 1,
                1,
            )
            fig_standard_deviation.add_trace(
                Volume(
                    x=pd.Series(x1.flatten(), name="Gradient 1"),
                    y=pd.Series(x2.flatten(), name="Gradient 2"),
                    z=pd.Series(x3.flatten(), name="Gradient 3"),
                    value=y_sd,
                    hoverinfo="skip",
                    showlegend=False,
                    colorscale="RdBu",
                    # showscale=False,
                    opacityscale=[[0, 0.8], [0.35, 0], [0.65, 0], [1, 0.8]],
                    colorbar={
                        "tickmode": "array",
                        "tickvals": [min(y_sd), max(y_sd)],
                        "ticktext": [
                            "Low uncertainty in predicted loading",
                            "High uncertainty in predicted loading",
                        ],
                    },
                    surface_count=25,
                ),
                i + 1,
                1,
            )
            fig_standard_deviation.update(
                layout_scene=dict(
                    xaxis_title="Gradient 1",
                    yaxis_title="Gradient 2",
                    zaxis_title="Gradient 3",
                    aspectmode="data",
                ),
            )
            unscaledloadings = loadings[[f"Component {i}"]]
            im = wordcloud(unscaledloadings)
            display_wordclouds(fig_estimated_mean, im, i)
            display_wordclouds(fig_standard_deviation, im, i)
    if not os.path.exists(path):
        os.makedirs(path)
    fig_estimated_mean.write_html(
        os.path.join(path, "estimated_mean.html"),
        default_width=2000,
        default_height=5000,
    )
    fig_standard_deviation.write_html(
        os.path.join(path, "standard_deviation.html"),
        default_width=2000,
        default_height=5000,
    )


# TODO Rename this here and in `to_html`
def display_wordclouds(arg0, im, i):
    """
    This is a multi-line Google style docstring.
    Args:
        arg0 (TYPE): Description of arg0
        im (TYPE): Description of im
        i (TYPE): Description of i
    Returns:
        TYPE: Description of return value
    """
    # im = Image.open("clouds/{}.png".format(i+1))
    arg0.add_layout_image(
        dict(
            source=im,
            xref="x",
            yref="y",
            x=0,
            y=0,
            xanchor="center",
            yanchor="middle",
            sizex=2,
            sizey=2,
        ),
        row=i + 1,
        col=2,
    )
    arg0.update_xaxes(
        range=[-1, 1],
        showticklabels=False,
        row=i + 1,
        col=2,
        showgrid=False,
        zeroline=False,
    )
    arg0.update_yaxes(
        range=[-1, 1],
        showticklabels=False,
        row=i + 1,
        col=2,
        showgrid=False,
        zeroline=False,
    )


def display_scores(loadings, scores, name="component", numloadings=5):
    """
    Display the results of the permutation test for each component.

    Args:
        loadings (pandas.DataFrame): The loadings of the components.
        scores (dict): The scores of the permutation test.
        name (str): The name of the components.
        numloadings (int): The number of loadings to display.

    Returns:
        None
    """
    for score in scores:
        if score == "total":
            print(
                f"""
            Results for ALL {name}s in common space:
            """
            )
            print(
                """
            Permutation test score: {},
            Permutation test significance: {}
            """.format(
                    scores[score][0], scores[score][1]
                )
            )
            continue
        try:
            loading = loadings[f"Component {score}"]
            loadingpos = loading.apply(lambda x: np.abs(x)).sort_values(ascending=False)
            tops = loading[loadingpos[:numloadings].index]
        except Exception:
            tops = "Loadings are unavailable"
        print(
            f"""
        Results for {name} {score}:
        Largest loadings: 
        """
        )
        with pd.option_context(
            "display.max_rows",
            5,
            "display.max_columns",
            None,
            "display.width",
            1000,
            "display.precision",
            3,
            "display.colheader_justify",
            "center",
        ):
            print(tops)
        print(
            """
        Permutation test score: {},
        Permutation test significance: {}
        """.format(
                scores[score][0], scores[score][1]
            )
        )