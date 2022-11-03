import math
import os

user_dir = os.path.expanduser("~")
os.environ["R_HOME"] = f"{user_dir}/anaconda3/envs/gpr/Lib/R"
os.environ["PATH"] = f"{user_dir}/anaconda3/envs/gpr/Lib/R/bin/x64;" + os.environ["PATH"]

from advanced_pca import CustomPCA
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo
from PIL import Image
from plotly.express.colors import sample_colorscale
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.model_selection import KFold, permutation_test_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils._testing import ignore_warnings
from wordcloud import WordCloud

pio.templates.default = "plotly_dark"



from sklearn.base import BaseEstimator

class gpr(BaseEstimator):
    def __init__(self,kernel= Matern(nu=2.5) + WhiteKernel()):
        self.kernel = kernel 
        self.model = GaussianProcessRegressor(
            kernel=kernel, random_state=3, normalize_y=False, alpha=0
        )
    def fit(self,X, y):
        @ignore_warnings(category=ConvergenceWarning)
        def _f():
            self.model.fit(X, y)
        _f()
        return self.model
    def predict(self, X, **kwargs):
        return self.model.predict(X,**kwargs)
    def score(self, X, y):
        return self.model.score(X, y)

def PCAfunc(PCAdata):
    chi_square_value, p_value = calculate_bartlett_sphericity(
    PCAdata
)  # I belive this checks for correlations within our dataset which would make a PCA weird
    print(f"Bartlett sphericity: {chi_square_value}, p-value: {p_value}")  # significant p-value means we're ok
    kmo_all, kmo_model = calculate_kmo(
    PCAdata
)  # Not even going to pretend I understand the Kaiser-Meyer-Olkin criteria
    print(f"KMO test: {kmo_model}")  # We want this to be > 0.6"kmo_all, kmo_model  # kmo_model > 0.6 is acceptable
    PCAmodel = CustomPCA(n_components=4,rotation='varimax')
    PCAmodel.fit(PCAdata)
    loadings = PCAmodel.components_
    names = PCAdata.columns
    loadings = pd.DataFrame(
    np.round(loadings.T, 3),
    index=names,
    columns=["Component 0", "Component 1", "Component 2", "Component 3"],
)
    PCAresults = PCAmodel.transform(PCAdata).T
    return loadings,PCAresults




def averageData(data, PCAresults,average=True):
    FAC = np.asarray([PCAresults[0], PCAresults[1], PCAresults[2], PCAresults[3]]).T
    GRAD = np.asarray([data["Gradient 1"], data["Gradient 2"], data["Gradient 3"]]).T
    Tasklabels, Taskindices = np.unique(data["Task_name"], return_inverse=True)
    if average == True:
        tasknum = len(data["Task_name"].unique())
        
        FAC_TaskCentres = np.zeros([tasknum, 4])
        for i in range(tasknum):
            FAC_TaskCentres[i, :] = FAC[Taskindices == i, :].mean(axis=0)
        Grad_TaskCentres = np.zeros([tasknum, 3])
        for i in range(tasknum):
            Grad_TaskCentres[i, :] = GRAD[np.ix_(Taskindices == (i), [0, 1, 2])].mean(axis=0)
    else:
        FAC_TaskCentres = FAC
        Grad_TaskCentres = GRAD
    return Tasklabels,FAC_TaskCentres,Grad_TaskCentres




def wordcloud(loadings):
    unscaledloadings = loadings.sort_values(
        by=loadings.columns[0], ascending=False
    )
    scaledloadings = MinMaxScaler().fit_transform(unscaledloadings).flatten().round(4)
    scaledloadings_col = (
        MinMaxScaler(feature_range=(-10, 10)).fit_transform(unscaledloadings).round(4)
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
# Fit whole dataset
def to_html(X,y,loadings,Tasklabels,path='./'):
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
    for i in range(y.shape[1]):
        for c in combs:
            X_ = X[:,c]
            standardscaler = StandardScaler()
            
            y_ = standardscaler.fit_transform(y[:, i].reshape(-1, 1))
            
            gprmodel = gpr()
            gprmodel.fit(X_, y_)
            lim = 0.6
            res = 30
            lin = np.linspace(-lim, lim, res)
            x1, x2, x3 = np.meshgrid(lin, lin, lin)
            xx = np.vstack((x1.flatten(), x2.flatten(), x3.flatten())).T
            y_mean, y_sd = gprmodel.predict(xx, return_std=True)
            fig_estimated_mean.add_trace(
            go.Volume(
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
            go.Scatter3d(
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
            go.Volume(
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
            
            unscaledloadings = loadings[["Component {}".format(i)]]
            im = wordcloud(unscaledloadings)
        # im = Image.open("clouds/{}.png".format(i+1))
            fig_estimated_mean.add_layout_image(
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
            fig_estimated_mean.update_xaxes(
            range=[-1, 1],
            showticklabels=False,
            row=i + 1,
            col=2,
            showgrid=False,
            zeroline=False,
        )
            fig_estimated_mean.update_yaxes(
            range=[-1, 1],
            showticklabels=False,
            row=i + 1,
            col=2,
            showgrid=False,
            zeroline=False,
        )
            fig_standard_deviation.add_layout_image(
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
            fig_standard_deviation.update_xaxes(
            range=[-1, 1],
            showticklabels=False,
            row=i + 1,
            col=2,
            showgrid=False,
            zeroline=False,
        )
            fig_standard_deviation.update_yaxes(
            range=[-1, 1],
            showticklabels=False,
            row=i + 1,
            col=2,
            showgrid=False,
            zeroline=False,
        )
    if not os.path.exists(path):
        os.makedirs(path)
    fig_estimated_mean.write_html(
    os.path.join(path,"estimated_mean.html"), default_width=2000, default_height=5000
)
    fig_standard_deviation.write_html(
    os.path.join(path,"standard_deviation.html"), default_width=2000, default_height=5000
)










def permtest(X,y,k=5):
    kf = KFold(n_splits=k, random_state=None)
    gprmodel = gpr()
    score_gradfac, perm_scores_gradfac, pvalue_gradfac = permutation_test_score(
    gprmodel,
    X,
    y,
    scoring="neg_mean_absolute_error",
    cv=kf,
    n_permutations=1000,
    n_jobs=-1,
    )
    scores = {'total': [score_gradfac, pvalue_gradfac]}
    for PCnum in range(y.shape[1]):
        y_ = y[:, PCnum]
        kf = KFold(n_splits=k, random_state=None)
        gprmodel = gpr()
        score_gradfac, perm_scores_gradfac, pvalue_gradfac = permutation_test_score(
        gprmodel,
        X,
        y_,
        scoring="neg_mean_absolute_error",
        cv=kf,
        n_permutations=1000,
        n_jobs=-1,
    )
        scores[PCnum] = [score_gradfac, pvalue_gradfac]


    return scores




def display_scores(loadings, scores,name="component",numloadings = 5):
    
    for score in scores:
        if score == 'total':
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
data = pd.read_csv(
    "output_.csv"
)  # Reading datafile (should be in the same directory as our IDE)
PCAdata = data.drop(
    [
        "Participant #",
        "Runtime_mod",
        "Task_name",
        "Gradient 1",
        "Gradient 2",
        "Gradient 3",
    ],
    axis=1,
)  # Getting rid of unneeded columns for PCA

loadings, PCAresults = PCAfunc(PCAdata)
Tasklabels, FAC_TaskCentres, Grad_TaskCentres = averageData(data, PCAresults,average=True)
to_html(X=Grad_TaskCentres,y=FAC_TaskCentres, loadings=loadings,Tasklabels=Tasklabels,path='figs')

scores = permtest(Grad_TaskCentres, FAC_TaskCentres)
display_scores(loadings, scores, "component")

to_html(X=FAC_TaskCentres,y=Grad_TaskCentres,loadings=loadings,Tasklabels=Tasklabels,path='pcafigs')

scores = permtest(FAC_TaskCentres,Grad_TaskCentres)
display_scores(loadings, scores, "gradient")




loadings, PCAresults = PCAfunc(PCAdata)
Tasklabels, FAC_TaskCentres, Grad_TaskCentres = averageData(data, PCAresults,average=False)
to_html(X=Grad_TaskCentres,y=FAC_TaskCentres, loadings=loadings,Tasklabels=Tasklabels,path='figs')

scores = permtest(Grad_TaskCentres, FAC_TaskCentres)
display_scores(loadings, scores, "component")

to_html(X=FAC_TaskCentres,y=Grad_TaskCentres,loadings=loadings,Tasklabels=Tasklabels,path='pcafigs')

scores = permtest(FAC_TaskCentres,Grad_TaskCentres)
display_scores(loadings, scores, "gradient")

# fig_estimated_mean = make_subplots(
#     rows=12,
#     cols=2,
#     column_widths=[0.7, 0.25],
#     vertical_spacing=0,
#     horizontal_spacing=0.2,
#     subplot_titles=(
#         [
#             "Component 1",
#             "Component 1 Wordcloud",
#             "Component 2",
#             "Component 2 Wordcloud",
#             "Component 3",
#             "Component 3 Wordcloud",
#             "Component 4",
#             "Component 4 Wordcloud",
#         ]
#     ),
#     specs=[
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#     ],
# )
# fig_standard_deviation = make_subplots(
#     rows=12,
#     cols=2,
#     column_widths=[0.7, 0.25],
#     vertical_spacing=0,
#     horizontal_spacing=0.2,
#     subplot_titles=(
#         [
#             "Component 1",
#             "Component 1 Wordcloud",
#             "Component 2",
#             "Component 2 Wordcloud",
#             "Component 3",
#             "Component 3 Wordcloud",
#             "Component 4",
#             "Component 4 Wordcloud",
#         ]
#     ),
#     specs=[
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#         [{"type": "surface"}, {"type": "xy"}],
#     ],
# )
# combs = list(combinations(range(FAC_TaskCentres.shape[1]), 3))
# ti = 0
# for i in range(Grad_TaskCentres.shape[1]):
#     for c in combs:
#         standardscaler = StandardScaler()
#         X = FAC_TaskCentres[:, c]
#         y = standardscaler.fit_transform(Grad_TaskCentres[:, i].reshape(-1, 1))
#         kernel = 1.0 * Matern(
#             length_scale=0.5, length_scale_bounds=(0.5, 1), nu=2.5
#         ) + WhiteKernel(noise_level_bounds=[0.001, 0.1], noise_level=0.05)
#         gpr = GaussianProcessRegressor(
#             kernel=kernel, random_state=3, normalize_y=False, alpha=0
#         )

#         @ignore_warnings(category=ConvergenceWarning)
#         def _f():
#             gpr.fit(X, y)

#         _f()
#         lim = 1.2
#         res = 20
#         lin = np.linspace(-lim, lim, res)
#         lins = [lin for x in range(X.shape[1])]
#         x_coords = np.meshgrid(*lins)
#         x_coords = [x.flatten() for x in x_coords]
#         xx = np.vstack(x_coords).T
#         y_mean, y_sd = gpr.predict(xx, return_std=True)
#         x1, x2, x3 = x_coords
#         fig_estimated_mean.add_trace(
#             go.Volume(
#                 x=x1,
#                 y=x2,
#                 z=x3,
#                 value=y_mean,
#                 hoverinfo="skip",
#                 opacityscale=[[0, 0.8], [0.35, 0], [0.65, 0], [1, 0.8]],
#                 surface_count=25,
#                 showlegend=False,
#                 colorscale="RdBu",
#                 colorbar={
#                     "tickmode": "array",
#                     "tickvals": [min(y_mean), max(y_mean)],
#                     "ticktext": ["Predicted low loading", "Predicted high loading"],
#                 },
#             ),
#             ti + 1,
#             1,
#         )
#         fig_estimated_mean.update(
#             layout_scene=dict(
#                 xaxis_title="Gradient 1",
#                 yaxis_title="Gradient 2",
#                 zaxis_title="Gradient 3",
#                 aspectmode="data",
#             ),
#         )
#         fig_estimated_mean.add_trace(
#             go.Scatter3d(
#                 x=FAC_TaskCentres[:, 0],
#                 y=FAC_TaskCentres[:, 1],
#                 z=FAC_TaskCentres[:, 2],
#                 marker_color=Grad_TaskCentres[:, i],
#                 marker_colorscale="RdBu",
#                 text=Tasklabels,
#                 mode="markers+text",
#                 showlegend=False,
#             ),
#             ti + 1,
#             1,
#         )
#         fig_standard_deviation.add_trace(
#             go.Volume(
#                 x=x1,
#                 y=x2,
#                 z=x3,
#                 value=y_sd,
#                 hoverinfo="skip",
#                 showlegend=False,
#                 colorscale="RdBu",
#                 # showscale=False,
#                 opacityscale=[[0, 0.8], [0.35, 0], [0.65, 0], [1, 0.8]],
#                 colorbar={
#                     "tickmode": "array",
#                     "tickvals": [min(y_sd), max(y_sd)],
#                     "ticktext": [
#                         "Low uncertainty in predicted loading",
#                         "High uncertainty in predicted loading",
#                     ],
#                 },
#                 surface_count=25,
#             ),
#             ti + 1,
#             1,
#         )
#         fig_standard_deviation.update(
#             layout_scene=dict(
#                 xaxis_title="Gradient 1",
#                 yaxis_title="Gradient 2",
#                 zaxis_title="Gradient 3",
#                 aspectmode="data",
#             ),
#         )
#         ti += 1
# fig_estimated_mean.write_html(
#     "estimated_mean_PCA_axis.html", default_width=2000, default_height=5000
# )
# fig_standard_deviation.write_html(
#     "standard_deviation_PCA_axis.html", default_width=2000, default_height=5000
# )
# # Tasklabels,Taskindices=np.unique(data.Task_name,return_inverse=True)
# # FAC_TaskCentres=np.zeros([tasknum,4])
# # for i in range(tasknum):
# #     FAC_TaskCentres[i,:]=FAC[Taskindices==i,:].mean(axis=0)
# # Grad_TaskCentres=np.zeros([tasknum,3])
# # for i in range(tasknum):
# #     Grad_TaskCentres[i,:]=GRAD[Taskindices==i,:].mean(axis=0)
# k = 4
# kernel = 1.0 * Matern(
#     length_scale=0.5, length_scale_bounds=(0.5, 1), nu=2.5
# ) + WhiteKernel(noise_level_bounds=[0.001, 0.5], noise_level=0.05)

# kf = KFold(n_splits=k, random_state=None)
# gpr = GaussianProcessRegressor(
#     kernel=kernel, random_state=None, normalize_y=True, alpha=0.1
# )


# X = FAC_TaskCentres
# scores = {}
# for PCnum in range(Grad_TaskCentres.shape[1]):
#     standardscaler = StandardScaler()
#     y = Grad_TaskCentres[:, PCnum]

#     @ignore_warnings(category=ConvergenceWarning)
#     def ___f() -> None:  # Google style docstring
#         score_gradfac, perm_scores_gradfac, pvalue_gradfac = permutation_test_score(
#             gpr,
#             X,
#             y,
#             scoring="neg_mean_absolute_error",
#             cv=kf,
#             n_permutations=1000,
#             n_jobs=1,
#         )
#         scores.update({PCnum: [score_gradfac, pvalue_gradfac]})

#     ___f()
# numloadings = 5
# for score in scores:
#     try:
#         loading = loadings[f"Component {score}"]
#         loadingpos = loading.apply(lambda x: np.abs(x)).sort_values(ascending=False)
#         tops = loading[loadingpos[:numloadings].index]
#     except Exception:
#         tops = "Loadings from SPSS are unavailable"
#     print(
#         """
#         Results for component {}:
#         Largest loadings: 
#         """.format(
#             score
#         )
#     )
#     with pd.option_context(
#         "display.max_rows",
#         5,
#         "display.max_columns",
#         None,
#         "display.width",
#         1000,
#         "display.precision",
#         3,
#         "display.colheader_justify",
#         "center",
#     ):
#         print(tops)
#     print(
#         """
#         Permutation test score: {},
#         Permutation test significance: {}
#         """.format(
#             scores[score][0], scores[score][1]
#         )
#     )
