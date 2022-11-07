
import os

# user_dir = os.path.expanduser("~")
# os.environ["R_HOME"] = f"{user_dir}/anaconda3/envs/gpr/Lib/R"
# os.environ["PATH"] = (
#     f"{user_dir}/anaconda3/envs/gpr/Lib/R/bin/x64;" + os.environ["PATH"]
# )
from sklearn.decomposition import FactorAnalysis
from factor_analyzer.rotator import Rotator
# from advanced_pca import CustomPCA
from nilearn.image import resample_to_img,new_img_like
from nilearn.masking import compute_background_mask,apply_mask
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats
import pandas as pd
import math
import warnings
import numbers
import time
from functools import partial
from traceback import format_exc
from contextlib import suppress
from collections import Counter

import numpy as np
import scipy.sparse as sp
from joblib import Parallel, logger

from sklearn.base import is_classifier, clone
from sklearn.utils import indexable, check_random_state, _safe_indexing
from sklearn.utils.validation import _check_fit_params
from sklearn.utils.validation import _num_samples
from sklearn.utils.fixes import delayed
from sklearn.utils.metaestimators import _safe_split
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection._split import check_cv
from sklearn.preprocessing import LabelEncoder

from joblib import Parallel, delayed
from itertools import combinations, repeat
from multiprocessing import Manager, Pool
import numpy as np
import pandas as pd
from nilearn.masking import compute_background_mask
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

from nilearn.image import new_img_like, resample_to_img
from nilearn.masking import apply_mask
from plotly.express.colors import sample_colorscale
from plotly.graph_objects import Scatter3d, Volume
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Matern, WhiteKernel
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.utils._testing import ignore_warnings
from wordcloud import WordCloud
from typing import Tuple
from sklearn.decomposition import PCA

def permutation_test_score(
    estimator,
    X,
    y,
    *,
    groups=None,
    cv=None,
    n_permutations=100,
    n_jobs=None,
    random_state=0,
    verbose=0,
    scoring=None,
    fit_params=None,
):
    """Evaluate the significance of a cross-validated score with permutations.

    Permutes targets to generate 'randomized data' and compute the empirical
    p-value against the null hypothesis that features and targets are
    independent.

    The p-value represents the fraction of randomized data sets where the
    estimator performed as well or better than in the original data. A small
    p-value suggests that there is a real dependency between features and
    targets which has been used by the estimator to give good predictions.
    A large p-value may be due to lack of real dependency between features
    and targets or the estimator was not able to use the dependency to
    give good predictions.

    Read more in the :ref:`User Guide <permutation_test_score>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : array-like of shape at least 2D
        The data to fit.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs) or None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like of shape (n_samples,), default=None
        Labels to constrain permutation within groups, i.e. ``y`` values
        are permuted among samples with the same group identifier.
        When not specified, ``y`` values are permuted among all samples.

        When a grouped cross-validator is used, the group labels are
        also passed on to the ``split`` method of the cross-validator. The
        cross-validator uses them for grouping the samples  while splitting
        the dataset into train/test set.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - `None`, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For `int`/`None` inputs, if the estimator is a classifier and `y` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            `cv` default value if `None` changed from 3-fold to 5-fold.

    n_permutations : int, default=100
        Number of times to permute ``y``.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the cross-validated score are parallelized over the permutations.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    random_state : int, RandomState instance or None, default=0
        Pass an int for reproducible output for permutation of
        ``y`` values among samples. See :term:`Glossary <random_state>`.

    verbose : int, default=0
        The verbosity level.

    scoring : str or callable, default=None
        A single str (see :ref:`scoring_parameter`) or a callable
        (see :ref:`scoring`) to evaluate the predictions on the test set.

        If `None` the estimator's score method is used.

    fit_params : dict, default=None
        Parameters to pass to the fit method of the estimator.

        .. versionadded:: 0.24

    Returns
    -------
    score : float
        The true score without permuting targets.

    permutation_scores : array of shape (n_permutations,)
        The scores obtained for each permutations.

    pvalue : float
        The p-value, which approximates the probability that the score would
        be obtained by chance. This is calculated as:

        `(C + 1) / (n_permutations + 1)`

        Where C is the number of permutations whose score >= the true score.

        The best possible p-value is 1/(n_permutations + 1), the worst is 1.0.

    Notes
    -----
    This function implements Test 1 in:

        Ojala and Garriga. `Permutation Tests for Studying Classifier
        Performance
        <http://www.jmlr.org/papers/volume11/ojala10a/ojala10a.pdf>`_. The
        Journal of Machine Learning Research (2010) vol. 11
    """
    X, y, groups = indexable(X, y, groups)

    cv = check_cv(cv, y, classifier=is_classifier(estimator))
    scorer = check_scoring(estimator, scoring=scoring)
    random_state = check_random_state(random_state)

    # We clone the estimator to make sure that all the folds are
    # independent, and that it is pickle-able.
    score = _permutation_test_score(
        clone(estimator), X, y, groups, cv, scorer, fit_params=fit_params
    )
    permutation_scores = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_permutation_test_score)(
            clone(estimator),
            X,
            _shuffle(y, groups, random_state),
            groups,
            cv,
            scorer,
            fit_params=fit_params,
        )
        for _ in range(n_permutations)
    )
    permutation_scores = np.array(permutation_scores)
    pvalue = (np.sum(permutation_scores >= score) + 1.0) / (n_permutations + 1)
    return score, permutation_scores, pvalue


def _permutation_test_score(estimator, X, y, groups, cv, scorer, fit_params):
    """Auxiliary function for permutation_test_score"""
    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    avg_score = []
    for train, test in cv.split(X, y, groups):
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)
        fit_params = _check_fit_params(X, fit_params, train)
        estimator.fit(X_train, y_train, **fit_params)
        avg_score.append(scorer(estimator, X_test, y_test))
    return np.mean(avg_score)


def _shuffle(y, groups, random_state):
    """Return a shuffled copy of y eventually shuffle among same groups."""
    if groups is None:
        indices = random_state.permutation(len(y))
    else:
        indices = np.arange(len(groups))
        for group in np.unique(groups):
            this_mask = groups == group
            indices[this_mask] = random_state.permutation(indices[this_mask])
    return _safe_indexing(y, indices)

def varimax_(Phi, gamma = 2, q = 50, tol = 1e-6):
    """ 
    Applies varimax rotation (taken from wikipedia.)
    """
    from numpy import eye, asarray, dot, sum, diag
    from numpy.linalg import svd
    p,k = Phi.shape # gives the total number of rows and total columns of the matrix Phi
    R = eye(k) # Given a k*k identity matrix (gives 1 on diagonal and 0 elsewhere)
    d=0
    for _ in range(q):
        d_old = d
        Lambda = dot(Phi, R) # Matrix multiplication
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda)))))) # Singular value decomposition svd
        R = dot(u,vh) # construct orthogonal matrix R
        d = sum(s) #Singular value sum
        if d/d_old < tol: break
    print("Number of iterations:", _+1)
    return dot(Phi, R) # Return the rotation matrix Phi*R


def permtest(X, y, k=5, subset=False):
    """
    This function performs a permutation test on the data.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples, n_targets)
        Target vector relative to X.
    k : int, optional, default: 5
        Number of folds.
    subset : bool, optional, default: False
        If True, the permutation test is performed on each PC separately.
    Returns
    -------
    scores : dict
        Dictionary containing the scores and p-values of the permutation test.
    """
    kf = KFold(n_splits=k, random_state=None)
    gprmodel = gpr()
    score_gradfac, perm_scores_gradfac, pvalue_gradfac = permutation_test_score(
        gprmodel,
        X,
        y,
        scoring="neg_mean_absolute_error",
        cv=kf,
        n_permutations=100,
        n_jobs=-1,
    )
    scores = {"total": [score_gradfac, pvalue_gradfac]}
    if subset == False:
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
                n_permutations=100,
                n_jobs=-1,
            )
            # scores[PCnum] = [score_gradfac, pvalue_gradfac]
            #zscore_gradfac = st.norm.  pvalue_gradfac
            scores[PCnum] = [score_gradfac, pvalue_gradfac]
    return scores


class gpr(BaseEstimator):
    def __init__(self, kernel: Kernel = Matern(nu=2.5) + WhiteKernel()) -> None:  # type: ignore
        """
        This is a multi-line Google style docstring.
        Args:
            kernel: Kernel to use in the Gaussian process.
        Returns:
            None
        """
        self.kernel = kernel
        self.model = GaussianProcessRegressor(
            kernel=kernel, random_state=3, normalize_y=False, alpha=0
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        """
        Fit the model to the data.
        Args:
            X: The input data.
            y: The target data.
        Returns:
            The fitted model.
        """

        @ignore_warnings(category=ConvergenceWarning)
        def _f():
            self.model.fit(X, y)

        _f()
        return self.model

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    def score(self, X, y):
        return self.model.score(X, y)


def PCAfunc(PCAdata: pd.DataFrame, n_comp: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function performs a PCA on the dataframe passed to it.
    Args:
        PCAdata: A dataframe containing the data to be analysed.
    Returns:
        loadings: A dataframe containing the loadings of the PCA.
        PCAresults: A dataframe containing the results of the PCA.
    """
    
    chi_square_value, p_value = calculate_bartlett_sphericity(
        PCAdata
    )  # I belive this checks for correlations within our dataset which would make a PCA weird
    print(
        f"Bartlett sphericity: {chi_square_value}, p-value: {p_value}"
    )  # significant p-value means we're ok
    kmo_all, kmo_model = calculate_kmo(
        PCAdata
    )  # Not even going to pretend I understand the Kaiser-Meyer-Olkin criteria
    print(
        f"KMO test: {kmo_model}"
    )  # We want this to be > 0.6"kmo_all, kmo_model  # kmo_model > 0.6 is acceptable
    #PCAmodel = CustomPCA(n_components=n_comp, rotation="varimax")
    from scipy.stats import zscore
    PCAdata = PCAdata.apply(lambda x: zscore(x),axis=1)  # zscore the dataframe to normalise it.)
    PCAmodel = PCA(n_components=n_comp)
    rot = Rotator()
    
    PCAmodel.fit(PCAdata)
    
    loadings = PCAmodel.components_
    loadings = rot.fit_transform(PCAmodel.components_.T).T
    #lods = rot.transform(PCAmodel.components_.T)
    names = PCAdata.columns
   
    loadings = pd.DataFrame(
        np.round(loadings.T, 3),
        index=names,
        columns=[f"Component {x}" for x in range(n_comp)],
    )
    PCAresults = np.dot(PCAdata,loadings)
    #PCAresults = PCAmodel.transform(PCAdata).T
    # PCAresults = pcmo.transform(PCAdata).T
    
    #crr = stats.pearsonr(PCAresults.flatten(), PCAresultsi.T.flatten())
    return loadings, PCAresults


def averageData(
    data: pd.DataFrame, PCAresults: np.ndarray, average: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function takes the dataframe and the PCA results and returns the task labels, the FAC task centres and the gradient task centres.
    :param data: The dataframe containing the data
    :param PCAresults: The PCA results
    :param average: Whether to average the data or not
    :return: The task labels, the FAC task centres and the gradient task centres
    """
    FAC = np.asarray([PCAresults[0], PCAresults[1], PCAresults[2], PCAresults[3]]).T
    GRAD = np.asarray([data["Gradient 1"], data["Gradient 2"], data["Gradient 3"]]).T
    Tasklabels, Taskindices = np.unique(data["Task_name"], return_inverse=True)
    if average:
        tasknum = len(data["Task_name"].unique())
        FAC_TaskCentres = np.zeros([tasknum, 4])
        for i in range(tasknum):
            FAC_TaskCentres[i, :] = FAC[Taskindices == i, :].mean(axis=0)
        Grad_TaskCentres = np.zeros([tasknum, 3])
        for i in range(tasknum):
            Grad_TaskCentres[i, :] = GRAD[np.ix_(Taskindices == (i), [0, 1, 2])].mean(
                axis=0
            )
    else:
        FAC_TaskCentres = FAC
        Grad_TaskCentres = GRAD
    return Tasklabels, Taskindices, FAC_TaskCentres, Grad_TaskCentres


def wordcloud(loadings):
    """
    This function generates a wordcloud from the loadings of a topic model.
    Args:
        loadings (pandas.DataFrame): The loadings of a topic model.
    Returns:
        im (PIL.Image.Image): The wordcloud image.
    """
    unscaledloadings = loadings.sort_values(by=loadings.columns[0], ascending=False)
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



def corr(ref_img, src_img, masker=None, parcel_num=None, parcellation=None):
    """
    This function calculates the correlation between the source and reference images.
    The source and reference images are passed as dictionaries.
    The function returns a pandas dataframe with the correlation values.

    Parameters
    ----------
    ref_img : dict
        Dictionary of reference images.
    src_img : dict
        Dictionary of source images.
    masker : NiftiMasker, optional
        Masker object, by default None
    parcel_num : int, optional
        Parcel number, by default None
    parcellation : Nifti1Image, optional
        Parcellation image, by default None

    Returns
    -------
    pd.DataFrame
        Dataframe with correlation values.
    """
    results = {}
    if masker is None:
        dummy_map = src_img[next(iter(src_img))]
        dummy_map = resample_to_img(
            dummy_map, parcellation, interpolation="nearest"
        )
        masker = compute_background_mask(dummy_map)
        #NiftiMasker().fit(dummy_map).mask_img_
    # maskingimg = resample_to_img(parcellation, masker)
    maskingimg = resample_to_img(parcellation,masker, interpolation="nearest")#, interpolation="nearest")
    maskingimgdata = np.squeeze(maskingimg.get_fdata().astype(np.int32))
    if parcel_num == 0:
        
        maskingimgdata_ = np.where(maskingimgdata == 0, 0, 1)
    else:
        maskingimgdata_interm = np.where(maskingimgdata == parcel_num, 1, 0)
        maskingimgdata_ = masker.get_fdata().astype(np.int32) - maskingimgdata_interm
    
        maskingimgdata_ = np.where(maskingimgdata_ < 0, 0, maskingimgdata_)
    newimg = new_img_like(dummy_map, maskingimgdata_)

    for srcimg in src_img:
        cur_src_img = resample_to_img(src_img[srcimg], newimg)
        
        src_data = apply_mask(cur_src_img, newimg)
        
        src_name = srcimg.split(".")
        src_name = f"Gradient {src_name[2]}"
        results[src_name] = {}
        for refimg in ref_img:
            cur_ref_img = resample_to_img(ref_img[refimg], newimg)
            
            ref_data = apply_mask(cur_ref_img, newimg)
            

            ref_data = np.squeeze(
                StandardScaler().fit_transform(ref_data.reshape(-1, 1))
            )
            src_data = np.squeeze(
                StandardScaler().fit_transform(src_data.reshape(-1, 1))
            )
            # fig1 = px.histogram(x=ref_data)
            # fig1.write_html('ref_data.html')
            # fig2 = px.histogram(x=src_data)
            # fig2.write_html('src_data.html')
            correlation, pval = stats.pearsonr(ref_data, src_data)
            results[src_name][refimg.split(".")[0]] = correlation
    results = pd.DataFrame(results)
    return results


def permutation_pipeline(
    data,
    gradientMaps,
    taskMaps,
    verbose=1,
    output_figures=False,
    n_comp=4,
    subject_level=False,
    parcel_num=None,
    parcellation=None,
):
    corrs = corr(
        ref_img=taskMaps,
        src_img=gradientMaps,
        parcel_num=parcel_num,
        parcellation=parcellation,
    )
    data.update(corrs)
    data = data.reset_index(level=0)
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
  
    loadings, PCAresults = PCAfunc(PCAdata, n_comp)
    Tasklabels,Taskindices, FAC_TaskCentres, Grad_TaskCentres = averageData(data, PCAresults, average=not subject_level)

    if output_figures:
        to_html(
            X=Grad_TaskCentres,
            y=FAC_TaskCentres,
            loadings=loadings,
            Tasklabels=Tasklabels,
            path="figs",
        )
    scores = permtest(Grad_TaskCentres, FAC_TaskCentres, subset=True)
    score_dict = {"PCA->Gradients": scores}
    if verbose > 0:
        display_scores(loadings, scores, "component")
    if output_figures:
        to_html(
            X=FAC_TaskCentres,
            y=Grad_TaskCentres,
            loadings=loadings,
            Tasklabels=Tasklabels,
            path="pcafigs",
        )
    scores = permtest(FAC_TaskCentres, Grad_TaskCentres, subset=True)
    score_dict["Gradients->PCA"] = scores
    if verbose > 0:
        display_scores(loadings, scores, "gradient")
   
    return score_dict

def func_perm(
    data,
    gradientMaps,
    taskMaps,
    parcelval,
    parcellation,
    parcelnames,
    PCAtoGradscores,
    GradtoPCAscores,
    subject_level=False
):
    score_permuation = permutation_pipeline(
        data,
        gradientMaps,
        taskMaps,
        verbose=1,
        n_comp=4,
        parcel_num=parcelval,
        parcellation=parcellation,
        subject_level=subject_level
    )
    print(parcelnames[parcelval])
    try:
        PCAtoGradscores[parcelnames[parcelval].decode("utf-8")] = [
            score_permuation["PCA->Gradients"]["total"][1],
            parcelnames[parcelval].decode("utf-8").split("_")[3],
        ]
    except Exception:
        PCAtoGradscores[parcelnames[parcelval].decode("utf-8")] = [
            score_permuation["PCA->Gradients"]["total"][1],
            parcelnames[parcelval].decode("utf-8"),
        ]
    try:
        GradtoPCAscores[parcelnames[parcelval].decode("utf-8")] = [
            score_permuation["Gradients->PCA"]["total"][1],
            parcelnames[parcelval].decode("utf-8").split("_")[3],
        ]
    except Exception:
        GradtoPCAscores[parcelnames[parcelval].decode("utf-8")] = [
            score_permuation["Gradients->PCA"]["total"][1],
            parcelnames[parcelval].decode("utf-8"),
        ]
    return PCAtoGradscores, GradtoPCAscores

def corrtest(X, y, k=5, subset=False, bootstrap=False):
    """
    This function performs a permutation test on the data.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.
    y : array-like, shape (n_samples, n_targets)
        Target vector relative to X.
    k : int, optional, default: 5
        Number of folds.
    subset : bool, optional, default: False
        If True, the permutation test is performed on each PC separately.
    Returns
    -------
    scores : dict
        Dictionary containing the scores and p-values of the permutation test.
    """
    # kf = KFold(n_splits=k, random_state=None)
    # for X__,y__ in kf(X,y):
    cval = cross_val_score(gpr(), X, y, scoring="neg_mean_absolute_error", cv=k)
    # gprresults,gprstd = gpr().fit(X__,y__)
    # .predict(X,return_std=True)
    scores = {"total": [gprresults, gprstd]}
    if subset == False:
        for PCnum in range(y.shape[1]):
            y_ = y[:, PCnum]
            gprresults, gprstd = gpr().fit(X, y_).predict(X, return_std=True)
            scores[PCnum] = [gprresults, gprstd]
    return scores


def correlation_pipeline(
    data,
    gradientMaps,
    taskMaps,
    verbose=1,
    output_figures=False,
    n_comp=4,
    subject_level=False,
    parcel_num=None,
    parcellization=None,
):
    """
    This is a multi-line Google style docstring.

    Args:
        data (pandas.DataFrame): The dataframe containing the data.
        gradientMaps (nibabel.Nifti1Image): The gradient maps.
        taskMaps (nibabel.Nifti1Image): The task maps.
        verbose (int): The verbosity level.
        output_figures (bool): Whether to output figures.
        n_comp (int): The number of components.
        subject_level (bool): Whether to do subject level analysis.
        parcel_num (int): The number of parcels.
        parcellization (str): The parcellization.

    Returns:
        dict: The score dictionary.
    """
    corrs = corr(
        ref_img=taskMaps,
        src_img=gradientMaps,
        parcel_num=parcel_num,
        parcellation=parcellization,
    )
    data.update(corrs)
    data = data.reset_index(level=0)
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

    loadings, PCAresults = PCAfunc(PCAdata, n_comp)
    Tasklabels,Taskindices, FAC_TaskCentres, Grad_TaskCentres = averageData(data, PCAresults, average=not subject_level)

    if output_figures:
        to_html(
            X=Grad_TaskCentres,
            y=FAC_TaskCentres,
            loadings=loadings,
            Tasklabels=Tasklabels,
            path="figs",
        )
    scores = corrtest(Grad_TaskCentres, FAC_TaskCentres, subset=True)
    score_dict = {"PCA->Gradients": scores}
    if verbose > 0:
        display_scores(loadings, scores, "component")
    if output_figures:
        to_html(
            X=FAC_TaskCentres,
            y=Grad_TaskCentres,
            loadings=loadings,
            Tasklabels=Tasklabels,
            path="pcafigs",
        )
    scores = corrtest(FAC_TaskCentres, Grad_TaskCentres, subset=True)
    score_dict["Gradients->PCA"] = scores
    if verbose > 0:
        display_scores(loadings, scores, "gradient")

    return score_dict


def func_corr(
    data,
    gradientMaps,
    taskMaps,
    parcelval,
    parcellation,
    parcelnames,
    TRUE_PCAtoGrad,
    TRUE_GradtoPCA,
    PCAtoGradscores,
    GradtoPCAscores,
    subject_level=False
):
    score_corr = correlation_pipeline(
        data,
        gradientMaps,
        taskMaps,
        verbose=0,
        n_comp=4,
        parcel_num=parcelval,
        parcellization=parcellation,
        subject_level=subject_level
    )
    print(parcelnames[parcelval])
    scorepcag = stats.pearsonr(
        TRUE_PCAtoGrad, score_corr["PCA->Gradients"]["total"][0].flatten()
    )[0]
    scoregpca = stats.pearsonr(
        TRUE_GradtoPCA, score_corr["Gradients->PCA"]["total"][0].flatten()
    )[0]
    temp1 = [scorepcag, parcelnames[parcelval].decode("utf-8").split("_")[2]]
    temp2 = [scoregpca, parcelnames[parcelval].decode("utf-8").split("_")[2]]
    PCAtoGradscores[parcelnames[parcelval].decode("utf-8")] = temp1
    GradtoPCAscores[parcelnames[parcelval].decode("utf-8")] = temp2
    return PCAtoGradscores, GradtoPCAscores

def parcel_dropout(
    data, gradientMaps, taskMaps, parcellation, parcelnames, metric="p-val", subject_level=False,nproc=1
):
    
    number_of_parcellations = int(parcellation.get_fdata().astype(np.int32).max())+1
    PCAtoGradscores = Manager()
    PCAtoGradscores = PCAtoGradscores.dict()
    GradtoPCAscores = Manager()
    GradtoPCAscores = GradtoPCAscores.dict()
    if metric == "p-val":
        if nproc != 1:
            outout = Parallel(n_jobs=nproc)(delayed(func_perm)(
                data,
                gradientMaps,
                taskMaps,
                i,
                parcellation,
                parcelnames,
                PCAtoGradscores,
                GradtoPCAscores,
                subject_level) for i in range(number_of_parcellations))
            print('e')
            # with Pool(processes=nproc) as pool:
            #     pool.starmap(
            #         func_perm,
            #         zip(
            #             repeat(data),
            #             repeat(gradientMaps),
            #             repeat(taskMaps),
            #             range(number_of_parcellations),
            #             repeat(parcellation),
            #             repeat(parcelnames),
            #             repeat(PCAtoGradscores),
            #             repeat(GradtoPCAscores),
            #             repeat(subject_level)
            #         ),
            #     )
        else:
            for i in range(number_of_parcellations):
                func_perm(data,gradientMaps,
                        taskMaps,
                        i,
                        parcellation,
                        parcelnames,
                        PCAtoGradscores,
                        GradtoPCAscores,
                        subject_level)
    else:
        score_corr = correlation_pipeline(
            data, gradientMaps, taskMaps, verbose=0, n_comp=4, subject_level=False
        )
        TRUE_PCAtoGrad = Manager().list()
        TRUE_GradtoPCA = Manager().list()
        TRUE_PCAtoGrad.append(score_corr["PCA->Gradients"]["total"][0].flatten())
        TRUE_GradtoPCA.append(score_corr["Gradients->PCA"]["total"][0].flatten())
        PCAtoGradscores["No lesion"] = [1.0, "Whole Brain"]
        GradtoPCAscores["No lesion"] = [1.0, "Whole Brain"]
        if nproc != 1:
            with Pool(processes=nproc) as pool:
                pool.starmap(
                    func_corr,
                    zip(
                        repeat(data),
                        repeat(gradientMaps),
                        repeat(taskMaps),
                        range(number_of_parcellations),
                        repeat(parcellation),
                        repeat(parcelnames),
                        repeat(TRUE_PCAtoGrad),
                        repeat(TRUE_GradtoPCA),
                        repeat(PCAtoGradscores),
                        repeat(GradtoPCAscores),
                    ),
                )
        else:
            for i in range(number_of_parcellations):
                func_corr(data,gradientMaps,
                        taskMaps,
                        i,
                        parcellation,
                        parcelnames,
                        TRUE_PCAtoGrad,
                        TRUE_GradtoPCA,
                        PCAtoGradscores,
                        GradtoPCAscores)


    return PCAtoGradscores, GradtoPCAscores
