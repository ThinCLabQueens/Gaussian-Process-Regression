import copy
from itertools import repeat
from multiprocessing import Manager, Pool
from typing import Tuple
from plotting import wordcloud
import numpy as np
import pandas as pd
from factor_analyzer.factor_analyzer import (calculate_bartlett_sphericity,
                                             calculate_kmo)
from nilearn.image import binarize_img
from factor_analyzer.rotator import Rotator
from joblib import Parallel, delayed
from nilearn.image import new_img_like, resample_to_img
from nilearn.masking import apply_mask, compute_background_mask
from scipy import stats
from scipy.stats import zscore
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Matern, WhiteKernel
from sklearn.metrics import check_scoring
# from advanced_pca import CustomPCA
# user_dir = os.path.expanduser("~")
# os.environ["R_HOME"] = f"{user_dir}/anaconda3/envs/gpr/Lib/R"
# os.environ["PATH"] = (
#     f"{user_dir}/anaconda3/envs/gpr/Lib/R/bin/x64;" + os.environ["PATH"]
# )
from sklearn.model_selection import (KFold, LeaveOneGroupOut, cross_val_score,
                                     permutation_test_score)
from sklearn.model_selection._split import check_cv
from sklearn.preprocessing import StandardScaler
from sklearn.utils import _safe_indexing, check_random_state, indexable
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.fixes import delayed
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _check_fit_params, _num_samples
from tqdm import tqdm

from plotting import display_scores, to_html


def permutation_test_score_(
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
            _shuffle(
                y, None if cv.__class__ == LeaveOneGroupOut else groups, random_state
            ),
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


def _permutation_test_score(estimator, X, y, groups, cv, scorer, fit_params) -> float:
    """Auxiliary function for permutation_test_score"""
    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    avg_score = []
    subject_level = fit_params["subject_level"]
    for train, test in (
        cv.split(X, y, groups) if cv.__class__ == LeaveOneGroupOut else cv.split(X, y)
    ):
        X_train, y_train = _safe_split(estimator, X, y, train)
        X_test, y_test = _safe_split(estimator, X, y, test, train)
        fit_params = _check_fit_params(X, fit_params, train)
        estimator.fit(X_train, y_train, train)
        y_test = estimator.pred_PCAs(y_test, mean=subject_level)
        if not subject_level:
            y_test = y_test.mean(axis=0).reshape(1, -1)
        avg_score.append(scorer(estimator, X_test, y_test))
    return np.mean(avg_score)


def _shuffle(
    y: np.ndarray, groups: np.ndarray, random_state: np.random.RandomState
) -> np.ndarray:
    """Return a shuffled copy of y eventually shuffle among same groups."""
    if groups is None:
        indices = random_state.permutation(len(y))
    else:
        indices = np.arange(len(groups))
        for group in np.unique(groups):
            this_mask = groups == group
            indices[this_mask] = random_state.permutation(indices[this_mask])
    return _safe_indexing(y, indices)


def permtest(
    data,
    PCAdata,
    indices=None,
    n_comp=4,
    subject_level=False,
    k=5,
    subset=False,
    version=None,
    debug=False,
) -> dict:
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
    kf = LeaveOneGroupOut()
    kf = KFold(n_splits=k)
    gprmodel = gpr(subject_level=subject_level, refdata=data, version=version)
    # loadings, PCAresults = PCAfunc(PCAdata, n_comp)
    if version == "PCA":
        Tasklabels, Taskindices, X, y = averageData(
            data, PCAresults=None, average=False
        )
    else:
        Tasklabels, Taskindices, y, X = averageData(
            data, PCAresults=None, average=False
        )
    score_gradfac, perm_scores_gradfac, pvalue_gradfac = permutation_test_score_(
        gprmodel,
        X,
        y,
        groups=Taskindices,
        scoring="neg_mean_absolute_error",
        cv=kf,
        n_permutations=1000 if debug == False else 1,
        n_jobs=-1 if debug == False else 1,
        fit_params={"subject_level": subject_level},
    )
    scores = {"total": [score_gradfac, pvalue_gradfac]}
    if subset == False:
        for PCnum in range(y.shape[1] if version == "PCA" else n_comp):
            # y_ = y[:, PCnum]
            # kf = LeaveOneGroupOut()
            # kf = KFold(n_splits=k)
            gprmodel = gpr(
                subject_level=subject_level, refdata=data, version=version, PCnum=PCnum
            )
            (
                score_gradfac,
                perm_scores_gradfac,
                pvalue_gradfac,
            ) = permutation_test_score_(
                gprmodel,
                X,
                y,
                groups=Taskindices,
                scoring="neg_mean_absolute_error",
                cv=kf,
                n_permutations=1000 if debug == False else 1,
                n_jobs=-1 if debug == False else 1,
                fit_params={"subject_level": subject_level},
            )
            # scores[PCnum] = [score_gradfac, pvalue_gradfac]
            # zscore_gradfac = st.norm.  pvalue_gradfac
            scores[PCnum] = [score_gradfac, pvalue_gradfac]
    return scores


class gpr(BaseEstimator):
    def __init__(self, kernel: Kernel = Matern(nu=2.5) + WhiteKernel(), subject_level: bool = False, n_comp: int = 4, refdata=None, version=None, PCnum=None) -> None:  # type: ignore
        self.n_comp = n_comp
        self.subject_level = subject_level
        self.kernel = kernel
        self.version = version
        self.refdata = refdata
        self.PCnum = PCnum
        self.model = GaussianProcessRegressor(
            kernel=kernel, random_state=3, normalize_y=False, alpha=0
        )

    def get_params(self, deep=True):
        return (
            copy.deepcopy(
                {
                    "kernel": self.kernel,
                    "subject_level": self.subject_level,
                    "n_comp": self.n_comp,
                    "refdata": self.refdata,
                    "version": self.version,
                    "PCnum": self.PCnum,
                }
            )
            if deep
            else {
                "kernel": self.kernel,
                "subject_level": self.subject_level,
                "n_comp": self.n_comp,
                "refdata": self.refdata,
                "version": self.version,
                "PCnum": self.PCnum,
            }
        )

    def fit(self, X: np.ndarray, y: np.ndarray, train=None) -> object:
        """
        Fit the model to the data.
        Args:
            X: The input data.
            y: The target data.
        Returns:
            The fitted model.
        """
        if train != None:
            refdata = self.refdata.iloc[train]
        else:
            refdata = self.refdata
        if self.version == "PCA":
            self.loadings, PCAresults, self.PCModel = PCAfunc(X, self.n_comp, verbose=0)
            __, _, X, y = averageData(
                refdata, PCAresults, average=not self.subject_level
            )
            if self.PCnum != None:
                y = y[:, self.PCnum]
        else:
            # if self.PCnum != None:
            self.loadings, PCAresults, self.PCModel = PCAfunc(y, self.n_comp, verbose=0)
            __, _, y, X = averageData(
                refdata, PCAresults, average=not self.subject_level
            )
            if self.PCnum != None:
                y = y[:, self.PCnum]

        @ignore_warnings(category=ConvergenceWarning)
        def _f():
            self.model.fit(X, y)

        _f()
        return self.model

    def pred_PCAs(self, y: np.ndarray, **kwargs) -> np.ndarray:
        """
        This is a multi-line Google style docstring.

        Args:
            y (np.ndarray): The input data.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            np.ndarray: The output data.
        """
        if self.version == "GRAD":
            y = self.PCModel.transform(y)
            y = np.dot(y, self.loadings)
            if self.PCnum != None:
                y = y[:, self.PCnum]
        elif self.version == "PCA":
            if self.PCnum != None:
                y = y[:, self.PCnum]
        return y

    def predict(self, X, **kwargs):
        if self.version == "PCA":
            X = self.PCModel.transform(X)
            X = np.dot(X, self.loadings)
        if not self.subject_level:
            X = X.mean(axis=0).reshape(1, -1)
        return self.model.predict(X, **kwargs)

    def score(self, X, y):
        return self.model.score(X, y)

# TODO: this breaks on output_figures
def PCAfunc(
    PCAdata: pd.DataFrame, n_comp: int, verbose=1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function performs a PCA on the dataframe passed to it.
    Args:
        PCAdata: A dataframe containing the data to be analysed.
    Returns:
        loadings: A dataframe containing the loadings of the PCA.
        PCAresults: A dataframe containing the results of the PCA.
    """
    if verbose > 0:
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
    # PCAmodel = CustomPCA(n_components=n_comp, rotation="varimax")
    # PCAdata = zscore(cols as array, axis=1)
    scaler = StandardScaler()
    try:
        PCAdata_ = scaler.fit_transform(PCAdata.values)
        PCAdata = pd.DataFrame(PCAdata_, columns=PCAdata.columns, index=PCAdata.index)
        # PCAdata = PCAdata.apply(lambda x: zscore(x),axis=0)  # zscore the dataframe to normalise it.
    except:
        # PCAdata = pd.DataFrame(PCAdata).apply(lambda x: zscore(x),axis=0)
        PCAdata_ = scaler.fit_transform(PCAdata)
        PCAdata = pd.DataFrame(PCAdata_)
    PCAmodel = PCA(n_components=n_comp)
    rot = Rotator()
    PCAmodel.fit(PCAdata)
    loadings = PCAmodel.components_
    loadings = rot.fit_transform(PCAmodel.components_.T).T
    # lods = rot.transform(PCAmodel.components_.T)
    names = PCAdata.columns
    # loadings = PCAmodel.components_
    loadings = pd.DataFrame(
        np.round(loadings.T, 3),
        index=names,
        columns=[f"Component {x}" for x in range(n_comp)],
    )
    PCAresults = np.dot(PCAdata, loadings).T
    # PCAresults = PCAmodel.transform(PCAdata).T
    # PCAresults = pcmo.transform(PCAdata).T
    # crr = stats.pearsonr(PCAresults.flatten(), PCAresultsi.T.flatten())
    return loadings, PCAresults, scaler


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
    if isinstance(
        PCAresults,
        (
            np.ndarray,
            pd.DataFrame,
        ),
    ):
        FAC = np.asarray(PCAresults).T
    else:
        FAC = np.asarray(
            data.drop(
                [
                    "Participant #",
                    "Runtime_mod",
                    "Task_name",
                    "Gradient 1",
                    "Gradient 2",
                    "Gradient 3",
                ],
                axis=1,
            )
        )
    GRAD = np.asarray([data["Gradient 1"], data["Gradient 2"], data["Gradient 3"]]).T
    Tasklabels, Taskindices = np.unique(data["Task_name"], return_inverse=True)
    if average:
        tasknum = len(data["Task_name"].unique())
        FAC_TaskCentres = np.zeros([tasknum, FAC.shape[1]])
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


def corr(
    ref_img, src_img, masker=None, parcel_num=None, parcellation=None, debug=False
):
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
    if debug:
        return pd.DataFrame(
            {
                "Grad1": {"map1": 0.5, "map2": 0.6},
                "Grad2": {"map1": 0.5, "map2": 0.6},
                "Grad3": {"map1": 0.5, "map2": 0.6},
            }
        )
    results = {}
    if masker is None:
        dummy_map = src_img[next(iter(src_img))]
        if parcellation != None:
            
            dummy_map = resample_to_img(dummy_map, parcellation, interpolation="nearest")
        else:
            parcellation = binarize_img(dummy_map)
        masker = compute_background_mask(dummy_map)
    maskingimg = resample_to_img(parcellation, masker, interpolation="nearest")
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
            correlation, pval = stats.spearmanr(ref_data, src_data)
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
    parcelnames=None,
    version=None,
    subset=True,
    debug=False,
):
    """
    This function runs the permutation pipeline.

    Args:
        data (pandas.DataFrame): Dataframe containing the data.
        gradientMaps (nibabel.Nifti1Image): Nifti image containing the gradient maps.
        taskMaps (nibabel.Nifti1Image): Nifti image containing the task maps.
        verbose (int): Verbosity level.
        output_figures (bool): Whether to output figures.
        n_comp (int): Number of components to use for PCA.
        subject_level (bool): Whether to run the permutation test at the subject level.
        parcel_num (int): Parcel number to lesion.
        parcellation (nibabel.Nifti1Image): Nifti image containing the parcellation.
        parcelnames (list): List of parcel names.
        version (str): Version of the pipeline to run.
        subset (bool): Whether to run the permutation test on a subset of the data.
        debug (bool): Whether to run in debug mode.

    Returns:
        score_dict (dict): Dictionary containing the scores.
    """
    score_dict = {}
    if version == "PCA":
        corrs = corr(
            ref_img=taskMaps,
            src_img=gradientMaps,
            parcel_num=0,
            parcellation=parcellation,
            debug=debug,
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
        
        saveDummyScores(PCAdata, n_comp, data)
        if parcel_num != "No lesion":
            PCAdata = PCAdata.drop([parcel_num], axis=1)
        if output_figures:
            loadings, PCAresults, _ = PCAfunc(PCAdata, n_comp)
            Tasklabels, Taskindices, FAC_TaskCentres, Grad_TaskCentres = averageData(
                data, PCAresults, average=not subject_level
            )
            to_html(
                X=Grad_TaskCentres,
                y=FAC_TaskCentres,
                loadings=loadings,
                Tasklabels=Tasklabels,
                path="figs",
            )
        scores = permtest(
            data,
            PCAdata,
            subject_level=subject_level,
            subset=subset,
            version=version,
            debug=debug,
        )
        score_dict["PCA->Gradients"] = scores
        if verbose > 0:
            loadings, PCAresults, _ = PCAfunc(PCAdata, n_comp)
            print(f"Results for lesioning {parcel_num}")
            display_scores(loadings, scores, "component")
    elif version == "GRAD":
        corrs = corr(
            ref_img=taskMaps,
            src_img=gradientMaps,
            parcel_num=parcel_num,
            parcellation=parcellation,
            debug=debug,
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
        
        saveDummyScores(PCAdata, n_comp, data)
        if output_figures:
            loadings, PCAresults, _ = PCAfunc(PCAdata, n_comp)
            Tasklabels, Taskindices, FAC_TaskCentres, Grad_TaskCentres = averageData(
                data, PCAresults, average=not subject_level
            )
            to_html(
                X=FAC_TaskCentres,
                y=Grad_TaskCentres,
                loadings=loadings,
                Tasklabels=Tasklabels,
                path="pcafigs",
            )
        scores = permtest(
            data,
            PCAdata,
            subject_level=subject_level,
            subset=subset,
            version=version,
            debug=debug,
        )
        score_dict["Gradients->PCA"] = scores
        if verbose > 0:
            loadings, PCAresults, _ = PCAfunc(PCAdata, n_comp)
            print(f"Results for lesioning {parcelnames[parcel_num]}")
            display_scores(loadings, scores, "gradient")
    return score_dict


def saveDummyScores(PCAdata, n_comp, data):
    loadings, PCAresults, _ = PCAfunc(PCAdata, n_comp)
    expresults = pd.DataFrame(PCAresults.T,index=data.index)
    expresults[["Task_name",
            "Participant #",
            "Gradient 1",
                "Gradient 2",
                "Gradient 3"]]=data[["Task_name",
            "Participant #",
            "Gradient 1",
                "Gradient 2",
                "Gradient 3"]]
    expresults.to_csv("debug/internal_scores.csv")
    loadings.to_csv("debug/internal_loadings.csv")
    for i in loadings.columns:
        lods = loadings[i]
        img = wordcloud(lods)
        img.save(f"debug/{i}_wordcloud.png")


def func_perm(
    data,
    gradientMaps,
    taskMaps,
    parcelval,
    parcellation,
    parcelnames,
    PCAtoGradscores,
    GradtoPCAscores,
    subject_level=False,
    version=None,
    subset=True,
    output_figures=False,
    verbose=1,
    debug=False,
):
    score_permuation = permutation_pipeline(
        data,
        gradientMaps,
        taskMaps,
        verbose=verbose,
        n_comp=4,
        parcel_num=parcelval,
        parcellation=parcellation,
        subject_level=subject_level,
        version=version,
        subset=subset,
        output_figures=output_figures,
        parcelnames=parcelnames,
        debug=debug,
    )
    if version == "GRAD":
        if subset:
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
        else:
            permscores = {
                x: score_permuation["Gradients->PCA"][x][1]
                for x in score_permuation["Gradients->PCA"]
            }
            try:
                GradtoPCAscores[parcelnames[parcelval].decode("utf-8")] = [
                    permscores,
                    parcelnames[parcelval].decode("utf-8").split("_")[3],
                ]
            except Exception:
                GradtoPCAscores[parcelnames[parcelval].decode("utf-8")] = [
                    permscores,
                    parcelnames[parcelval].decode("utf-8"),
                ]
    elif version == "PCA":
        if subset:
            try:
                PCAtoGradscores[parcelval] = [
                    score_permuation["PCA->Gradients"]["total"][1],
                    parcelval.split("_")[0],
                ]
            except Exception:
                PCAtoGradscores[parcelval] = [
                    score_permuation["PCA->Gradients"]["total"][1],
                    parcelval,
                ]
        else:
            permscores = {
                x: score_permuation["PCA->Gradients"][x][1]
                for x in score_permuation["PCA->Gradients"]
            }
            try:
                PCAtoGradscores[parcelval] = [
                    permscores,
                    parcelval.split("_")[0],
                ]
            except Exception:
                PCAtoGradscores[parcelval] = [
                    permscores,
                    parcelval,
                ]


def parcel_dropout(
    data,
    gradientMaps,
    taskMaps,
    parcellation,
    parcelnames,
    metric="p-val",
    subject_level=False,
    nproc=1,
    subset=True,
    output_figures=False,
    debug=False,
    verbose=1,
):

    datacols = data.drop(
        [
            "Participant #",
            "Runtime_mod",
            "Gradient 1",
            "Gradient 2",
            "Gradient 3",
        ],
        axis=1,
    ).columns.to_list()  # Getting rid of unneeded columns for PCA
    datacols.insert(0, "No lesion")
    if parcellation != None:
        number_of_parcellations = int(parcellation.get_fdata().astype(np.int32).max()) + 1
    else:
        number_of_parcellations = 1
    PCAtoGradscores = Manager()
    PCAtoGradscores = PCAtoGradscores.dict()
    GradtoPCAscores = Manager()
    GradtoPCAscores = GradtoPCAscores.dict()
    if metric != "p-val":
        raise ValueError("Metric not recognized")
    if nproc != 1:
        print("Performing PCA predictions")
        Parallel(n_jobs=nproc)(
            delayed(func_perm)(
                data,
                gradientMaps,
                taskMaps,
                i,
                parcellation,
                parcelnames,
                PCAtoGradscores,
                GradtoPCAscores,
                subject_level,
                version="GRAD",
                subset=subset,
                output_figures=output_figures,
                verbose=verbose,
                debug=debug,
            )
            for i in range(number_of_parcellations)
        )
        print("Performing gradient predictions")
        Parallel(n_jobs=nproc)(
            delayed(func_perm)(
                data,
                gradientMaps,
                taskMaps,
                i,
                parcellation,
                parcelnames,
                PCAtoGradscores,
                GradtoPCAscores,
                subject_level,
                version="PCA",
                subset=subset,
                output_figures=output_figures,
                verbose=verbose,
                debug=debug,
            )
            for i in datacols
        )
    else:
        for i in range(number_of_parcellations):
            func_perm(
                data,
                gradientMaps,
                taskMaps,
                i,
                parcellation,
                parcelnames,
                PCAtoGradscores,
                GradtoPCAscores,
                subject_level,
                version="GRAD",
                subset=subset,
                output_figures=output_figures,
                verbose=verbose,
                debug=debug
            )
            for i in datacols:
                func_perm(
                    data,
                    gradientMaps,
                    taskMaps,
                    i,
                    parcellation,
                    parcelnames,
                    PCAtoGradscores,
                    GradtoPCAscores,
                    subject_level,
                    version="PCA",
                    subset=subset,
                    output_figures=output_figures,
                    verbose=verbose,
                    debug=debug
                )
    return PCAtoGradscores, GradtoPCAscores
