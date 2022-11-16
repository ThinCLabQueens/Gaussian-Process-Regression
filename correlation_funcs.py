
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
    data, gradientMaps, taskMaps, parcellation, parcelnames, metric="p-val", subject_level=False,nproc=1,subset=True
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
    datacols.insert(0,"No lesion")
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
                subject_level,
                version="GRAD",
                subset=subset) for i in range(number_of_parcellations))
            outout1 = Parallel(n_jobs=nproc)(delayed(func_perm)(
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
                subset=subset) for i in datacols)
            print(outout,outout1)    
            
        else:
            for i in range(number_of_parcellations):
                func_perm(data,gradientMaps,
                        taskMaps,
                        i,
                        parcellation,
                        parcelnames,
                        PCAtoGradscores,
                        GradtoPCAscores,
                        subject_level,
                        version="GRAD",
                        subset=subset)
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
                    subset=subset)
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