import os
from multiprocessing import cpu_count, set_start_method
import nibabel as nib
import numpy as np
import pandas as pd

from defs import parcel_dropout
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_atlas_yeo_2011
from nilearn.image import load_img, new_img_like, resample_to_img
from plotly.io import templates

import fire

cpus = cpu_count()

templates.default = "plotly_dark"


def load_fmris(path: str) -> dict:
    """
    Loads all the fmri images from the given path.
    Args:
        path: The path to the folder containing the fmri images.
        img_type: The type of the image.
    Returns:
        A dictionary containing the fmri images.
    """
    image_paths = [os.path.join(path, x) for x in os.listdir(path)]
    imgs = {}
    for file in image_paths:
        img = load_img(file)
        imgs[file.split("\\")[1]] = img
    return imgs


def pipeline(yeo = True, splitmaps = False,yeover = "thick_7", subset = False,subject_level = False,shafer_rois=400,datapath="output.csv",n_jobs=-1,lesions=True,debug=False,verbose=1):
    import itertools
    import plotly.express as px
    if lesions:
        if yeo:
            yeover = "thick_7"  # Options: "thick_7", "thick_17", "thin_7", "thin_17"
            yeonum = yeover.split("_")[-1]
            atlasdir = "yeo_networks"
            parcellation = load_img(fetch_atlas_yeo_2011(atlasdir)[yeover])
            if splitmaps:
                mdata = np.squeeze(parcellation.get_fdata())
                for i in range(int(yeonum)):
                    i = i + 1
                    mdata_ = np.where(np.logical_and(mdata != 0, mdata != i), 10, mdata)
                    mdata_ = np.where(mdata_ == i, 100, mdata_)
                    mapn = new_img_like(parcellation, mdata_)
                    nib.save(mapn, f"yeo_{yeonum}_{i}.nii.gz")
                return 
            if yeonum == "17":
                parcelnames = [
                    b"No lesion",
                    b"VisCent",
                    b"VisPeri",
                    b"Somatomotor A",
                    b"Somatomotor B",
                    b"Dorsal Attention A",
                    b"Dorsal Attention B",
                    b"Salience/Ventral Attention A",
                    b"Salience/Ventral Attention B",
                    b"Limbic B",
                    b"Limbic A",
                    b"Control A",
                    b"Control B",
                    b"Control C",
                    b"Temporal Parietal",
                    b"Default A",
                    b"Default B",
                    b"Default C",
                ]
            elif yeonum == "7":
                parcelnames = [
                    b"No lesion",
                    b"Visual",
                    b"Somatomotor",
                    b"Dorsal Attention",
                    b"Ventral Attention",
                    b"Limbic",
                    b"Frontoparietal",
                    b"Default",
                ]
        else:
            atlasdir = "shaefer_atlas"
            parcellation = fetch_atlas_schaefer_2018(n_rois=shafer_rois, data_dir=atlasdir)
            parcelnames = parcellation["labels"].tolist()
            parcellation = load_img(parcellation["maps"])
    else:
        parcellation = None
        parcelnames = [
                    b"No lesion"]
    data = pd.read_csv(
        datapath
    )  # Reading datafile (should be in the same directory as our IDE)
    
    data = data.set_index("Task_name")
    gradientMaps = load_fmris("gradients")
    taskMaps = load_fmris("taskmaps")
    
    PCAtoGrad, GradtoPCA = parcel_dropout(
        data,
        gradientMaps,
        taskMaps,
        parcellation,
        parcelnames,
        metric="p-val",
        subject_level=subject_level,
        nproc=(cpus - 2 if n_jobs == -1 else n_jobs) ,
        subset=subset,
        debug=debug,
        verbose=verbose
    )
    PCAtoGrad = dict(PCAtoGrad)
    GradtoPCA = dict(GradtoPCA)
    if not subset:
        

        mapping = {y: PCAtoGrad[y][1] for y in PCAtoGrad}
        d = {y: PCAtoGrad[y][0] for y in PCAtoGrad}
        PCAtoGrad = dict(
            [
                (x, dict([(k, d[k][x]) for k in d if x in d[k]]))
                for x in set(itertools.chain(*list(d.values())))
            ]
        )
        mappingframe = pd.DataFrame.from_dict(mapping, orient="index")
    for key in PCAtoGrad:
        PCAtoGradframe = pd.DataFrame.from_dict(
            PCAtoGrad[key],
            orient="index",
            columns=[
                "p-values of PCAS predicting lesioned-brain Gradients",
            ],
        )
        PCAtoGradframe["Network"] = mappingframe
        PCAtoGradframe = PCAtoGradframe.sort_values(
            "p-values of PCAS predicting lesioned-brain Gradients",
            ascending=False,
        )
        PCAtoGradframe["Lesioned Parcel"] = PCAtoGradframe.index
        fig = px.bar(
            PCAtoGradframe,
            x="Lesioned Parcel",
            y="p-values of PCAS predicting lesioned-brain Gradients",
            color="Network",
            category_orders={
                "Lesioned Parcel": PCAtoGradframe["Lesioned Parcel"].to_list()
            },
        )
        try:
            fig.write_html(f"figures/predicting gradient {int(key)+1}.html")
        except:
            fig.write_html("figures/predicting all gradients.html")
    if not subset:
        import itertools

        mapping = {y: GradtoPCA[y][1] for y in GradtoPCA}
        d = {y: GradtoPCA[y][0] for y in GradtoPCA}
        GradtoPCA = dict(
            [
                (x, dict([(k, d[k][x]) for k in d if x in d[k]]))
                for x in set(itertools.chain(*list(d.values())))
            ]
        )
        mappingframe = pd.DataFrame.from_dict(mapping, orient="index")
    for key in GradtoPCA:
        GradtoPCAframe = pd.DataFrame.from_dict(
            GradtoPCA[key],
            orient="index",
            columns=[
                "p-values of lesioned-brain Gradients predicting PCAs",
            ],
        )
        GradtoPCAframe["Network"] = mappingframe
        GradtoPCAframe = GradtoPCAframe.sort_values(
            "p-values of lesioned-brain Gradients predicting PCAs",
            ascending=False,
        )
        GradtoPCAframe["Lesioned Parcel"] = GradtoPCAframe.index
        import plotly.express as px

        fig = px.bar(
            GradtoPCAframe,
            x="Lesioned Parcel",
            y="p-values of lesioned-brain Gradients predicting PCAs",
            color="Network",
            category_orders={
                "Lesioned Parcel": GradtoPCAframe["Lesioned Parcel"].to_list()
            },
        )
        try:
            fig.write_html(f"figures/predicting PCA number {int(key) + 1}.html")
        except:
            fig.write_html("figures/predicting all PCAs.html")
            
if __name__ == '__main__':
    pipeline(debug=False,verbose=-1)
    #fire.Fire(pipeline)
