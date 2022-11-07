import os
from multiprocessing import set_start_method
import nibabel as nib
import numpy as np
import pandas as pd
import plotly.express as px
from defs import parcel_dropout
from nilearn.datasets import fetch_atlas_schaefer_2018, fetch_atlas_yeo_2011
from nilearn.image import load_img, new_img_like, resample_to_img
from plotly.io import templates

# user_dir = os.path.expanduser("~")
# os.environ["R_HOME"] = f"{user_dir}/anaconda3/envs/gpr/Lib/R"
# os.environ["PATH"] = (
#     f"{user_dir}/anaconda3/envs/gpr/Lib/R/bin/x64;" + os.environ["PATH"]
# )
# user_dir = os.path.expanduser("~")
# os.environ["R_HOME"] = f"{user_dir}/anaconda3/envs/gpr/Lib/R"
# os.environ["PATH"] = (
#     f"{user_dir}/anaconda3/envs/gpr/Lib/R/bin/x64;" + os.environ["PATH"]
# )
# set_start_method('loky')
templates.default = "plotly_dark"
# Fit whole dataset
def load_fmris(path: str) -> dict:
    """
    Loads all the fmri images from the given path.
    Args:
        path: The path to the folder containing the fmri images.
    Returns:
        A dictionary containing the fmri images.
    """
    image_paths = [os.path.join(path, x) for x in os.listdir(path)]
    imgs = {}
    for file in image_paths:
        img = load_img(file)
        imgs[file.split("\\")[1]] = img
    return imgs


if __name__ == "__main__":
    yeo = True
    if yeo:
        yeover = "thick_7"
        yeonum = yeover.split("_")[-1]
        atlasdir = "yeo_networks"
        parcellation = load_img(fetch_atlas_yeo_2011(atlasdir)[yeover])
        namesfile = fetch_atlas_yeo_2011(atlasdir)[f"colors_{yeonum}"]
        with open(namesfile, "r") as f:
            textdata = f.read().split("\n")
        splitmaps = False
        if splitmaps:
            mdata = np.squeeze(parcellation.get_fdata())
            for i in range(int(yeonum)):
                i = i + 1
                mdata_ = np.where(np.logical_and(mdata != 0, mdata != i), 10, mdata)
                mdata_ = np.where(mdata_ == i, 100, mdata_)
                mapn = new_img_like(parcellation, mdata_)
                nib.save(mapn, f"yeo_{yeonum}_{i}.nii.gz")
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
        parcellation = fetch_atlas_schaefer_2018(n_rois=400, data_dir=atlasdir)
        parcelnames = parcellation["labels"].tolist()
        parcellation = load_img(parcellation["maps"])
    data = pd.read_csv(
        "output_.csv"
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
        subject_level=True,
        nproc=8,
    )
    PCAtoGrad = dict(PCAtoGrad)
    GradtoPCA = dict(GradtoPCA)
    PCAtoGradframe = pd.DataFrame.from_dict(
        PCAtoGrad,
        orient="index",
        columns=[
            "p-values of PCAS predicting lesioned-brain Gradients",
            "Network",
        ],
    )
    PCAtoGradframe = PCAtoGradframe.sort_values(
        "p-values of PCAS predicting lesioned-brain Gradients",
        ascending=False,
    )
    PCAtoGradframe["Lesioned Parcel"] = PCAtoGradframe.index
    GradtoPCAframe = pd.DataFrame.from_dict(
        GradtoPCA,
        orient="index",
        columns=[
            "p-values of lesioned-brain Gradients predicting PCAs",
            "Network",
        ],
    )
    GradtoPCAframe = GradtoPCAframe.sort_values(
        "p-values of lesioned-brain Gradients predicting PCAs",
        ascending=False,
    )
    GradtoPCAframe["Lesioned Parcel"] = GradtoPCAframe.index
    import plotly.express as px

    fig = px.bar(
        PCAtoGradframe,
        x="Lesioned Parcel",
        y="p-values of PCAS predicting lesioned-brain Gradients",
        color="Network",
        category_orders={
            "Lesioned Parcel": PCAtoGradframe["Lesioned Parcel"].to_list()
        },
    )
    fig.write_html("predicting gradients.html")
    fig = px.bar(
        GradtoPCAframe,
        x="Lesioned Parcel",
        y="p-values of lesioned-brain Gradients predicting PCAs",
        color="Network",
        category_orders={
            "Lesioned Parcel": GradtoPCAframe["Lesioned Parcel"].to_list()
        },
    )
    fig.write_html("predicting PCAs.html")
    print("e")
