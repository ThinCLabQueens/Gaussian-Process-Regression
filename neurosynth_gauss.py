import os
from pprint import pprint

from nimare.extract import download_abstracts, fetch_neuroquery, fetch_neurosynth
from nimare.io import convert_neurosynth_to_dataset

out_dir = "nsynth"
os.makedirs(out_dir, exist_ok=True)

files = fetch_neurosynth(
    data_dir=out_dir,
    version="7",
    #overwrite=False,
    #source="abstract",
    #vocab="terms",
)
# Note that the files are saved to a new folder within "out_dir" named "neurosynth".
pprint(files)
neurosynth_db = files[0]

neurosynth_dset = convert_neurosynth_to_dataset(
    coordinates_file=neurosynth_db["coordinates"],
    metadata_file=neurosynth_db["metadata"],
    annotations_files=neurosynth_db["features"],
)
neurosynth_dset.save(os.path.join(out_dir, "neurosynth_dataset.pkl.gz"))
print(neurosynth_dset)

neurosynth_dset = download_abstracts(neurosynth_dset, "goodallhalliwell.i@queensu.ca")
neurosynth_dset.save(os.path.join(out_dir, "neurosynth_dataset_with_abstracts.pkl.gz"))

files = fetch_neuroquery(
    data_dir=out_dir,
    version="1",
    overwrite=False,
    source="combined",
    vocab="neuroquery6308",
    type="tfidf",
)
# Note that the files are saved to a new folder within "out_dir" named "neuroquery".
pprint(files)
neuroquery_db = files[0]

# Note that the conversion function says "neurosynth".
# This is just for backwards compatibility.
neuroquery_dset = convert_neurosynth_to_dataset(
    coordinates_file=neuroquery_db["coordinates"],
    metadata_file=neuroquery_db["metadata"],
    annotations_files=neuroquery_db["features"],
)
neuroquery_dset.save(os.path.join(out_dir, "neuroquery_dataset.pkl.gz"))
print(neuroquery_dset)

# NeuroQuery also uses PMIDs as study IDs.
neuroquery_dset = download_abstracts(neuroquery_dset, "goodallhalliwell.i@queensu.ca")
neuroquery_dset.save(os.path.join(out_dir, "neuroquery_dataset_with_abstracts.pkl.gz"))