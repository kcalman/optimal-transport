# piggyback off of the demo at: 
# https://cellrank.readthedocs.io/en/latest/notebooks/tutorials/kernels/500_real_time.html

import cellrank
import scanpy as sc

# download to local folder data/ (relative to this script)
adata = cellrank.datasets.reprogramming_schiebinger('data/reprogramming_schiebinger.h5ad', subset_to_serum=False)

adata.obs["day_numerical"] = adata.obs["day"].astype(float)

# We want to get in the details of this a bit.
# Where is the data matrix ?!
sc.pl.embedding(
    adata,
    basis="force_directed",
    color=["day_numerical", "cell_sets"],
    color_map="gnuplot",
)

