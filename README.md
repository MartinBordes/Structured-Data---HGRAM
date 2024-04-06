# Structured Data Project
Clément Teulier, Martin Bordes

## Requirements

You first need to install the desired requirements:

```bash
pip install -r requirements.txt
```
Then, use conda to install a dgl version enabling a GPU usage:

```bash
conda install -c dglteam/label/cu121 dgl
```

## Run
We decided to write the data processing as a cell in each notebook because it varies among the different datasets. This step imports the desired data in the folder `data\`. The training has also been launched from the notebooks to be able to see the results. Thus, each notebook corresponds to a specific eponym dataset : `TreeCycle.ipynb`, `TreeGrid.ipynb` and `KarateClub.ipynb`.

We had to adapt to the original data processing file (`node_process.py`) and slightly modify the `meta.py` script.

