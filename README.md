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
We decided to write the data processing as a cell in each notebook because it varies among the different datasets. This step import the desired data in the folder `data`. The training has also been launched from the notebooks to be able to see the results. Thus, each notebook corresponds to a specific eponym dataset : TreeCycle, TreeGrid and KarateClub.

