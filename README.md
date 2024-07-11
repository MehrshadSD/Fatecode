# Fatecode
Hello, here you can find the codes related to our paper (“Fatecode enables cell fate regulator prediction using classification-supervised autoencoder perturbation”). Fatecode focuses on cell fate decision-making, which is a fundamental process in the development of organisms. This process involves the regulation of gene expression in cells, which ultimately determines the fate of the cell and its role in the organism. I have added comments to each file to make it easier to understand, but if you have any further questions, feel free to contact me. Thank you for your interest in our paper and the accompanying codes.

Now let me explain the files we have here: 
1) Different AEs:  You have a dataset and want to choose which type of autoencoder should be used. This file compares AE, VAE and CVAE (as previously has been mentioned PLEASE perfrom a  search on the number of layers, nodes and activation functions for each of the models).

2) hippo_Data: Example of using Fatecode on Hippocampus development data. (This file has the code to generate the Sankey plots. However, the plots are not used in the paper)

3) Gene_set_enrichment: This file generates gene set enrichment analysis (GSEA) on a set of genes selected based on a prioritization score.

4) hierarchy curve: This file compares the performance of a model with differential gene expression (DGE). (TP, FN, FP, Recall, and precision are also computed here).

- You can find the paper [here](https://www.cell.com/cell-reports-methods/fulltext/S2667-2375(24)00184-X)
- You can also find the application of Fatecode in cellular reprogramming in our recent experimental [paper](https://www.biorxiv.org/content/10.1101/2024.05.28.596294v1). In this paper, we used one of these cool plots (I think it's called a Polar plot). You can use them if you want, but they're not great when you do multiple perturbations.
- Next, I am interested in working on the connection between this [paper](https://ieeexplore.ieee.org/abstract/document/6918504) (The Potential Energy of an Autoencoder) and non-equilibrium statistical physics.  
