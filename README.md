# Fatecode
Hello, here you can find the codes related to our paper (“Fatecode: Cell fate regulator prediction using classification autoencoder perturbation”). Fatecode focuses on cell fate decision making, which is a fundamental process in the development of organisms. This process involves the regulation of gene expression in cells, which ultimately determines the fate of the cell and its role in the organism. I have added comments to each file to make it easier to understand, but if you have any further questions, feel free to contact me. Thank you for your interest in our paper and the accompanying codes.

Now let me exaplain the files we have here: 
1) Different AEs:  You have a dataset and want to choose which type of autoencoder should be used. This file compares AE, VAE and CVAE (as previously has been mentioned PLEASE perfrom a  search on the number of layers, nodes and activation functions for each of the models).

2) hippo_Data: Example of using Fatecode on Hippocamous development data. (This file has the code to generate the Sankey plots. However, the plots are not used in the paper)

3) Gene_set_enrichment: This file generates gene set enrichment analysis (GSEA) on a set of genes selected based on a prioritization score.

4) hierarchy curve: This file compares the performance of a model with differential gene expression (DGE). (TP, FN, FP, Recall, precision are also computed here).

