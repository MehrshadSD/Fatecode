# -*- coding: utf-8 -*-

import gseapy
import csv
import numpy as np
import pandas as pd
import gseapy as gp
import matplotlib.pyplot as plt
import seaborn as sns
from gseapy.plot import barplot, dotplot


names = gseapy.get_library_name()
print(names)


# After running Fatecode and detecting the important genes for the process you can save them as a CSV file and run the code below: 
# Here we looked at the 100 genes selected by the prioritization scores for a specific cell type:

df = pd.read_csv('hippo.csv',index_col=0)
up_1d=df.iloc[0:100,].index.tolist()
# down_1d=df.iloc[-100:,].index.tolist()



def enrichmenty(x):
    enr_GOBP_up = gp.enrichr(gene_list=x ,
     gene_sets=['GO_Biological_Process_2021'],
     organism='Mouse', 
     outdir='test/_GOBP_up',
     cutoff=0.5)
    
    enr_GOMF_up = gp.enrichr(gene_list=x ,
     gene_sets=['GO_Molecular_Function_2021'],
     organism='Mouse', 
     outdir='test/GOMF_up',
     cutoff=0.5)
    
    enr_GOCC_up = gp.enrichr(gene_list=x ,
     gene_sets=['GO_Cellular_Component_2021'],
     organism='Mouse', 
     outdir='test/GOCC_up',
     cutoff=0.5)
    
    enr_Reactome_up = gp.enrichr(gene_list=x ,
     gene_sets=['Reactome_2016'],
     organism='Mouse', 
     outdir='test/Reactome_up',
     cutoff=0.5)
    return(enr_GOBP_up,enr_GOMF_up,enr_GOCC_up,enr_Reactome_up)



x,y,z,k=enrichmenty(up_1d)
# xx,yy,zz,kk=enrichmenty(down_1d)


# enr_GOBP_up.results.head(5)
barplot(x.res2d.iloc[:20,],title='GO Biological Processes',color = 'r', top_term=15, ofname="GBP.pdf",dpi=500)
barplot(y.res2d.iloc[:20,],title='GO Molecular Function',color = 'r', top_term=15, ofname="GMF.pdf",dpi=500)
barplot(z.res2d.iloc[:20,],title='Enrichment for Reactome pathways',color = 'r', top_term=15, ofname="REACT.pdf",dpi=500)

barplot(x.res2d.iloc[:20,],title='GO Biological Processes',color = 'r', top_term=15)
barplot(y.res2d.iloc[:20,],title='GO Molecular Function',color = 'r', top_term=15)
barplot(z.res2d.iloc[:20,],title='Enrichment for Reactome pathways',color = 'r', top_term=15)