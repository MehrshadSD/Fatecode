# -*- coding: utf-8 -*-

from matplotlib import pyplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# from openpyxl import Workbook
# tf.compat.v1.disable_eager_execution()
# from matplotlib_venn import venn3

#%%#######################################################################
# Here we want to generate the code for Figure 3b of the paper where we compare Fatecode with DGE using the mechanistic model. If you saw the previous version, I know it was not good (too many conditionals). 
# Sorry about that. I've rewritten the code to make it better. The input file has the top candidates chosen by DGE and Fatecode for 5 different datasets. Also has the list of Master regulator genes predefined by SERGIO:

genedetector_results = [Detection.iloc[:151, i] for i in range(6)]
dge_results = [Detection.iloc[:151, i] for i in range(8, 13)]

# Compute the intersection between two lists
def intersection(lst1, lst2):
    return [value for value in lst1 if value in lst2]

def calculate_recall(master_regulators, detected_genes, column_index):
    recall = []
    listasli = []
    count = 0
    
    for i in range(len(detected_genes)):
        recall.append(count)
        gene = Detection.iloc[i, column_index]
        
        if gene in master_regulators and gene not in listasli:
            listasli.append(gene)
            count += 1
            
    return count, recall

recalls_gd = [calculate_recall(genedetector_results[0], genedetector_results[i], i) for i in range(1, 6)]
recalls_dge = [calculate_recall(genedetector_results[0], dge_results[i], i + 8) for i in range(5)]

recall_values_gd = [recall for count, recall in recalls_gd]
recall_values_dge = [recall for count, recall in recalls_dge]

def calculate_min_max_performances(recall_values):
    y_max = [max(values) for values in zip(*recall_values)]
    y_min = [min(values) for values in zip(*recall_values)]
    return y_max, y_min

y1_max, y1_min = calculate_min_max_performances(recall_values_gd)
y2_max, y2_min = calculate_min_max_performances(recall_values_dge)

# Plot the results
fig, ax = plt.subplots(figsize=(15, 10))
x_axis = range(len(genedetector_results[1]))

ax.fill_between(x_axis, y2_max, y2_min, color="green", alpha=0.4, label="DEG")
ax.fill_between(x_axis, y1_max, y1_min, color="red", alpha=0.4, label="Fatecode")

sns.lineplot(x=x_axis, y=np.add(y1_max, y1_min) / 2, color='yellow', linewidth=1.5, ax=ax)
sns.lineplot(x=x_axis, y=np.add(y2_max, y2_min) / 2, color='black', linewidth=1.5, ax=ax)

ax.set_xlabel("Number of genes included")
ax.set_ylabel("True positives")
ax.set_ylim(0, 20)
ax.set_xlim(0, 152)
ax.set_yticks(range(0, 20, 3))
ax.legend(loc='upper left')
plt.savefig("cloud.png", dpi=500, bbox_inches='tight')

fig, ax = plt.subplots(figsize=(15, 10))
sns.set(font_scale=2)
sns.set_style("white")

colors = ['red', 'green', 'blue', 'yellow', 'black']
for i, recall in enumerate(recall_values_gd):
    sns.lineplot(x=x_axis, y=recall, color=colors[i], linewidth=1.5, ax=ax)

dge_colors = ['darkred', 'olive', 'dodgerblue', 'cyan', 'orange']
for i, recall in enumerate(recall_values_dge):
    sns.lineplot(x=x_axis, y=recall, color=dge_colors[i], linewidth=1.2, linestyle='--', ax=ax)

ax.set_xlabel("Number of genes included")
ax.set_ylabel("True positives")
ax.set_ylim(0, 20)
ax.set_xlim(0, 152)
ax.set_yticks(range(0, 20, 3))
ax.legend(["Fatecode", "DEG"], loc='upper left')
plt.savefig("Compare.pdf", dpi=500, bbox_inches='tight')

def calculate_precision_recall(master_regulators, detected_genes, column_index):
    precision, recall, tp, fp, fn = [], [], [], [], []
    listasli = []
    count = 0
    
    for i in range(len(detected_genes)):
        gene = Detection.iloc[i, column_index]
        
        if gene in master_regulators and gene not in listasli:
            listasli.append(gene)
            count += 1
            
        tp.append(count)
        fp.append(i + 1 - tp[i])
        fn.append(len(master_regulators) - tp[i])
        
        precision.append(tp[i] / (tp[i] + fp[i]))
        recall.append(tp[i] / (tp[i] + fn[i]))
    
    return precision, recall

precision_recall_gd = [calculate_precision_recall(genedetector_results[0], genedetector_results[i], i) for i in range(1, 6)]
precision_recall_dge = [calculate_precision_recall(genedetector_results[0], dge_results[i], i + 8) for i in range(5)]

# Extract precision and recall values
precision_values_gd, recall_values_gd = zip(*precision_recall_gd)
precision_values_dge, recall_values_dge = zip(*precision_recall_dge)

fig, ax = plt.subplots(figsize=(15, 10))
ax.fill_between(x_axis, y2_max, y2_min, color="green", alpha=0.4, label="DEG")
ax.fill_between(x_axis, y1_max, y1_min, color="red", alpha=0.4, label="Fatecode")

sns.lineplot(x=x_axis, y=np.add(y1_max, y1_min) / 2, color='blue', linewidth=1.5, ax=ax, label="Fatecode")
sns.lineplot(x=x_axis, y=np.add(y2_max, y2_min) / 2, color='black', linewidth=1.5, ax=ax, label="DEG")

ax.set_xlabel("Precision")
ax.set_ylabel("True positives")
ax.set_ylim(0, 20)
ax.set_xlim(0, 152)
ax.set_yticks(range(0, 20, 3))
ax.legend(loc='upper left')

def compare_lists(list1, list2):
    return [1 if item in list1 else 0 for item in list2]
comparison_results = [compare_lists(Detection.iloc[:20, 0], Detection.iloc[:, i]) for i in range(1, 13)]
comparison_df = pd.concat([pd.DataFrame(result) for result in comparison_results], axis=1)



