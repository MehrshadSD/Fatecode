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
# Here we want to generate the code for the figure 3b of the paper where we compare fatecode with DGE using mechanistic model.
# The Detection.CSV file has top candidates chosen by DGE and Fatecode for 5 different datasets. Also has the list of Master regulator genes predefined by SERGIO:
Detection=pd.read_csv("Detection.csv")

# These are Genedetector results
A=list(Detection.iloc[:151,0])
B=list(Detection.iloc[:151,1])
C=list(Detection.iloc[:151,2])
D=list(Detection.iloc[:151,3])
E=list(Detection.iloc[:151,4])
F=list(Detection.iloc[:151,5])

# These are DGE results
A_N=list(Detection.iloc[:151,8])
B_N=list(Detection.iloc[:151,9])
C_N=list(Detection.iloc[:151,10])
D_N=list(Detection.iloc[:151,11])
E_N=list(Detection.iloc[:151,12])


#Computing intersection between each method and master regulators
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

# DO you have a MR? 
def asli(A,B,k):
    Precision=[]
    Recall=[]
    listasli=[]
    count=0
    for i in range(0,len(B)):
        # C=intersection(A,B[1:i])
        Recall.append(count)
        if np.any(Detection.iloc[i,k]==(Detection.iloc[:,0])):
            if np.any(Detection.iloc[i,k]==listasli):
                print("NOOOOOO")
            else:
                listasli.append(Detection.iloc[i,k])
                count=count+1
    return count, Recall


count, Recall = asli(A,B,1)
count2, Recall2 = asli(A,C,2)
count3, Recall3 = asli(A,D,3)
count4, Recall4= asli(A,E,4)
count5, Recall5= asli(A,F,5)


countN, Recalln = asli(A,A_N,8)
count2N, Recall2n = asli(A,C_N,9)
count3N, Recall3n = asli(A,D_N,10)
count4N, Recall4n= asli(A,E_N,11)
count5N, Recall5n= asli(A,B_N,12)


# To plot the min and max performances of each method using 5 generated data by SERGIO.
list1 = [Recall, Recall2, Recall3, Recall4, Recall5]
Y1max = [max(elem) for elem in zip(*list1)]
Y1min = [min(elem) for elem in zip(*list1)]


list2 = [Recalln, Recall2n, Recall3n, Recall4n, Recall5n]
Y2max = [max(elem) for elem in zip(*list2)]
Y2min = [min(elem) for elem in zip(*list2)]








fig, ax = plt.subplots(figsize=(15, 10))
ax.fill_between(range(0,len(B)), Y2max, Y2min, color="green", alpha=0.4,label="DEG")
ax.fill_between(range(0,len(B)), Y1max, Y1min, color="red", alpha=0.4, label="Fatecode")
ax=sns.lineplot(range(0,len(E)), np.add(Y1max, Y1min)/2, color='yellow', linewidth=1.5)
ax=sns.lineplot(range(0,len(E)), np.add(Y2max, Y2min)/2, color='black', linewidth=1.5)
pyplot.xlabel("Number of genes included")
pyplot.ylabel("True postivies")
ax.set(ylim=(0, 20))
ax.set(xlim=(0, 152))
plt.legend(loc='upper left')
ax.set_yticks(range(0,20,3))
# pyplot.savefig("cloud.png",dpi = 500, bbox_inches = 'tight')



X_axis=range(0,len(B))     
pyplot.figure(figsize=(15, 10)) # width and height in inches 
sns.set(font_scale = 2)
sns.set_style("white")
ax=sns.lineplot(range(0,len(B)), Recall, color='red', linewidth=1.5)
ax=sns.lineplot(range(0,len(C)), Recall2, color='green', linewidth=1.5)
ax=sns.lineplot(range(0,len(D)), Recall3, color='blue', linewidth=1.5)
ax=sns.lineplot(range(0,len(E)), Recall4, color='yellow', linewidth=1.5)
ax=sns.lineplot(range(0,len(F)), Recall5, color='black', linewidth=1.5, label="Fatecode")

ax=sns.lineplot(range(0,len(A_N)), Recalln, color='darkred', linewidth=1.2, linestyle='--')
ax=sns.lineplot(range(0,len(C_N)), Recall2n, color='olive', linewidth=1.2, linestyle='--')
ax=sns.lineplot(range(0,len(D_N)), Recall3n, color='dodgerblue', linewidth=1.2, linestyle='--')
ax=sns.lineplot(range(0,len(E_N)), Recall4n, color='cyan', linewidth=1.2, linestyle='--')
ax=sns.lineplot(range(0,len(B_N)), Recall5n, color='orange', linewidth=1.2, linestyle='--',label="DEG")

pyplot.xlabel("Number of genes included")
pyplot.ylabel("True postivies")
ax.set(ylim=(0, 20))
ax.set(xlim=(0, 152))
ax.set_yticks(range(0,20,3))
# ax.legend(['label 1', 'label 2', 'label 3'])
# pyplot.savefig("Compare.pdf",dpi = 500, bbox_inches = 'tight')




#%%% If you are interested in looking at precision-recall plots. Check this part out:
def asli2(A,B,k):
    Precision=[]
    Recall = []
    TP=[]
    FP = []
    FN = []
    listasli=[]
    count=0
    for i in range(0,len(B)):
        # C=intersection(A,B[1:i])
        if np.any(Detection.iloc[i,k]==(Detection.iloc[:,0])):
            if np.any(Detection.iloc[i,k]==listasli):
                print("na")

            else:
                listasli.append(Detection.iloc[i,k])
                count=count+1    
            
        TP.append(count)
        FP.append(i+1-TP[i])
        FN.append(len(Detection.iloc[:20,0])-TP[i])

        Precision.append(TP[i]/(TP[i]+FP[i]))
        Recall.append(TP[i]/(TP[i]+FN[i]))
    
    return Precision, Recall


# To plot the min and max performances of each method using 5 generated data by SERGIO.
list2 = [countN, count2N, count3N, count4N, count5N]
Y2max = [max(elem) for elem in zip(*list2)]
Y2min = [min(elem) for elem in zip(*list2)]


list1 = [count, count2, count3, count4, count5]
Y1max = [max(elem) for elem in zip(*list1)]
Y1min = [min(elem) for elem in zip(*list1)]


np.add(Y1max, Y1min)/2


fig, ax = plt.subplots(figsize=(15, 10))
ax.fill_between(range(0,len(B)), Y2max, Y2min, color="green", alpha=0.4,label="DEG")
ax.fill_between(range(0,len(B)), Y1max, Y1min, color="red", alpha=0.4, label="Fatecode")

plt.plot(range(0,len(E)), np.add(Y1max, Y1min)/2, color='blue', linewidth=1.5, label= "Fatecode")
plt.plot(range(0,len(E)), np.add(Y2max, Y2min)/2, color='black', linewidth=1.5, label= "DEG")
pyplot.xlabel("Precision")
pyplot.ylabel("True postivies")
ax.set(ylim=(0, 20))
ax.set(xlim=(0, 152))
plt.legend(loc='upper left')
ax.set_yticks(range(0,20,3))


# Precision, Recall, TP, FP, FN = asli2(A,C,2)
count, Recall = asli2(A,C,2)
count2, Recall2 = asli2(A,C,2)
count3, Recall3 = asli2(A,D,3)
count4, Recall4= asli2(A,E,4)
count5, Recall5= asli2(A,F,5)

countN, Recalln = asli2(A,A_N,8)
count2N, Recall2n = asli2(A,C_N,9)
count3N, Recall3n = asli2(A,D_N,10)
count4N, Recall4n= asli2(A,E_N,11)
count5N, Recall5n= asli2(A,B_N,12)


plt.plot(range(0,len(E)), np.add(Y2max, Y2min)/2, color='black', linewidth=1.5, label= "DEG")
plt.plot(range(0,len(E)), np.add(Y1max, Y1min)/2, color='blue', linewidth=1.5, label="Fatecode")
plt.gca().legend(("DEG", "Fatecode"))




# Check if a float in one list exist in another list
def compare3(A,B):
    C=[]
    for i in range(0,len(B)):
        if np.any(A==B[i]):
            C.append(1)
        else:
            C.append(0)
    return C

A_B = compare3(Detection.iloc[:20,0],Detection.iloc[:,1])
A_C= compare3(Detection.iloc[:20,0],Detection.iloc[:,2])
A_D= compare3(Detection.iloc[:20,0],Detection.iloc[:,3])
A_E= compare3(Detection.iloc[:20,0],Detection.iloc[:,4])
A_F= compare3(Detection.iloc[:20,0],Detection.iloc[:,5])

A_A_N = compare3(Detection.iloc[:20,0],Detection.iloc[:,8])
A_C_N = compare3(Detection.iloc[:20,0],Detection.iloc[:,9])
A_D_N = compare3(Detection.iloc[:20,0],Detection.iloc[:,10])
A_E_N= compare3(Detection.iloc[:20,0],Detection.iloc[:,11])
A_B_N= compare3(Detection.iloc[:20,0],Detection.iloc[:,12])

# Make a dataframe with the results of the comparison
def compare4(A,B,C,D,E,F,G,H,I,J):
    A = pd.DataFrame(A)
    B = pd.DataFrame(B)
    C = pd.DataFrame(C)
    D = pd.DataFrame(D)
    E = pd.DataFrame(E)
    F = pd.DataFrame(F)
    G = pd.DataFrame(G)
    H = pd.DataFrame(H)
    I = pd.DataFrame(I)
    J = pd.DataFrame(J)
    result = pd.concat([A,B,C,D,E,F,G,H,I,J], axis=1)
    return result

result = compare4(A_B,A_C,A_D,A_E,A_F,A_A_N,A_C_N,A_D_N,A_E_N,A_B_N)



