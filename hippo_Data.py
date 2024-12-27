

#%%
import scvelo as scv
import pandas as pd
import numpy as np

#%% CHECK THE DIFFERENT AEs file to chooose the right arch and more details of the method.
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import random 
import random as rn
np.random.seed(42)
rn.seed(1254)


# For the Hippocampus data you can download the file from GEO or use scv to read it as an object. After the preprocessing step and choosing the right model and architecture.
adata = scv.datasets.dentategyrus()
scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
scv.tl.velocity(adata)
scv.tl.velocity_graph(adata)

cols=pd.DataFrame(adata.var)
cols=cols.index
X=pd.DataFrame.sparse.from_spmatrix(adata.X, columns=cols)
Y=pd.DataFrame(adata.obs.clusters)
X.index=Y.index

classes, counts = np.unique(Y, return_counts=True)
# Encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encodedY = encoder.transform(Y)
dummyY = to_categorical(encodedY)
dummyY=pd.DataFrame(dummyY, index=Y.index,columns=[classes])
############################################## number of input columns #####################################################
# This part is for those who dont want to choose between different AEs and they already know what type of autoencoder they want to use. 
# Here for this dataset we are using AE:

n_inputs = X.shape[1]
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, dummyY, latent_dim = round(float(n_inputs) / 150), test_size=0.15, random_state=1)

def build_two_layer_network_model(X):
    n_inputs = X.shape[1]
    # define encoder
    visible = Input(shape=(n_inputs,))
    # encoder level 1
    e = Dense(round(float(n_inputs) / 40))(visible)
    e1 = BatchNormalization()(e)
    e2 = LeakyReLU()(e1)
    # encoder level 2
    e3 = Dense(round(float(n_inputs) / 90))(e2)
    e4 = BatchNormalization()(e3)
    e5 = LeakyReLU()(e4)
    # bottleneck
    n_bottleneck = latent_dim
    bottleneck = Dense(n_bottleneck)(e5)
    encoder = Model(visible, bottleneck, name="encoder")
    ppp = latent_dim
    visible2 = Input(shape=(ppp,))
    # define decoder, level 1
    d = Dense(round(float(n_inputs) / 90))(visible2)
    d1 = BatchNormalization()(d)
    d2 = LeakyReLU()(d1)
    # decoder level 2
    d3 = Dense(round(float(n_inputs) / 40))(d2)
    d4 = BatchNormalization()(d3)
    d5 = LeakyReLU()(d4)
    # output layer
    output = Dense(n_inputs, activation='linear')(d5)
    decoder = Model(visible2, output, name="decoder")

    encoded = encoder(visible)
    decoded = decoder(encoded)
    
    class1=Dense(25,activation='relu')(visible2)
    class2=Dense(14,activation='softmax')(class1)
    Classifier = Model(visible2, class2, name="Classifier")
    Classifiered=Classifier(encoded)

    model = Model(inputs=visible, outputs=[decoded,Classifiered])
    # compile autoencoder model
    
    model.compile(optimizer='adam', loss=['mse', 'categorical_crossentropy'])
    return model,encoder,decoder,Classifier
    # return model,decoder,encoder

model,encoder,decoder,Classifier=build_two_layer_network_model(X)

# fit the autoencoder model to reconstruct input.
history = model.fit(X_train, [X_train,y_train], epochs=100, batch_size=500, verbose=2, validation_data=(X_test,[X_test,y_test]))
encoder.summary()
decoder.summary()
Classifier.summary()
model.summary()


yhat = model.predict(X_test)
class_labels = np.argmax(yhat[1], axis=1)
y_test = np.argmax(np.array(y_test), axis=1)
cm = confusion_matrix(class_labels, y_test)
fig, ax= plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot=True, fmt='g', ax=ax, xticklabels=['Astrocytes','Cajal Retzius','Cck-Tox','Endothelial','GABA', 'Granule immature','Granule mature', 'Microglia','Mossy', 'Neuroblast', 'OL', 'OPC', 'Radial Glia-like', 'nIPC'], 
            yticklabels=['Astrocytes','Cajal Retzius','Cck-Tox','Endothelial','GABA', 'Granule immature','Granule mature', 'Microglia','Mossy', 'Neuroblast', 'OL', 'OPC', 'Radial Glia-like', 'nIPC'])


# Make sure your loss function is deacreasing. YOU DONT WANT TO OVERFIT. Also if interested you can try early stopping and drop out 
plt.figure(figsize=(12,8))
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
# save the encoder to file
encoder.save('encoder.h5')

# encode the train data
X_train_encode = encoder.predict(X_train)
# encode the test data
X_test_encode = encoder.predict(X_test)
# encode all of the data
X_encode = encoder.predict(X)

#%% ###################### Perturbation #################################
# After training the model and making sure its not overfitting first we perturb diffrent nodes of the latent layer with different values (can be single or multiple perturbations for this paper we just tried single node perturbation). 
# After each perturbation we look at the distribution and frequency changes of cell types and choose the node pertuabtion that leads to the type of distribution we want. 
# For example imagine for our biological process we want to have more Microglia cells than Neuroblast. By changing different nodes of the latent layer we find out which node leads to an increase in Microglia and decrease in Neuroblast.
# The parameter table has the information of changes after each perturbation.

X_encode = encoder.predict(X)
khuruji = decoder.predict(X_encode)
khuruji=pd.DataFrame(khuruji, index=X.index, columns=X.columns)
pl=[]
empty=[]
lll=0
for i in range(0,X_encode.shape[1]):
    for j in range(1,21):
        data= np.copy(X_encode)
        data[:,i] = data[:,i] * j * 0.5
        empty += [data]
        lll=lll+1
    #acc.append(data[:,i] * j * 0.5)
for i in range(0,lll):     
    yhatt = Classifier.predict(empty[i])
    class_labels = np.argmax(yhatt, axis=1)
    pl.append(class_labels)
    
# Computing the accuracy of different perturbation   
acc=[]
for k in range(0,lll):
    acc.append(accuracy_score(encodedY, pl[k]))
    
# Plot this 
plt.figure(figsize=(12,8))
plt.plot(list(range(0,lll)), acc, label = 'machine learning')
plt.ylabel('accuracy')
plt.xlabel('perturbation sets')


table=pd.DataFrame()
for i in range(0,lll):
    table=table.append(pd.DataFrame(pl[i]).value_counts(sort=False) ,ignore_index=True)    
table.columns=['Astrocytes','Cajal Retzius','Cck-Tox','Endothelial','GABA', 'Granule immature','Granule mature', 'Microglia','Mossy', 'Neuroblast', 'OL', 'OPC', 'Radial Glia-like', 'nIPC']
table


#%%%
# To understand what genes are involved in this process we perform the following operation on the latent layer. By substracting the perturbed from the unperturbed node of the latent layer and feeding it to 
# the decoder we can see the score for each gene in each cell type. By taking the average of prioritization score of the each cell in each celltype, a score for each specific celltype can be achieved. 

lat_sub = empty[39] - X_encode 
lat_sub_decod = decoder.predict(lat_sub)
yhatt = Classifier.predict(empty[39])
class_labels = np.argmax(yhatt, axis=1)
class_labels = pd.DataFrame(class_labels, index=Y.index)
lat_sub_decod = pd.DataFrame(lat_sub_decod, index=X.index, columns=X.columns)
result = pd.concat([lat_sub_decod, class_labels], axis=1)
result = result.rename(columns={0: 'groups'})

average=[]
for i in range(0,14):
    rslt_df = result[result['groups'] == i] 
    average.append(rslt_df.mean())
    
making = {'Astrocytes':average[0],'Cajal Retzius':average[1],'Cck-Tox':average[2],'Endothelial':average[3],'GABA':average[4], 'Granule immature':average[5],'Granule mature':average[6], 
          'Microglia':average[7],'Mossy':average[8], 'Neuroblast':average[9], 'OL':average[10], 'OPC':average[11], 'Radial Glia-like':average[12], 'nIPC':average[13]}
making=pd.DataFrame(making ,columns=['Astrocytes','Cajal Retzius','Cck-Tox','Endothelial','GABA', 'Granule immature','Granule mature', 'Microglia','Mossy', 'Neuroblast', 'OL', 'OPC', 'Radial Glia-like', 'nIPC'])
avg_pos_making = abs(making).mean(axis=1) 
making=pd.concat([making, avg_pos_making], axis=1)
making=making.drop(columns=0,axis=1)
making.to_csv("making.csv")


# Plot the heatmap of the scores of the genes for each celltype
plt.figure(figsize=(5,20))
sns.set(font_scale=1)
# sns.clustermap(making,cmap="vlag", vmin=-2, vmax=2)
sns.heatmap(making,cmap="vlag", vmin=-2, vmax=2)
plt.gcf().set_size_inches(5, 12)
plt.yticks(fontsize=7)
plt.savefig("heatmap_hippo.pdf")
#%%########################################## Trend and Sankey plot  ###########################################
# This plot shows how much each celltype changes as we perturb a specific node. For example node number 5 is pertubed with different values this plot can help you 
# to see the effect (Just to let you know the effect is not linear)

ant=table[20:40]  #Perturbing node 2 with 20 different values
plt.figure()
ant['index']=np.linspace(0.5,10,20)
fig, ax = plt.subplots(figsize=(25, 15))
# multiple line plots
plt.plot( 'index', 'Astrocytes', data=ant, marker='o', markerfacecolor='b', markersize=5, color='lawngreen', linewidth=4, label="Astrocytes")
plt.plot( 'index', 'Cajal Retzius', data=ant, marker='', color='g', linewidth=2, label="Cajal Retzius")
plt.plot( 'index', "Cck-Tox", data=ant, marker='', color='r', linewidth=2, linestyle='-', label="Cck-Tox")
plt.plot( 'index', "Endothelial", data=ant, marker='', color='c', linewidth=2, linestyle='--', label="GABA")
plt.plot( 'index', "GABA", data=ant, marker='', color='m', linewidth=2, linestyle='-.', label="Lymphoid")
plt.plot( 'index', 'Granule immature', data=ant, marker='o', markerfacecolor='y', markersize=5, color='skyblue', linewidth=4,label="Granule immature")
plt.plot( 'index', 'Granule mature', data=ant, marker='', color='k', linewidth=4, label="Granule mature")
plt.plot( 'index', "Microglia", data=ant, marker='', color='#4b0082', linewidth=2, linestyle='-', label="Microglia")
plt.plot( 'index', "Mossy", data=ant, marker='', color='b', linewidth=2, linestyle='solid', label="Mossy")
plt.plot( 'index', "Neuroblast", data=ant, marker='', color='darkorange', linewidth=3, linestyle='dotted', label="Neuroblast")
plt.plot( 'index', "OL", data=ant, marker='', color='r', linewidth=4, linestyle='dotted', label="OL")
plt.plot( 'index', "OPC", data=ant, marker='o', color='c', linewidth=4, linestyle=':', label="OPC")
plt.plot( 'index', "Radial Glia-like", data=ant, marker='', color='#8A2BE2', linewidth=4, linestyle='dotted', label="Radial Glia-like")
plt.plot( 'index', "nIPC", data=ant, marker='o', color='#66CDAA', linewidth=4, linestyle=':', label="nIPC")
# show legend
plt.xticks(np.linspace(0.5,10,20), fontsize=15)
# plt.ylim([-10, 170])
plt.legend(loc=5, fontsize=15)
plt.yticks(fontsize=15)
ax.set_xlabel('Perturbation value',fontsize=20)
ax.set_ylabel('Number of cells',fontsize=20)
plt.savefig("Hippo_trade.pdf")
plt.show()


#%% We decided to not include the Sankey plots in the paper. However its possible to have a Sankey of changes after the perturbation. I share the code here in case anyone wants to use it:
import pandas as pd
import chart_studio.plotly as py
import kaleido
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
df=pd.read_csv("DATA.csv")
df.index.name = 'state'
# df=df.rename(columns={'###########':"state"})
# df=df.set_index(df.iloc[:,0])
df=pd.melt(df)
# df1["state"]=["perturb" if df1["variable"].str.contains('per')]
all_d=list(df["variable"].str.contains('per'))

df['state']='say'
for i in range(0,len(all_d),2):
        df.iloc[i,2]="perturb"
for i in range(1,len(all_d),2):
        df.iloc[i,2]="unpertubed"


sources = df["variable"].drop_duplicates().tolist()
platforms = df["state"].drop_duplicates().tolist()
# platforms=platforms[1:]
all_nodes = sources + platforms

n = len(all_nodes)
n2 = len(df["state"])

df1 = pd.DataFrame(all_nodes, columns = ['node'])
df1 = df1.reset_index()
df2 = pd.merge(pd.merge(df, df1, how = 'inner', left_on = "state", right_on ="node"), df1, how = 'inner', left_on = "variable", right_on ="node", suffixes = ('_state','_target'))
 
data = dict(
    type='sankey',
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(
        color = "#435951",
        width = 0.5
      ),
      label = all_nodes,
      color = ["#84baa6"] * n
    ),
    link = dict(
      source = df2["index_state"],
      target = df2["index_target"],
      value = df2["value"],
      color = ['#bdf9e5'] * n2
  ))
 
# Setting up the layout settings in the "layout" argument
layout =  dict(
    title = "An Example Sankey Diagram",
    font = dict(
      size = 12
    )
)
 
fig = dict(data=[data], layout=layout)


# data to dict, dict to sankey
a = list(df.iloc[:,2])
b = list(df.iloc[:,0])
c = list(df.iloc[:,1])


link = dict(source = a, target = b, value = df.iloc[:,1])
# node = dict(label = all_nodes, pad=50, thickness=5)
data = go.Sankey(link = link)
# plot
fig = go.Figure(data)
fig.show()
fig.write_html("sankey-diagram-plotly12.html")



aa = [1]*27
bb = [1,2,3]*9

aa = [(x=='perturb')*10+100 for x in a]
1
bb = []
for i in range(len(np.unique(b))):
    bb.append(i)
    bb.append(i)    




source = [0, 0, 1, 1, 0]
target = [2, 3, 4, 5, 4]
value = [8, 2, 2, 8, 4]

link = dict(source = aa, target = bb, value = c)
data = go.Sankey(link = link)

fig = go.Figure(data)

fig.show()
fig.write_html("test.html")




