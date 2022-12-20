# -*- coding: utf-8 -*-
"""

"""
#%%
# libraries:
from tensorflow.keras.layers import concatenate as concat
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Lambda
from matplotlib import pyplot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import plot_confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
#import tensorflow as tf
# import np_utils
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K, activations
# from openpyxl import Workbook
from sklearn.metrics import confusion_matrix
import random 
from sklearn.metrics import accuracy_score
from tensorflow import keras
from matplotlib.pyplot import figure
import seaborn as sns
import scvelo as scv
import pandas as pd
import numpy as np
import pandas as pd
import scipy.io as sio
import scanpy as sc
import os
figure(figsize=(5, 3), dpi=500)

#%%#################################################################
# samlping for VAE and CVAE:
# latent_dim = round(float(n_inputs) / 100)
def sampling(args, latent_dim=10):
    ''' 
    function to sample for VAE and CVAE
    '''
    global result
    global epsilon
    global z_log_sigma,z_mean
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=1)
    result  = z_mean + K.exp(z_log_sigma/2) * epsilon 
    return result


#%%##################################################################
#This part of the code is about reading and preprocessing data. I recommned you to try differnt preprocessing ways since as it has been shown by Gorin et al. that
# current preprocessing methods might lead to a more unrealistic models which omits all biological variation
# So here you can use the exisiting packages to read the data or read the count matrix as csv. Both can be seen below: 
# Also we are using label encoder to convert the labels into a numeric form so AE and the classifier understand what we are doing. 
# Just make sure you choose your loss funtion correctly based on the type of data you have and the preprocessing step you took.



# adata = scv.datasets.dentategyrus()
# scv.pp.filter_and_normalize(adata, min_shared_counts=30, n_top_genes=2000)
# scv.tl.velocity(adata)
# scv.tl.velocity_graph(adata)
# cols=pd.DataFrame(adata.var)
# cols=cols.index
# X_orig=pd.DataFrame.sparse.from_spmatrix(adata.X, columns=cols)
# # X_orig = (X_orig - X_orig.mean())/(X_orig.std())
# Y=pd.DataFrame(adata.obs.clusters)
# X_orig.index=Y.index
# classes, counts = np.unique(Y, return_counts=True)
# # Encode class values as integers
# encoder = LabelEncoder()
# encoder.fit(Y)
# encodedY = encoder.transform(Y)
# dummyY = to_categorical(encodedY)
# dummyY=pd.DataFrame(dummyY, index=Y.index,columns=[classes])    
    
    
#We tried Zebrafish dataset here (thats why you see zebra as the name of the pandas dataframe)
zebra= pd.read_csv("YOURFILE.csv")
# Changing the index
zebra=zebra.set_index('Cells')                   
X_orig = zebra.loc[:, zebra.columns != 'clusters']
X_orig = (X_orig - X_orig.mean())/(X_orig.std()) # You can log normalize too or delete cell cycle or mito genes (check common pipelines)
Y= zebra['clusters']
encoder = LabelEncoder()
encoder.fit(Y)
encodedY = encoder.transform(Y)
dummyY = to_categorical(encodedY)
dummyY=pd.DataFrame(dummyY, index=Y.index,columns=["Erythrocytes","HSPC","Monocytes","Neutrophils","Thrombocytes"])

#%%##################################################################################################################################
# Lets split the data into train and test sets. You can choose different ratio between training and testing (dont use all your data to test... doesnt make sense):    
X_train, X_test, y_train, y_test = train_test_split(X_orig, dummyY, test_size=0.15, random_state=1)
n_inputs = X_orig.shape[1] #checking the shape of the X here.
n_y = y_train.shape[1] #checking the shape of the y here.



# Here we defined 3 different types of autoencoders AE, VAE and CVAE you can change the type here. The default size of latent layer is #featuers(input_size)/150 however you HAVE TO try different
# arch desgins for your data. The performance of the AE really depends on the architecture. Again please PERFORM a hyperparameter search: 
def build_two_layer_network_model(X, type = 'AE', latent_dim = round(float(n_inputs) / 150), n_y = n_y):
    ''' 
    building a two layer AE, VAE or CVAE
    input is X and output is the encoder, decoder, model and classifier
    '''
    n_inputs = X.shape[1]
    
    # concat label if CVAE:
    if type == 'CVAE':
        X = Input(shape=(n_inputs,))
        label = Input(shape=(n_y,))
        visible = concat([X, label])
    else:
        visible = Input(shape=(n_inputs,))

    # define encoder
    # encoder level 1
    e = Dense(round(float(n_inputs) / 20))(visible)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    # encoder level 2
    e = Dense(round(float(n_inputs) / 40))(e)
    e = BatchNormalization()(e)
    e = LeakyReLU()(e)
    n_bottleneck = latent_dim
    bottleneck = Dense(n_bottleneck)(e)

    # bottleneck
    if type == 'AE':
        encoder = Model(visible, bottleneck, name="encoder")
        #encoder.summary()
        ppp = latent_dim
        visible2 = Input(shape=(ppp,))

    elif type == 'VAE' or type == 'CVAE':
        ''' finding mu and sigma '''
        mu      = Dense(latent_dim, name='latent_mu')(e)
        sigma   = Dense(latent_dim, name='latent_sigma')(e)

        if type == 'VAE':
            def sampling(args, latent_dim= latent_dim):
                z_mean, z_log_sigma = args
                epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=0.5)
                return z_mean + K.exp(z_log_sigma/2) * epsilon
            
            z = Lambda(sampling)([mu, sigma])
            #z = Dense(n_bottleneck)(z)
            encoder = Model(visible, [mu, sigma, z], name='encoder')
            # [mu, sigma, z]
            visible2   = Input(shape=(latent_dim,))


        else:
            def sampling(args, latent_dim= latent_dim):
                z_mean, z_log_sigma = args
                epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=0.5)
                return z_mean + K.exp(z_log_sigma/2) * epsilon
            
            # if CVAE 
            z = Lambda(sampling, output_shape = (latent_dim, ))([mu, sigma])
            #z = Dense(n_bottleneck)(z)
            z= concat([z, label])
            encoder = Model([X, label] ,[mu, sigma, z], name='encoder')
            visible2 = z
            # [mu, sigma, z]    
    encoder.summary()


    # define decoder, level 1
    d = Dense(round(float(n_inputs) / 40))(visible2)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)
    # decoder level 2
    d = Dense(round(float(n_inputs) / 20))(d)
    d = BatchNormalization()(d)
    d = LeakyReLU()(d)

    # output layer
    output = Dense(n_inputs, activation='linear')(d)

    if type == 'CVAE':
        d_in = Input(shape=(latent_dim+n_y,))
        d_cvae = Dense(round(float(n_inputs) / 40))(d_in)
        d_cvae = BatchNormalization()(d_cvae)
        d_cvae = LeakyReLU()(d_cvae)
        # decoder level 2
        d_cvae = Dense(round(float(n_inputs) / 20))(d_cvae)
        d_cvae = BatchNormalization()(d_cvae)
        d_cvae = LeakyReLU()(d_cvae)
        cvae_output  = Dense(n_inputs, activation='linear')(d_cvae)
        decoder = Model(d_in, cvae_output)

    
    else:
        decoder = Model(visible2, output, name="decoder")
        #import pdb; pdb.set_trace()
        encoded = encoder(visible)
        decoded = decoder(encoded)

            
        class1=Dense(20,activation='relu')(visible2)
        class2=Dense(14,activation='softmax')(class1)
        Classifier = Model(visible2, class2, name="Classifier")
        Classifiered=Classifier(encoded)
        model = Model(inputs=visible, outputs=[decoded,Classifiered])
        
    if type == 'AE':
     model.compile(optimizer='adam', loss=['mse', 'categorical_crossentropy'])

    elif type == 'VAE':
        model = Model(inputs =visible, name = 'vae_m', outputs = [decoded,Classifiered])

        def vae_loss(x, x_decoded_mean):
            xent_loss = keras.losses.binary_crossentropy(x, x_decoded_mean)
            kl_loss = - 0.5 * K.sum(1 + sigma - K.square(mu) - K.exp(sigma), axis=-1)
            return xent_loss + kl_loss

        model.compile(optimizer='adam', loss = vae_loss)

    elif type == 'CVAE':
        model = Model([X, label], output, name="decoder")

        def vae_loss(y_true, y_pred):
            recon = K.sum(
                K.binary_crossentropy(y_true, y_pred), axis=-1)
            kl = 0.5 * K.sum(K.exp(sigma) + K.square(mu) - 1. - sigma, axis=-1)
            return recon + kl

        def KL_loss(y_true, y_pred):
            return(0.5 * K.sum(
                K.exp(sigma) + K.square(mu) - 1. - sigma, axis=1))

        def recon_loss(y_true, y_pred):
            return K.sum(
                K.binary_crossentropy(y_true, y_pred), axis=-1)

        model.compile(
            optimizer='adam', 
            loss=vae_loss, 
            metrics = [KL_loss, recon_loss])
        Classifier = []
        

    
    decoder.summary()
    return model,encoder,decoder,Classifier
    # return model,decoder,encoder

#%%##########################################################################################################
# building the model. Choose the type of AE and arch that is better suited for you and gives you a low loss and high correlation:
model,encoder,decoder, Classifier = build_two_layer_network_model(X_orig, type ='CVAE', latent_dim = round(float(n_inputs) / 150))
decoder.summary()
#%%#########################################################################################################
# fit the autoencoder model to reconstruct input. Now you have 2 options if you want to run VAE and AE run this part. If you like VAE run the cell below.
# for AE and VAE:
history = model.fit(X_train, [X_train,y_train], epochs=100, batch_size=200, verbose=2, validation_data=(X_test,[X_test,y_test]))
model.summary()

# Make sure your loss function is deacreasing. YOU DONT WANT TO OVERFIT. Also if interested you can try early stopping and drop out 
fig=plt.figure(figsize=(12,8))
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()
fig.savefig('Error_plot.pdf')

#%%#########################################################################################################
# for CVAE:
m = 130
n_epoch = 100
cvae_hist = model.fit([X_train, y_train], X_train, verbose = 1, batch_size=m, epochs=n_epoch, validation_data = ([X_test, y_test], X_test))
model.summary()
#%%%------------------------------------------------------------
# Now lets see if the AEs were able to reproduce and reduce the data well or not. We are using Correlation and Other Asssesments:
np_train = X_train.to_numpy()
# new_output = model.predict(X_train) # For AE or VAE
new_output = model.predict([X_train, y_train])  #For  CVAE 

# using pearson correlation
from scipy.stats.stats import pearsonr
def row_corr(A,B):
    correlt_all = []
    for i in range(len(A)):
        correlt = pearsonr(A[i,:],B[i,:])
        correlt_all.append(correlt[0])
    return correlt_all

    
# correlated_models = row_corr(np_train,new_output)
# np.mean(correlated_models) # average correlation of cells for each celltype
#####################################################################
#%%
def bar_plot_all(title, values, file_name):

    '''
    function to plot correlation
    '''
    plt.title(title)
    plt.bar(["Erythrocytes","HSPC","Monocytes","Neutrophils","Thrombocytes"], values)   # Write your cell types here

    plt.grid()
    plt.ylim([0, 2])
    plt.xlabel('Cell Type')
    plt.savefig("{}.pdf".format(file_name), format='pdf', dpi = 300,
    bbox_inches = 'tight')


def corr_cell_types(title, filename, type = 'CVAE'):

    '''
    function to find AE output, find correlation, mse and plot and save them 
    for different AE models 
    '''

    corr_all1 = []
    mse_all = []
    for i in range(1,6):
        Y_train = y_train[:]
        Y_train.columns = [1,2,3,4,5]
        Y_train = Y_train.idxmax(axis=1)
        Y_train_filtered = Y_train[Y_train==i]

        X_train_filtered = X_train[X_train.index.isin(Y_train_filtered.index)]
        np_train_filtered = X_train_filtered.to_numpy()

        if type == 'CVAE':
            y_train_filtered = y_train[y_train.index.isin(Y_train_filtered.index)]
            new_output_filtered = model.predict([X_train_filtered,y_train_filtered])
            corr_filtered = row_corr(np_train_filtered,
                                new_output_filtered)
            mse = (np.square(new_output_filtered - np_train_filtered)).mean(axis=1)


        else:
            new_output_filtered = model.predict(X_train_filtered)
            corr_filtered = row_corr(np_train_filtered, new_output_filtered[0])

            mse = (np.square(new_output_filtered[0] - np_train_filtered)).mean(axis=1)


        mse_filtered = np.mean(mse)
        mse_all.append(mse_filtered)

        corr_filtered = np.mean(corr_filtered)
        corr_all1.append(corr_filtered)

    bar_plot_all("MSE "+title, mse_all, filename+"_mse")
    plt.show()
    bar_plot_all("Correlation "+title, corr_all1, filename+"_corr")  
    plt.show()

    return(corr_all1,mse_all)

#%%###############################################################################################################################################
# Running this just gives you results for the CVAE. Uncomment the other two lines or you can easily write a for loop to get results of all 3 AEs:
    
corr_CVAE, mse_CVAE = corr_cell_types("Plot for Autoencoder", "cell_corr_vae", "CVAE")
# corr_AE, mse_AE = corr_cell_types("Plot for Autoencoder", "cell_corr_vae", "AE")
# corr_VAE, mse_VAE = corr_cell_types("Plot for Autoencoder", "cell_corr_vae", "VAE")

# Assumin you ran all 3 methods to choose the best one (Thats the right thing to do)
Corr = pd.DataFrame({'Cells': ["Erythrocytes","HSPC","Monocytes","Neutrophils","Thrombocytes"],'AE': corr_AE, 'VAE': corr_VAE, 'CVAE' : corr_CVAE })
Errror = pd.DataFrame({'Cells': ["Erythrocytes","HSPC","Monocytes","Neutrophils","Thrombocytes"],'AE': mse_AE, 'VAE': mse_VAE, 'CVAE' : mse_CVAE})


# Corr plot
# Now lets use seaborn to plot the barplots: 
#set seaborn plotting aesthetics
#create stacked bar chart
fig, ax = plt.subplots(figsize=(25,15))
ax= Corr.set_index('Cells').plot(kind='bar', stacked=False, color=['steelblue', 'red','green'])
plt.ylabel('Correlation')
ax.set_facecolor("white")
plt.style.context("seaborn-white")
ax.spines['bottom'].set_color('0.5')
ax.spines['top'].set_color('0.5')
ax.spines['right'].set_color('0.5')
ax.spines['left'].set_color('0.5')
ax.set(xlabel=None)
# plt.title('Corr for different AE architectures', fontsize=14)
plt.savefig('Corr for different AE architectures.pdf', bbox_inches="tight")



# Error plot
#set seaborn plotting aesthetics
#create stacked bar chart
fig= plt.figure(figsize=(25,15))
# fig.set_facecolor('white')
ax=Errror.set_index('Cells').plot(kind='bar', stacked=False, color=['steelblue', 'red','green'])
plt.ylabel('Error')
ax.set_facecolor("white")
plt.style.context("seaborn-white")
ax.spines['bottom'].set_color('0.5')
ax.spines['top'].set_color('0.5')
ax.spines['right'].set_color('0.5')
ax.spines['left'].set_color('0.5')
ax.set(xlabel=None)
plt.savefig('Error for different AE architectures.png', bbox_inches="tight")



#%%
# plotting latent space of autoencoders:
Y_train = y_train[:]
Y_train.columns = [1,2,3,4,5]
Y_train = Y_train.idxmax(axis=1)
#z_train = encoder.predict([X_train, y_train]) # for CVAE
z_train = encoder.predict(X_train) #For VAE and AE
plt.figure(figsize=(6, 6))
scatter = plt.scatter(z_train[:,0], z_train[:,1],
            alpha=.4, s=3**2, c= Y_train, cmap='viridis')

plt.colorbar()
plt.legend()
plt.ylim([-1,1])
plt.xlim([-1.5,2.3])
plt.title("Latent Space of Variational Autoencoder")
plt.xlabel('z1')
plt.ylabel('z2')
classes = ["Erythrocytes","HSPC","Monocytes","Neutrophils","Thrombocytes"]
plt.legend(handles=scatter.legend_elements()[0], labels=classes)
plt.savefig("LS_VAEـupdated.png", format='png', dpi = 300, bbox_inches = 'tight')
plt.show()



