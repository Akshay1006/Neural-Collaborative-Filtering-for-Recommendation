# Input Data Format is given in the dataset folder

import pandas as pd, numpy as np
from sklearn import model_selection, metrics, preprocessing

df=pd.read_csv('./data.csv',header=None)
df.columns=['unique_id','ad_id','date','click']

#Identify the click by customer on all the possible ads

df_grouped=pd.DataFrame(df.groupby(['unique_id','ad_id'])['click'].max().reset_index())

#Encoding the ad_id with a number and unique_id with a number

ad_code=df_grouped.ad_id.astype('category')
ad_map=dict(enumerate(ad_code.cat.categories))

len(ad_map)

cust_code=df_grouped.unique_id.astype('category')
cust_map=dict(enumerate(cust_code.cat.categories))

len(cust_map)

ad_index={idx: ad for ad,idx in ad_map.items()}
cust_index={idx: cust for cust,idx in cust_map.items()}

print(len(ad_map),len(ad_index),len(cust_map),len(cust_index))

# Encode the values in the dataset as NN's need data in the encoded format to generate embeddings

df_grouped.ad_id=df_grouped.ad_id.astype('category').cat.codes.values
df_grouped.unique_id=df_grouped.unique_id.astype('category').cat.codes.values

import keras
from keras.layers import Input, Embedding, Dot, Reshape, Dense, Dropout,Concatenate
from keras.models import Model

#Dimension size of the Embedded Vector

n_latent_ad=25 # MLP Dimension
n_latent_customer=75 # MLP Dimension
n_latent_mf=40 #MF Dimension

# Initialize the Embedding-MF

ad_input_mf = Input(name = 'ad_mf', shape = [1])
customer_input_mf = Input(name = 'customer_mf', shape = [1])

ad_embedding_mf = keras.layers.Embedding(len(ad_map)  , n_latent_mf, name='ad-Embedding_mf')(ad_input_mf)
customer_embedding_mf = keras.layers.Embedding(len(cust_map)  , n_latent_mf, name='custs-Embedding_mf')(customer_input_mf)

merged_mf = Dot(name = 'dot_product', normalize = True, axes = 2)([ad_embedding_mf, customer_embedding_mf]) # Matrix factorization 

merged_mf=keras.layers.Flatten(name='flatten_embeddings_mf')(merged_mf)

# Initialize the Embedding-MLP

ad_embedding_mlp=keras.layers.Embedding(len(ad_map) ,n_latent_ad,name='ad-Embedding_mlp')(ad_input_mf)
ad_embedding_mlp=keras.layers.Flatten(name='flatten_ad_mlp')(ad_embedding_mlp)
ad_embedding_mlp=keras.layers.Dropout(0.2)(ad_embedding_mlp)

customer_embedding_mlp=keras.layers.Embedding(len(cust_map) ,n_latent_customer,name='customer-Embedding_mlp')(customer_input_mf)
customer_embedding_mlp=keras.layers.Flatten(name='flatten_customer_mlp')(customer_embedding_mlp)
customer_embedding_mlp=keras.layers.Dropout(0.2)(customer_embedding_mlp)

#concatenating the ad and customer embeddings for MLP:

concat_mlp = Concatenate(name='Concatenated_mlp')([ad_embedding_mlp,customer_embedding_mlp])

dense_1=keras.layers.Dense(100,name='Dense_1',activation='relu')(concat_mlp)
dense_1=keras.layers.Dropout(0.3)(dense_1)

dense_2=keras.layers.Dense(50,name='Dense_2',activation='relu')(dense_1)
dense_2=keras.layers.Dropout(0.2)(dense_2)

dense_3=keras.layers.Dense(20,name='dense_3',activation='relu')(dense_2)
dense_3=keras.layers.Dropout(0.2)(dense_3)

#Concatenate MF and MLP 

concat_all=Concatenate(name='Concatenated_all')([dense_3,merged_mf])

final = Dense(1, activation = 'sigmoid')(concat_all)

model = Model(inputs = [ad_input_mf, customer_input_mf], outputs = final)
model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.summary()

#Model fitting
#ad_id and unique_id are the inputs, and click is the output
fitting = model.fit([df_grouped.ad_id, df_grouped.unique_id], df_grouped.click, epochs=50,batch_size=100000, verbose=2)

#Extract the Embeddings for further fine tuning 

ad_embedding_mf_learnt = model.get_layer(name='ad-Embedding_mf').get_weights()[0]
customer_embedding_mf_learnt = model.get_layer(name='custs-Embedding_mf').get_weights()[0]
customer_embedding_mlp_learnt = model.get_layer(name='customer-Embedding_mlp').get_weights()[0]
ad_embedding_mlp_learnt = model.get_layer(name='ad-Embedding_mlp').get_weights()[0]

cols_ad_mf=["ad_embed_mf{}".format(x + 1) for x in range(ad_embedding_mf_learnt.shape[1])]
cols_cust_mf=["cust_embed_mf{}".format(x + 1) for x in range(customer_embedding_mf_learnt.shape[1])]

df_ad_embedding_mf=pd.DataFrame(ad_embedding_mf_learnt,columns=cols_ad_mf)
df_cust_embedding_mf=pd.DataFrame(customer_embedding_mf_learnt,columns=cols_cust_mf)

cols_ad_mlp=["ad_embed_mlp{}".format(x + 1) for x in range(ad_embedding_mlp_learnt.shape[1])]
cols_cust_mlp=["cust_embed_mlp{}".format(x + 1) for x in range(customer_embedding_mlp_learnt.shape[1])]

df_ad_embedding_mlp=pd.DataFrame(ad_embedding_mlp_learnt,columns=cols_ad_mlp)
df_cust_embedding_mlp=pd.DataFrame(customer_embedding_mlp_learnt,columns=cols_cust_mlp)

df_offr_map=pd.DataFrame([ad_map]).T
df_cust_map=pd.DataFrame([cust_map]).T

#MF Final embeddings
df_ad_embed_final_mf=pd.merge(df_ad_embedding_mf,df_offr_map,left_index=True,right_index=True)
df_cust_embed_final_mf=pd.merge(df_cust_embedding_mf,df_cust_map,left_index=True,right_index=True)

#MLP final embeddings
df_ad_embed_final_mlp=pd.merge(df_ad_embedding_mlp,df_offr_map,left_index=True,right_index=True)
df_cust_embed_final_mlp=pd.merge(df_cust_embedding_mlp,df_cust_map,left_index=True,right_index=True)

df_ad_embed_final_mf.rename(columns={0:'ad_id'},inplace=True)
df_cust_embed_final_mf.rename(columns={0:'unique_id'},inplace=True)

df_ad_embed_final_mlp.rename(columns={0:'ad_id'},inplace=True)
df_cust_embed_final_mlp.rename(columns={0:'unique_id'},inplace=True)
