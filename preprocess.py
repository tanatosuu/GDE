import torch
import numpy as np
import gc
import time
import pandas as pd
#torch.cuda.set_device(1)
#choose datasets

dataset='gowalla'
#ml_1m
#user_size,item_size=6040,3952
#ml_100k
#user_size,item_size=943,1682
#pinterest
#user_size,item_size=37501,9836
#citeulike
#user_size,item_size=5551,16981
#gowalla
user_size,item_size=29858,40981

#the number of required features
smooth_ratio=0.1
rough_ratio=0.0


# Adj: adjacency matrix
# size: the number of required features
# largest: Ture (default) for k-largest (smoothed) and Flase for k-smallest (rough) eigenvalues
# niter: maximum number of iterations
def cal_spectral_feature(Adj, size, type='user', largest=True, niter=5):
	# params for the function lobpcg
	# k: the number of required features
	# largest: Ture (default) for k-largest (smoothed)  and Flase for k-smallest (rough) eigenvalues
	# niter: maximum number of iterations
	# for more information, see https://pytorch.org/docs/stable/generated/torch.lobpcg.html

	value,vector=torch.lobpcg(Adj,k=size, largest=largest,niter=niter)


	if largest==True:
		feature_file_name=dataset+'_smooth_'+type+'_features.npy'
		value_file_name=dataset+'_smooth_'+type+'_values.npy'

	else:
		feature_file_name=dataset+'_rough_'+type+'_features.npy'
		value_file_name=dataset+'_rough_'+type+'_values.npy'


	np.save(r'./'+dataset+r'/'+value_file_name,value.cpu().numpy())
	np.save(r'./'+dataset+r'/'+feature_file_name,vector.cpu().numpy())




# build rating matrix for datasets
df_train=pd.read_csv(dataset+r'/train_sparse.csv')
rate_matrix=torch.zeros(user_size,item_size).cuda()
for row in df_train.itertuples():
	rate_matrix[row[1],row[2]]=1


np.save(r'./'+dataset+r'/'+'rate_sparse.npy',rate_matrix.cpu().numpy())

#user degree and item degree
D_u=rate_matrix.sum(1)
D_i=rate_matrix.sum(0)


#in the case any users or items have no interactions
for i in range(user_size):
	if D_u[i]!=0:
		D_u[i]=1/D_u[i].sqrt()

for i in range(item_size):
	if D_i[i]!=0:
		D_i[i]=1/D_i[i].sqrt()



#(D_u)^{-0.5}*rate_matrix*(D_i)^{-0.5}
rate_matrix=D_u.unsqueeze(1)*rate_matrix*D_i

#clear GPU
del D_u, D_i 
gc.collect()
torch.cuda.empty_cache()

#user-user matrix
L_u=rate_matrix.mm(rate_matrix.t())

#smoothed feautes for user-user relations
cal_spectral_feature(L_u,int(smooth_ratio*user_size),type='user', largest=True)
#rough feautes for user-user relations
if rough_ratio!=0:
	cal_spectral_feature(L_u,int(rough_ratio*user_size),type='user',largest=False)

#clear GPU
del L_u 
gc.collect()
torch.cuda.empty_cache()

#item-item matrix
L_i=rate_matrix.t().mm(rate_matrix)


#smoothed feautes for item-item relations
cal_spectral_feature(L_i,int(smooth_ratio*item_size),type='item', largest=True)
#rough feautes for item-item relations
if rough_ratio!=0:
	cal_spectral_feature(L_i,int(rough_ratio*item_size),type='item',largest=False)
