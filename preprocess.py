import torch
import numpy as np
import gc
import time
import pandas as pd
#torch.cuda.set_device(1)
#choose datasets

dataset='ml_100k'
#ml_1m
#user_size,item_size=6040,3952
#ml_100k
#user_size,item_size=943,1682
#pinterest
user_size,item_size=37501,9836
#citeulike
#user_size,item_size=5551,16981
#gowalla
#user_size,item_size=29858,40981

#the number of required features
smooth_ratio=0.2
rough_ratio=0.002


# Adj: adjacency matrix
# size: the number of required features
# argest: Ture (default) for k-largest (smoothed) and Flase for k-smallest (rough) eigenvalues
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



int main():
# build rating matrix for datasets
df_train=pd.read_csv(dataset+r'/train_sparse.csv')
rate_matrix=torch.zeros(user_size,item_size).cuda()
for row in df_train.itertuples():
	rate_matrix[row[1],row[2]]=1


np.save(r'./'+dataset+r'/'+'rate_sparse.npy',rate_matrix.cpu().numpy())

#user degree and item degree
user_degree=rate_matrix.sum(1)
item_degree=rate_matrix.sum(0)

D=torch.diag(user_degree)
M=torch.diag(item_degree)

#in the case any users or items have no interactions
for i in range(len(user_degree)):
	if D[i,i]!=0:
		D[i,i]=1/D[i,i].sqrt()

for i in range(len(item_degree)):
	if M[i,i]!=0:
		M[i,i]=1/M[i,i]





L_u=D.mm(rate_matrix).mm(M).mm(rate_matrix.t()).mm(D)




D=torch.diag(item_degree)
M=torch.diag(user_degree)

for i in range(len(item_degree)):
	if D[i,i]!=0:
		D[i,i]=1/D[i,i].sqrt()

for i in range(len(user_degree)):
	if M[i,i]!=0:
		M[i,i]=1/M[i,i]

L_i=D.mm(rate_matrix.t()).mm(M).mm(rate_matrix).mm(D)




# The size of user-user and item-item adjacency matrix
print(L_u.shape,L_i.shape)


#clear memory
del D,M, rate_matrix, user_degree, item_degree 
gc.collect()
torch.cuda.empty_cache()



print('step 1 finished.')


########################################
########################################
#calcualte spectral features




#smoothed feautes for user-user relations
cal_spectral_feature(L_u,int(smooth_ratio*user_size),type='user', largest=True)
#rough feautes for user-user relations
if rough_ratio!=0:
	cal_spectral_feature(L_u,int(rough_ratio*user_size),type='user',largest=False)



#clear memory if not enough
del L_u
gc.collect()
torch.cuda.empty_cache()



#smoothed feautes for item-item relations
cal_spectral_feature(L_i,int(smooth_ratio*item_size),type='item', largest=True)
#rough feautes for item-item relations
if rough_ratio!=0:
	cal_spectral_feature(L_i,int(rough_ratio*item_size),type='item',largest=False)



