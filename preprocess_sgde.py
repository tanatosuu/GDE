import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import time
import gc


dataset='./movielens'
df_train=pd.read_csv(dataset+r'/train_sparse.csv')

#yelp
#user,item=25677,25815
#ML-1M
user,item=6040,3952
#citeulike
#user,item=5551,16981
#pinterest
#user,item=37501,9836
#gowalla
#user,item=29858,40981


alpha=2


rate_matrix=torch.zeros(user,item).cuda()

for row in df_train.itertuples():

	rate_matrix[row[1],row[2]]=1


#save interaction matrix
np.save(dataset+r'/rate_sparse.npy',rate_matrix.cpu().numpy())


D_u=rate_matrix.sum(1)+alpha
D_i=rate_matrix.sum(0)+alpha




for i in range(user):
	if D_u[i]!=0:
		D_u[i]=1/D_u[i].sqrt()

for i in range(item):
	if D_i[i]!=0:
		D_i[i]=1/D_i[i].sqrt()


#\tilde{R}
rate_matrix=D_u.unsqueeze(1)*rate_matrix*D_i


#free space
del D_u,D_i
gc.collect()
torch.cuda.empty_cache()


'''
q:the number of singular vectors in descending order
'''
print('start!')
start=time.time()
U,value,V=torch.svd_lowrank(rate_matrix,q=400,niter=30)
#U,value,V=torch.svd(R)
end=time.time()
print('processing time is %f' % (end-start))
print('singular value range %f ~ %f' % (value.min(),value.max()))


np.save(dataset+r'/svd_u.npy',U.cpu().numpy())
np.save(dataset+r'/svd_v.npy',V.cpu().numpy())
np.save(dataset+r'/svd_value.npy',value.cpu().numpy())
