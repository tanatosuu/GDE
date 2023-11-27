import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pandas as pd
import time
import gc


#gowalla
user,item=29858,40981
#yelp
#user,item=25677,25815
#ml-1m
#user,item=6040,3952

result=[]
dataset='./gowalla'
learning_rate=8.5
batch_size=256


#瀵煎叆鏁版嵁
df_train=pd.read_csv(dataset+ r'/train_sparse.csv')
df_test=pd.read_csv(dataset+ r'/test_sparse.csv')

#load the train/test data
#load the data
train_samples=0
#train_data=[[] for i in range(user)]
test_data=[[] for i in range(user)]
for row in df_train.itertuples():
	#train_data[row[1]].append(row[2])
	train_samples+=1
for row in df_test.itertuples():
	test_data[row[1]].append(row[2])
##########################################
#interaction matrix
rate_matrix=torch.Tensor(np.load(dataset+ r'/rate_sparse.npy')).cuda()
#user-user, item-item graph
user_matrix=((rate_matrix.mm(rate_matrix.t()))!=0).float()
item_matrix=((rate_matrix.t().mm(rate_matrix))!=0).float()


class SGDE(nn.Module):
	def __init__(self, user_size, item_size, beta=2.0, std=0.01,coef_u=0.5,coef_i=0.3, req_vec=90, latent_size=64, reg=0.01):
		super(SGDE, self).__init__()

		self.latent_size=latent_size
		self.std=std
		self.coef_u=coef_u
		self.coef_i=coef_i
		self.reg=reg
		self.beta=beta
		self.user_size=user_size
		self.item_size=item_size

		svd_filter=self.weight_func(torch.Tensor(np.load(dataset+ r'/svd_value.npy')[:req_vec]).cuda())
		self.user_vector=(torch.Tensor(np.load(dataset+ r'/svd_u.npy')[:,:req_vec])).cuda()*svd_filter
		self.item_vector=(torch.Tensor(np.load(dataset+ r'/svd_v.npy')[:,:req_vec])).cuda()*svd_filter
		self.FS=Variable(torch.nn.init.uniform_(torch.randn(req_vec,latent_size),-np.sqrt(6. / (req_vec+latent_size) ) ,np.sqrt(6. / (req_vec+latent_size) )).cuda(),requires_grad=True)

	def weight_func(self,sig):
		return torch.exp(self.beta*sig)



	def predict(self):

		final_user=self.user_vector.mm(self.FS)
		final_item=self.item_vector.mm(self.FS)
		return (final_user.mm(final_item.t())).sigmoid()-rate_matrix*1000


	def forward(self,u,p,n,up,un,pp,pn):

		final_user,final_pos,final_nega=torch.normal(self.user_vector[u],std=self.std).mm(self.FS),torch.normal(self.item_vector[p],std=self.std).mm(self.FS),torch.normal(self.item_vector[n],std=self.std).mm(self.FS)
		final_user_p,final_user_n=torch.normal(self.user_vector[up],std=self.std).mm(self.FS),torch.normal(self.user_vector[un],std=self.std).mm(self.FS)
		final_pos_p,final_pos_n=torch.normal(self.item_vector[pp],std=self.std).mm(self.FS),torch.normal(self.item_vector[pn],std=self.std).mm(self.FS)

		out=((final_user*final_pos).sum(1)-(final_user*final_nega).sum(1)).sigmoid()
		self_loss_u=torch.log(((final_user*final_user_p).sum(1)-(final_user*final_user_n).sum(1)).sigmoid()).sum()
		self_loss_i=torch.log(((final_pos*final_pos_p).sum(1)-(final_pos*final_pos_n).sum(1)).sigmoid()).sum()
		regu_term=self.reg*(final_user**2+final_pos**2+final_nega**2+final_user_p**2+final_user_n**2+final_pos_p**2+final_pos_n**2).sum()

		return (-torch.log(out).sum()-self.coef_u*self_loss_u-self.coef_i*self_loss_i+regu_term)/batch_size


	def test(self):
		#calculate idcg@k(k={1,...,20})
		def cal_idcg(k=20):
			idcg_set=[0]
			scores=0.0
			for i in range(1,k+1):
				scores+=1/np.log2(1+i)
				idcg_set.append(scores)

			return idcg_set

		def cal_score(topn,now_user,trunc=20):
			dcg10,dcg20,hit10,hit20=0.0,0.0,0.0,0.0
			for k in range(trunc):
				max_item=topn[k]
				if test_data[now_user].count(max_item)!=0:
					if k<=10:
						dcg10+=1/np.log2(2+k)
						hit10+=1
					dcg20+=1/np.log2(2+k)
					hit20+=1

			return dcg10,dcg20,hit10,hit20



		#accuracy on test data
		ndcg10,ndcg20,recall10,recall20=0.0,0.0,0.0,0.0
		predict=self.predict()

		idcg_set=cal_idcg()
		for now_user in range(user):
			test_lens=len(test_data[now_user])

			#number of test items truncated at k
			all10=10 if(test_lens>10) else test_lens
			all20=20 if(test_lens>20) else test_lens
		
			#calculate dcg
			topn=predict[now_user].topk(20)[1]

			dcg10,dcg20,hit10,hit20=cal_score(topn,now_user)


			ndcg10+=(dcg10/idcg_set[all10])
			ndcg20+=(dcg20/idcg_set[all20])
			recall10+=(hit10/all10)
			recall20+=(hit20/all20)			

		ndcg10,ndcg20,recall10,recall20=round(ndcg10/user,4),round(ndcg20/user,4),round(recall10/user,4),round(recall20/user,4)
		print(ndcg10,ndcg20,recall10,recall20)

		result.append([ndcg10,ndcg20,recall10,recall20])



#Model training and test

model = SGDE(user, item)

epoch=train_samples//batch_size

for i in range(15):
	total_loss,loss=0.0,0
	start=time.time()
	for j in range(0,epoch):
		
		
		u=np.random.randint(0,user,batch_size)
		p=torch.multinomial(rate_matrix[u],1,True).squeeze(1)
		nega=torch.multinomial(1-rate_matrix[u],1,True).squeeze(1)
		up=torch.multinomial(user_matrix[u],1,True).squeeze(1)
		un=torch.multinomial(1-user_matrix[u],1,True).squeeze(1)
		pp=torch.multinomial(item_matrix[p],1,True).squeeze(1)
		pn=torch.multinomial(1-item_matrix[p],1,True).squeeze(1)
		
		loss=model(u,p,nega,up,un,pp,pn)
		loss.backward()
		with torch.no_grad():
			model.FS-=learning_rate*model.FS.grad
			model.FS.grad.zero_()
		total_loss+=loss.item()

	end=time.time()
	print(end-start)
	
	print('epoch %d training loss:%f' %(i,total_loss/epoch))
	if (i+1)%1==0 and (i+1)>=15:
		model.test()

exit()


output=pd.DataFrame(result)
output.to_csv(r'./csgde.csv')

