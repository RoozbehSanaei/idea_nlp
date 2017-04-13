import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random




def costs(cl,M):
	IC = 0
	EC = 0
	CS = [cl.tolist().count(i) for i in range(max(cl.tolist())+1)]
	for i in range(len(M)):
		for j in range(len(M)):
			if (cl[i]==cl[j]):
				IC = IC + M[i][j]/ (CS[cl[i]])
			else:
				EC = EC + M[i][j]/len(M)
	return IC,EC


def cross1(cl1):
	M = np.zeros((len(cl1),len(cl1)))
	for i in range(len(cl1)):
		for j in range(len(cl1)):
			if (cl1[i]==cl1[j]):
				M[i,j] = 1
	return(M)

import pickle
f = open('store.pckl', 'rb')
results = pickle.load(f)
f.close()


clusters = [np.array(results[i][7]) for i in range(len(results))]
matrices = [np.array(results[i][13]) for i in range(len(results))]


fb = 'n50_de_bc'
fileb = '/home/roozbeh/data/wiki/data/'+fb+'.csv'
dfb = pd.read_csv(fileb)
clb = dfb.ix[:,0]
Mb = cross1(clb)


fr = 'n50_de_rma'
filer = '/home/roozbeh/data/wiki/data/'+fr+'.csv'
dfr = pd.read_csv(filer)
clr = dfr.ix[:,0]
Mr = cross1(clr)


dmr = [];pmr=[];fpr=[];fnr=[];
for cl in clusters:
	M =  cross1(cl)
	dmr.append(float(np.sum(abs(M-Mr))))
	pmr.append(float(np.sum(np.multiply(M,Mr))))
	fpr.append(float(np.sum((M>Mr)*1)))
	fnr.append(float(np.sum((M<Mr)*1)))


dmb = [];pmb=[];fpb=[];fnb=[];
for cl in clusters:
	M =  cross1(cl)
	dmb.append(float(np.sum(abs(M-Mb))))
	pmb.append(float(np.sum(np.multiply(M,Mb))))
	fpb.append(float(np.sum((M>Mb)*1)))
	fnb.append(float(np.sum((M<Mb)*1)))

dm_r = []
pm_r= []
fp_r = []
fn_r = []

dm_b = []
pm_b= []
fp_b = []
fn_b = []


for j in range(1,20):
	for i in range(10000):
		cl = [random.randint(0,j) for r in range(50)]
		M = cross1(cl)
		dm_r.append(float(np.sum(abs(M-Mr))))
		pm_r.append(float(np.sum(np.multiply(M,Mr))))
		fp_r.append(float(np.sum((M>Mr)*1)))
		fn_r.append(float(np.sum((M<Mr)*1)))


for j in range(1,20):
	for i in range(10000):
		cl = [random.randint(0,j) for r in range(50)]
		M = cross1(cl)
		dm_b.append(float(np.sum(abs(M-Mb))))
		pm_b.append(float(np.sum(np.multiply(M,Mb))))
		fp_b.append(float(np.sum((M>Mb)*1)))
		fn_b.append(float(np.sum((M<Mb)*1)))







dm_rb=float(np.sum(abs(Mr-Mb)))
pm_rb=float(np.sum(np.multiply(Mr,Mb)))
fp_rb=float(np.sum((Mr>Mb)*1))
fn_rb=float(np.sum((Mr<Mb)*1))

dm_rr=float(np.sum(abs(Mr-Mr)))
pm_rr=float(np.sum(np.multiply(Mr,Mr)))
fp_rr=float(np.sum((Mr>Mr)*1))
fn_rr=float(np.sum((Mr<Mr)*1))

dm_bb=float(np.sum(abs(Mb-Mb)))
pm_bb=float(np.sum(np.multiply(Mb,Mb)))
fp_bb=float(np.sum((Mb>Mb)*1))
fn_bb=float(np.sum((Mb<Mb)*1))


dm_br=float(np.sum(abs(Mb-Mr)))
pm_br=float(np.sum(np.multiply(Mb,Mr)))
fp_br=float(np.sum((Mb>Mr)*1))
fn_br=float(np.sum((Mb<Mr)*1))


plt.scatter(pm_r, pm_b, s=5, c='black')
plt.scatter(pmr, pmb, s=5, c='blue')
plt.scatter([pm_rr], [pm_rb], s=5, c='red')
plt.scatter([pm_br], [pm_bb], s=5, c='green')
plt.xlabel('ryan')
plt.ylabel('brad')
plt.title('True Positive')
plt.show()

S = [i for i in range(len(pmr)) if (fnr[i]<10)&(fnb[i]<10)]
len(S)
cl = clusters[706]

f = 'n50_de_bc'
df = pd.read_csv('/home/roozbeh/data/wiki/data/'+f+'.csv')
df['clusters'] = cl
df = df.sort_values(['clusters'], ascending=[True])
df.to_csv('/home/roozbeh/data/wiki/results/test.csv')

plt.scatter(fp_r, fp_b, s=5, c='black')
plt.scatter(fpr, fpb, s=5, c='blue')
plt.scatter([fp_rr], [fp_rb], s=5, c='red')
plt.scatter([fp_br], [fp_bb], s=5, c='green')
plt.xlabel('ryan')
plt.ylabel('brad')
plt.title('False Positive')
plt.show()




plt.scatter(fn_r, fn_b, s=5, c='black')
plt.scatter(fnr, fnb, s=5, c='blue')
plt.scatter([fn_rr], [fn_rb], s=5, c='red')
plt.scatter([fn_br], [fn_bb], s=5, c='green')
plt.xlabel('ryan')
plt.ylabel('brad')
plt.title('False Negative')
plt.show()

