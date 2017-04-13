import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

def cross1(cl1):
	M = np.zeros((len(cl1),len(cl1)))
	for i in range(len(cl1)):
		for j in range(len(cl1)):
			if (cl1[i]==cl1[j]):
				M[i,j] = 1
	return(M)


file = "all_indexed2.csv"
df = pd.read_csv(file)
dmc = list(df.ix[:,9])
pmc = list(df.ix[:,10])
fpc = list(df.ix[:,12])
fnc = list(df.ix[:,13])


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


dm = []
pm = []
fp = []
fn = []

M0 = Mr

for j in range(1,20):
	for i in range(10000):
		cl = [random.randint(0,j) for r in range(50)]
		M = cross1(cl)
		dm.append(float(np.sum(abs(M-M0))))
		pm.append(float(np.sum(np.multiply(M,M0))))
		fp.append(float(np.sum((M>M0)*1)))
		fn.append(float(np.sum((M<M0)*1)))

dmr = float(np.sum(abs(Mr-M0)))
pmr = float(np.sum(np.multiply(Mr,M0)))
fpr = float(np.sum((Mr>M0)*1))
fnr = float(np.sum((Mr<M0)*1))

dmb = float(np.sum(abs(Mb-M0)))
pmb = float(np.sum(np.multiply(Mb,M0)))
fpb = float(np.sum((Mb>M0)*1))
fnb = float(np.sum((Mb<M0)*1))

plt.scatter(dm, pm, s=5, c='black')
plt.scatter(dmc, pmc, s=5, c='blue')
plt.scatter([dmr],[pmr],s=20, c='red')
plt.scatter([dmb],[pmb],s=20, c='green')
plt.xlabel('mismatch')
plt.ylabel('match')
plt.show()

plt.scatter(fp, fn, s=5, c='black')
plt.scatter(fnc, fpc, s=5, c='blue')
plt.scatter([fpr],[fnr],s=20, c='red')
plt.scatter([fpb],[fnb],s=20, c='green')
plt.show()
