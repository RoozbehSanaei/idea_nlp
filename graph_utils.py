import numpy

def costs(cl1,M):
	IC = 0
	EC = 0
	cl = numpy.array(cl1)
	CS = [cl.tolist().count(i) for i in range(max(cl.tolist())+1)]
	for i in range(len(M)):
		for j in range(len(M)):
			if (cl[i]==cl[j]):
				IC = IC + M[i][j]/ (CS[cl[i]])
			else:
				EC = EC + M[i][j]/len(M)
	return IC,EC


def cross(v1,v_t):
	M = numpy.zeros((len(v_t),len(v_t)))
	for V1 in v1:
		for v_1 in V1:
			for v_2 in V1:
					M[v_t.index(v_1),v_t.index(v_2)]=1
	return(M)

def cross1(cl1):
	M = numpy.zeros((len(cl1),len(cl1)))
	for i in range(len(cl1)):
		for j in range(len(cl1)):
			if (cl1[i]==cl1[j]):
				M[i,j] = 1
	return(M)