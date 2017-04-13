def cluster_rep(cc):
	mmax = list(cc[0])[0]
	for c in cc:
		mmax = max (mmax,max(c))
	cl = [0]*(mmax+1)
	for j in range(len(cc)):
		for i in cc[j]:
			cl[i] = j
	return cl