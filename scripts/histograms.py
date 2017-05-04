import jaccard


files = ['n50_de_bc','n50_de_rma','n50_ge_bc','n50_ge_rma','n50_gn_bc','n50_gn_rma','n50_dn_bc','n50_dn_rma','n100_de','n100_dn','n100_ge','n100_gn','n100_de_v3','n100_dn_v3','n100_ge_v3','n100_gn_v3']
match_types =['co_occurance_score','greedy_match_sets']


for f in files:
	for match_type in match_types:
		for i in range(8,13):
			jaccard.save_histogram(f ,i,match_type)


