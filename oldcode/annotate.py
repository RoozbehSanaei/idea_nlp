import pre_processing
import pandas as pd
f = 'n50_de_bc'
file = '/home/roozbeh/data/wiki/data/'+f+'.csv'
df = pd.read_csv(file)
words,sentences,selected_lemmas,pos_tagged_lemmas,word_similarity_matrix = pre_processing.pre_process(df)
df.ix[:,4] = pos_tagged_lemmas
f = 'n50_de_bc1'
file = '/home/roozbeh/data/wiki/data/'+f+'.csv'
df.to_csv(file)
