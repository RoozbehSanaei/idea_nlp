import csv
import igraph
from igraph import *

with open('superset.csv') as input_file:
	rows = csv.reader(input_file, delimiter=';')
	res = list(zip(*rows))

l = len(res);

F = [list(map(float, res[i][1:l])) for i in range(1,l)];
F = [[int(F[i][j]+F[j][i]) for i in range(0,l-1)] for j in range(0,l-1)];


A = [[1,1,0,0],[1,1,1,0],[0,1,1,1],[0,0,1,1]];
g = igraph.Graph.Adjacency(F,mode = "undirected")
g = g.simplify();
d = g.community_edge_betweenness()
m = d.as_clustering().membership

