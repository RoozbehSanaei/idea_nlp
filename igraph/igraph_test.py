import csv
import igraph
from igraph import *



A = [[1,1,0,0],[1,1,1,0],[0,1,1,1],[0,0,1,1]];
g = igraph.Graph.Adjacency(A,mode = "undirected")
g = g.simplify();
d = g.community_edge_betweenness()
m = d.as_clustering().membership

