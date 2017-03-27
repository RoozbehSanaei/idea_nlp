from mcl_clustering import mcl
import numpy;
A = numpy.random.rand(100,100)
M, clusters = mcl(A)
