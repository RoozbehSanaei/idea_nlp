import pickle
import xlwt
import pandas as pd
import csv


def save_to_file(variable,filename):
	pickle.dump( variable, open(filename, "wb" ) )

def load_from_file(filename):
	variable = pickle.load( open(filename, "rb" ) )
	return variable


def xls_with_header(headers,M,filename):
	book = xlwt.Workbook(encoding="utf-8")
	sheet1 = book.add_sheet("sheet 1")
	for i in range(len(headers)):
		sheet1.write(0, i+1, headers[i])
		sheet1.write(i+1, 0, headers[i])

	for i in range(len(headers)):
		for j in range(len(headers)):
			sheet1.write(i+1, j+1, M[i][j])

	book.save(filename)

def extract_col(file,i):
	df = pd.read_csv(file)
	headers = df.ix[:,i].tolist()
	return headers

def cols_to_csv(inps,cols,f):
	inps_t = list(map(list, zip(*inps)))
	panda_results = pd.DataFrame(inps_t)
	panda_results.columns = cols
	panda_results.to_csv(f)

def rows_to_csv(inps_t,cols,f):
	panda_results = pd.DataFrame(inps_t)
	panda_results.columns = cols
	panda_results.to_csv(f)


def save_to_csv(M,filename):
	with open("clusters0.csv", "w") as output_file:
		writer = csv.writer(output_file, delimiter=';')
		writer.writerows(M)




def load_from_csv(M,filename):
	with open("clusters0.csv", "w") as output_file:
		reader = csv.reader(output_file, delimiter=';')
		reader.writerows(M)


def save_to_txt_file(M,filename):
	thefile = open(filename, 'w')
	for item in M:
		thefile.write("%s\n" % item)
