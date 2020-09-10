#import csv module
import csv

#open output files in write mode
wf1 = open("out1.csv", 'w', newline='')
wf2 = open("out2.csv", 'w', newline='')
wf3 = open("out3.csv", 'w', newline='')

#get csv writer for each output file
writer1 = csv.writer(wf1)
writer2 = csv.writer(wf2)
writer3 = csv.writer(wf3)

# number of input files
N = 311

#write the first row as column names
a=[]
for i in range(1,769):
	a+=['col_'+ str(i)]
writer1.writerow(a)
writer2.writerow(a)
writer3.writerow(a)
#now process all the input files
for i in range(N):
	# get ith file's name
	file_name= str(i)+'.csv'
	# open ith file in read mode
	f =open(file_name,'r')
	# read file f
	# split it by '\n' character. It will give a list containing all the rows
	# take slice from 276 to 1040 (rows from each file we need)
	data = f.read().split('\n')[276:1044]

	# 3 lists
	new_data1 = [] #containing column 2
	new_data2 = [] #containing column 3
	new_data3 = [] #containing column 4

	# process all the rows in data (this file)
	for j in data:
		# add column 2 to new_data1
		new_data1 += [float(j.split(',')[2][1:-1])]
		# add column 3 to new_data2
		new_data2 += [float(j.split(',')[3][1:-1])]
		# add column 4 to new_data3
		new_data3 += [float(j.split(',')[4][1:-1])]

	# write new_data1 to output file1
	writer1.writerow(new_data1)
	# write new_data2 to output file2
	writer2.writerow(new_data2)
	# write new_data3 to output file3
	writer3.writerow(new_data3)

	# close the current file
	f.close()