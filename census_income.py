
filename1 = "data/census-income.data"
file1 = open(filename1, 'rt')
text1 = file1.read()
file1.close

filename2 = "data/census-income.test"
file2 = open(filename2, 'rt')
text2 = file2.read()
file2.close

filename3 = "data/census-income.names"
file3 = open(filename3, 'rt')
text3 = file3.readlines()
text3 = text3[23:67]
file3.close

income_data = []
income_test = []
income_names = []

def filter(arr, text):
    rows = text.split("\n")
    for col in rows:
        row = col.split(", ")
        row[-1] = row[-1].strip(".") 
        arr.append(row)
    return arr

def filter2(arr, text):
    for col in text:
        row = col.split("\t")
        for s in range(len(row)):
            row[s] = row[s].strip("|\n\t ")
        row = [i for i in row if i]
        arr.append(row)
    return arr

filter(income_data, text1)
filter(income_test, text2)
filter2(income_names, text3)

#print(income_data[:2])
#print(income_names[:2])
#print(income_test[:2])