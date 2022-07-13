f = open('INDEX_general_PL_data.2020')
stringls = f.readlines()
f.close()
newls = []
counter = 0
for i in stringls:
    if counter < 19023:
        newstring = i[:4] + ',' + i[19:23]
    else:
        newstring = i[:4] + ',' + i[18:23]
    newls.append(newstring)
    counter += 1

filecont = 'PDB,pK\n' + '\n'.join(newls)
print(filecont)

file = open('bindingdata.csv','w')
file.write(filecont)
file.close()