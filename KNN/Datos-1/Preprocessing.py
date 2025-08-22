import codecs
import random

lista=[]
with codecs.open("dataset.txt","r","UTF-8") as file:
    for line in file:
        elements=(line.strip('\n')).split(",")
        temp=[str(float(x)) for x in elements]
        if temp[-1]=="0.0":
            temp[-1]="Absence"
        else:    
            temp[-1]="Present"
        lista.append(temp)

random.shuffle(lista)

print("training")
with codecs.open("training.txt","w","UTF-8") as file:
    for x in lista[:242]:
        file.write(",".join(x)+"\n")

print("test")
with codecs.open("test.txt","w","UTF-8") as file:
    for x in lista[242:]:
        file.write(",".join(x)+"\n")    
