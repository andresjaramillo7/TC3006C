import codecs
import random

lista=[]
with codecs.open("cancer.txt","r","UTF-8") as file:
    for line in file:
        elements=(line.strip('\r\n')).split(",")
        temp=[str(float(x)) if x != "?" else "0.0"  for x in elements[1:]]
        if temp[-1]=="2.0":
            temp[-1]="benign"
        else:    
            temp[-1]="malignant"
        lista.append(temp)

random.shuffle(lista)

print("training")
with codecs.open("cancerTraining.txt","w","UTF-8") as file:
    for x in lista[:599]:
        file.write(",".join(x)+"\n")

print("test")
with codecs.open("cancerTest.txt","w","UTF-8") as file:
    for x in lista[599:]:
        file.write(",".join(x)+"\n")    
