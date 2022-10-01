from collections import defaultdict
linedict = defaultdict(lambda:0)
with open('ux.txt', 'r') as reader:
    for line in reader:
        line  = line.strip()
        linedict[line] +=1

for k, v in linedict.items():
    if v>1:
        print(k,v)