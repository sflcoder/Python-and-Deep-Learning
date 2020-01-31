infile = open('wordCount.txt','r')
WordsCounts = dict()
line = infile.readline()
print("The first line: ")
while line != "":
    for word in line.split():
        if word in WordsCounts:
            WordsCounts[word] += 1
        else:
            WordsCounts[word] = 1
    line = infile.readline()
print(WordsCounts)

infile = open('output2.txt','w')
for word in WordsCounts:
    appendStr = word + ": " + str(WordsCounts[word])
    print(appendStr)
    infile.write(appendStr)
    infile.write('\n')