infile = open('wordCount.txt','r')
originalWords = infile.read()
WordsCounts = dict()
for word in originalWords.split():
    if word in WordsCounts:
        WordsCounts[word] += 1
    else:
        WordsCounts[word] = 1

infile = open('output.txt','w')
for word in WordsCounts:
    appendStr = word + ": " + str(WordsCounts[word])
    print(appendStr)
    infile.write(appendStr)
    infile.write('\n')

