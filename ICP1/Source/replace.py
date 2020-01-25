originalStr = 'I love playing with python'
wordList = originalStr.split(' ')

result = ''
for word in wordList:
    if word == 'python':
        result += 'pythons'
    else:
        result += word
    result += ' '

print(result)
