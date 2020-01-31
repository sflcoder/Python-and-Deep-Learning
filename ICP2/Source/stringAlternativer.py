def string_alternative(input):
    # returns every other char of a given string
    wordNum = len(input)
    str = ''
    for i in range(int(wordNum)):
        if i%2 == 0:
            str += input[i]
    return str



originalStr = input ('please input the string: ')
print(string_alternative(originalStr))