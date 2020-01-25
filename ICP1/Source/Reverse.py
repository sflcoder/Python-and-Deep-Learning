originalString = input('Please enter a string: ')

print(originalString[-3::-1])


strList = list(originalString)
strList.reverse()
reverseStr = ''.join(strList)
print(reverseStr[2:])


