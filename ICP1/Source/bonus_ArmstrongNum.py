num = int(input('Please enter an integer: '))
sum = 0
tempNum = num
while tempNum != 0:
    remainder = tempNum % 10
    sum += remainder**3
    tempNum = int (tempNum / 10)

if num == sum:
    print( str(num) + ' is an Armstrong number')
else:
    print(str(num) + ' is not an Armstrong number')





