originalInput = input('Please enter an integer: ')

if (originalInput == originalInput[::-1]):
    print(originalInput + ' is a palindrome')
else:
    print(originalInput + ' is not a palindrome')

