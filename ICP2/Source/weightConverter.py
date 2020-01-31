n = input("How many weights of students do you have? ")
weights = []
for i in range(int(n)):
    x = int(input("Enter a weight(lbs) >> "))
    # convert the weight to kilograms
    weights.append(x * 0.45359237)
print("The weights of the studets are: ", weights )