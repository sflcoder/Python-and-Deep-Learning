x1 = {1: 5, 3: 4, 4: 3, 2: 1, 0: 0}
x2 = {5: 5, 7: 19, 6: 8, 9: 16, 8: 7}
x = dict()

# concat dictionary and sort it
y = {k: v for k, v in sorted({**x1, **x2}.items(), key=lambda item: item[1])}
print(y)
