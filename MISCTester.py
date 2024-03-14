digits = [5, 8, 2]
elements = []
for i in range(len(digits)):
    row = [0] * i
    row.append(1)
    row = row + ([0] * (len(digits) - 1 - i))
    row.append(digits[i])
    elements.append(row)
# Homogeneous point
row = ([0] * len(digits)) + [1]
elements.append(row)

for row in elements:
    print(row)
