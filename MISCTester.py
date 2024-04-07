digits = [1, 4, 2]

elements = []
for i in range(len(digits)):
    row = [0] * i
    row.append(digits[i])
    row = row + ([0] * (len(digits) - i))
    elements.append(row)
# Homogeneous point
row = ([0] * len(digits)) + [1]
elements.append(row)

for row in elements:
    print(row)
