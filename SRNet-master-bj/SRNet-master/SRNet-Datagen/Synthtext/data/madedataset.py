import numpy as np
fileRecord = open("numbers.txt", "a", encoding="utf-8")
number = 5000
for i in range(number):
    digit = np.random.randint(10000,50000000)
    print(digit)
    fileRecord.write(str(digit)+"\n")

