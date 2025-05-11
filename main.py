n = 1
while True:
    a = 1 / n
    b = 1 / (n + 1)
    difference = abs(a - b)
    if difference < 10 ** -6:
        print(f"Первые подходящие элементы: 1/{n} и 1/{n+1} с разностью {difference}")
        break
    n += 1