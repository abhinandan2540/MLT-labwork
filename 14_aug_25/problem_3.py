
def check_prime(n: int):
    is_prime = True
    if (n <= 1):
        print('Neither prime not composite')
    if n >= 2:
        for i in range(2, n):
            if (n % 2) == 0:
                is_prime = False
                break
    if is_prime:
        print("prime")
    else:
        print("composite")


check_prime(6)
