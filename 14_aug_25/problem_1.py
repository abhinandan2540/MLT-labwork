
# printing n fibonachi series
# 0,1,1,2,3,5,8....
def n_fibonachii(n: int):
    f_0 = 0
    f_1 = 1
    for _ in range(n):
        print(f_0, end=" ")
        f_0, f_1 = f_1, f_0+f_1


n_fibonachii(5)
