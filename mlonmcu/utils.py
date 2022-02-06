def is_power_of_two(n):
    return (n & (n - 1) == 0) and n != 0
