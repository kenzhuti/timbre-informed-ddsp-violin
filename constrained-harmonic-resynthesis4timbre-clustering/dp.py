def fib(n, memo):
    if n in memo:
        return memo[n]
    if n <= 1:
        return n
    memo[n] = fib(n-1, memo) + fib(n-2, memo)
    return memo[n]


if __name__ == '__main__':
    # Initialize the memo dictionary
    memo = {}
    print(fib(5, memo))  # Output: 5
