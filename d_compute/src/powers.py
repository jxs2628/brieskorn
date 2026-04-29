import subprocess
from sympy import prime
#[prime(i) for i in range(5, 9)]
primes = [2,3]



def d_invariant(p1, p2, p3):
    """
    Wraps the ./d_invariant executable.
    Takes 3 integers, returns 1 integer.
    """
    cmd = ["./d_invariant", str(p1), str(p2), str(p3)]
    
    # run() handles the execution; capture_output keeps it out of your terminal
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    # Convert the string output back to an integer
    return int(result.stdout.strip())


# Now you can use it elegantly in a list comprehension or loop
for p in primes:
    results = [d_invariant(5,11, p**n) for n in range(1,20)]
    print(f'{p}: {results}') 


#[2, 2, 2, 2, 2, 4, 8, 4, 2, 4, 4, 2, 4, 2, 2, 2, 2, 2, 6, 2, 2, 4, 8, 4, 2, 2, 0, 0, 0]
#2^n,7,11

#R={a−1 ​if b≡−1 (moda) or 2a otherwise​