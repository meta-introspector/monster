# Test GCD with values involving 71
def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

# Test cases involving 71
test_cases = [
    (71, 1),
    (71, 71),
    (142, 71),  # 2*71
    (213, 71),  # 3*71
    (71, 2),
    (71, 3),
    (71, 5),
]

print("Testing GCD with prime 71:")
for a, b in test_cases:
    result = gcd(a, b)
    print(f"  gcd({a:3}, {b:2}) = {result:2}")
