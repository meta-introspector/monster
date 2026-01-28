"""Simplified Hilbert modular form computation"""

def hilbert_norm(a, b, d):
    """Compute norm in Q(√d)"""
    return a*a - d*b*b

def is_totally_positive(a, b, d):
    """Check if element is totally positive"""
    return a > 0 and hilbert_norm(a, b, d) > 0

def hilbert_level(d):
    """Compute level of Hilbert modular form"""
    # Simplified: level is related to discriminant
    if d % 71 == 0:
        return 71
    return abs(d)

def compute_fourier_coefficient(n, d):
    """Compute nth Fourier coefficient"""
    # Simplified computation
    result = 0
    for k in range(1, n+1):
        if n % k == 0:
            result += k * hilbert_norm(k, 1, d)
    return result % 71  # Reduce mod 71

# Test with discriminant 71
print("Testing Hilbert modular forms with discriminant 71:")
d = 71

print(f"\nDiscriminant: {d}")
print(f"Level: {hilbert_level(d)}")

# Compute some norms
print(f"\nNorms in Q(√{d}):")
for a in range(1, 6):
    for b in range(0, 3):
        norm = hilbert_norm(a, b, d)
        pos = is_totally_positive(a, b, d)
        print(f"  N({a} + {b}√{d}) = {norm}, totally positive: {pos}")

# Compute Fourier coefficients
print(f"\nFourier coefficients (mod 71):")
for n in range(1, 11):
    coeff = compute_fourier_coefficient(n, d)
    print(f"  a_{n} = {coeff}")
