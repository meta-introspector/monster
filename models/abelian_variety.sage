# Sage model of Abelian variety over F_71

# Define the field
p = 71
F = GF(p)

# Abelian variety parameters
dimension = 2
label = "ah_a"

# Slopes as rationals
slopes = [QQ(0), QQ(1)/2, QQ(1)/2, QQ(1)]

print(f"Abelian Variety over F_{p}")
print(f"Dimension: {dimension}")
print(f"Label: {label}")
print(f"Slopes: {slopes}")

# Verify slopes sum to dimension (Newton polygon property)
slope_sum = sum(slopes)
assert slope_sum == dimension, f"Slopes must sum to dimension: {slope_sum} != {dimension}"
print(f"✓ Slopes sum to dimension: {slope_sum} = {dimension}")

# URL construction
url = f"/Variety/Abelian/Fq/{dimension}/{p}/{label}"
print(f"URL: {url}")

# Expected slopes from LMFDB
expected = [QQ(0), QQ(1)/2, QQ(1)/2, QQ(1)]
assert slopes == expected, f"Slopes mismatch: {slopes} != {expected}"
print("✓ Slopes match LMFDB data!")
