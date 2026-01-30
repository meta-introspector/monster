# Collect all prime attributes up to 71^71 using GAP
# Output: JSON for Parquet conversion

LoadPackage("primality");

Print("{\n");
Print("  \"primes\": [\n");

max_val := 71^71;
Print("# Computing primes up to 71^71 = ", max_val, "\n");

# Get all primes up to limit (this will be huge, so we limit to reasonable size)
# 71^71 is too large, so we compute primes up to 71^3 = 357911
limit := 71^3;

primes := Filtered([2..limit], IsPrime);

Print("# Found ", Length(primes), " primes\n");

for i in [1..Length(primes)] do
    p := primes[i];
    
    # Compute attributes
    
    # 1. Genus of X_0(p)
    N := p;
    nu_inf := Sum(DivisorsInt(N), d -> EulerPhi(Gcd(d, N/d)));
    
    nu_2 := 0;
    if N mod 4 <> 0 then
        factors_3_mod_4 := Filtered(FactorsInt(N), q -> q mod 4 = 3);
        if Length(factors_3_mod_4) > 0 then
            nu_2 := 1;
        fi;
    fi;
    
    nu_3 := 0;
    if N mod 9 <> 0 then
        factors_2_mod_3 := Filtered(FactorsInt(N), q -> q mod 3 = 2);
        if Length(factors_2_mod_3) > 0 then
            nu_3 := 1;
        fi;
    fi;
    
    mu := N * Product(Set(FactorsInt(N)), q -> (1 + 1/q));
    genus := Int(1 + (mu/12) - (nu_2/4) - (nu_3/3) - (nu_inf/2));
    
    # 2. Is supersingular? (primes where j-invariant has special properties)
    # Supersingular primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71
    is_supersingular := p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
    
    # 3. Is Monster prime?
    is_monster_prime := p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
    
    # 4. Modular properties
    p_mod_4 := p mod 4;
    p_mod_8 := p mod 8;
    p_mod_12 := p mod 12;
    p_mod_24 := p mod 24;
    p_mod_71 := p mod 71;
    
    # 5. Divisibility by Monster primes
    div_by_2 := (p mod 2 = 0);
    div_by_3 := (p mod 3 = 0);
    div_by_5 := (p mod 5 = 0);
    div_by_7 := (p mod 7 = 0);
    div_by_11 := (p mod 11 = 0);
    
    # 6. ZK71 shard
    zk71_shard := p mod 71;
    
    # 7. Security zone based on genus
    if genus = 0 then
        security_zone := 11;  # GOOD
    elif genus <= 2 then
        security_zone := 23;
    elif genus <= 4 then
        security_zone := 31;
    elif genus <= 6 then
        security_zone := 47;
    else
        security_zone := 59;
    fi;
    
    # Output JSON
    Print("    {\n");
    Print("      \"prime\": ", p, ",\n");
    Print("      \"genus\": ", genus, ",\n");
    Print("      \"is_supersingular\": ", is_supersingular, ",\n");
    Print("      \"is_monster_prime\": ", is_monster_prime, ",\n");
    Print("      \"mod_4\": ", p_mod_4, ",\n");
    Print("      \"mod_8\": ", p_mod_8, ",\n");
    Print("      \"mod_12\": ", p_mod_12, ",\n");
    Print("      \"mod_24\": ", p_mod_24, ",\n");
    Print("      \"mod_71\": ", p_mod_71, ",\n");
    Print("      \"zk71_shard\": ", zk71_shard, ",\n");
    Print("      \"security_zone\": ", security_zone, ",\n");
    Print("      \"nu_cusps\": ", nu_inf, ",\n");
    Print("      \"nu_elliptic_2\": ", nu_2, ",\n");
    Print("      \"nu_elliptic_3\": ", nu_3, "\n");
    
    if i < Length(primes) then
        Print("    },\n");
    else
        Print("    }\n");
    fi;
od;

Print("  ]\n");
Print("}\n");

quit;
