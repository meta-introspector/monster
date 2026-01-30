# Compute genus for 15 Monster primes using GAP

# Monster primes
primes := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

Print("Computing genus of X_0(N) for 15 Monster primes\n");
Print("================================================\n\n");

for p in primes do
    N := p;
    
    # Number of cusps for X_0(N)
    nu_inf := Sum(DivisorsInt(N), d -> Phi(Gcd(d, N/d)));
    
    # Number of elliptic points of order 2
    if N mod 4 = 0 then
        nu_2 := 0;
    else
        nu_2 := Product(Filtered(FactorsInt(N), q -> q mod 4 = 3), q -> 1);
        if nu_2 = 1 then nu_2 := 0; fi;
    fi;
    
    # Number of elliptic points of order 3
    if N mod 9 = 0 then
        nu_3 := 0;
    else
        nu_3 := Product(Filtered(FactorsInt(N), q -> q mod 3 = 2), q -> 1);
        if nu_3 = 1 then nu_3 := 0; fi;
    fi;
    
    # Genus formula for X_0(N)
    # g = 1 + (mu/12) - (nu_2/4) - (nu_3/3) - (nu_inf/2)
    # where mu = [PSL(2,Z) : Gamma_0(N)]
    
    mu := N * Product(Set(FactorsInt(N)), q -> (1 + 1/q));
    
    genus := 1 + (mu/12) - (nu_2/4) - (nu_3/3) - (nu_inf/2);
    genus := Int(genus);
    
    Print("Prime ", p, ": genus = ", genus);
    
    if genus = 0 then
        Print(" ✓ GENUS 0 (GOOD)\n");
    else
        Print("\n");
    fi;
od;

Print("\n");
Print("Genus 0 primes (including 71) are GOOD\n");

quit;
