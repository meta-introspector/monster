# Chunk 0: 2 to 1002
primes := Filtered([2..1002], IsPrime);

Print("{\n");
Print("  \"chunk\": 0,\n");
Print("  \"start\": 2,\n");
Print("  \"end\": 1002,\n");
Print("  \"url\": \"https://zkprologml.org/primes/chunk_0\",\n");
Print("  \"chord\": [");

# Musical chord from chunk number (mod 71)
chord_notes := [0 mod 71, (0 * 2) mod 71, (0 * 3) mod 71];
Print(chord_notes[1], ", ", chord_notes[2], ", ", chord_notes[3]);

Print("],\n");
Print("  \"primes\": [\n");

for i in [1..Length(primes)] do
    p := primes[i];
    
    # Compute genus
    N := p;
    nu_inf := Sum(DivisorsInt(N), d -> EulerPhi(Gcd(d, N/d)));
    
    nu_2 := 0;
    if N mod 4 <> 0 then
        if Length(Filtered(FactorsInt(N), q -> q mod 4 = 3)) > 0 then
            nu_2 := 1;
        fi;
    fi;
    
    nu_3 := 0;
    if N mod 9 <> 0 then
        if Length(Filtered(FactorsInt(N), q -> q mod 3 = 2)) > 0 then
            nu_3 := 1;
        fi;
    fi;
    
    mu := N * Product(Set(FactorsInt(N)), q -> (1 + 1/q));
    genus := Int(1 + (mu/12) - (nu_2/4) - (nu_3/3) - (nu_inf/2));
    
    is_monster := p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];
    
    Print("    {\"p\": ", p, ", \"g\": ", genus, ", \"m\": ", is_monster, ", \"s\": ", p mod 71, "}");
    
    if i < Length(primes) then
        Print(",\n");
    else
        Print("\n");
    fi;
od;

Print("  ]\n");
Print("}\n");

quit;
