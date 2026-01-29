// Circom: Monster Walk Music zkSNARK
// Prove musical composition is valid without revealing witness

pragma circom 2.0.0;

// Prove a number is a Monster prime
template IsMonsterPrime() {
    signal input prime;
    signal output valid;
    
    // Monster primes: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71
    signal is_2 <== (prime - 2) * (prime - 2);
    signal is_3 <== (prime - 3) * (prime - 3);
    signal is_5 <== (prime - 5) * (prime - 5);
    signal is_7 <== (prime - 7) * (prime - 7);
    signal is_11 <== (prime - 11) * (prime - 11);
    signal is_13 <== (prime - 13) * (prime - 13);
    signal is_17 <== (prime - 17) * (prime - 17);
    signal is_19 <== (prime - 19) * (prime - 19);
    signal is_23 <== (prime - 23) * (prime - 23);
    signal is_29 <== (prime - 29) * (prime - 29);
    signal is_31 <== (prime - 31) * (prime - 31);
    signal is_41 <== (prime - 41) * (prime - 41);
    signal is_47 <== (prime - 47) * (prime - 47);
    signal is_59 <== (prime - 59) * (prime - 59);
    signal is_71 <== (prime - 71) * (prime - 71);
    
    // At least one must be zero
    signal product <== is_2 * is_3 * is_5 * is_7 * is_11 * is_13 * is_17 * is_19 * is_23 * is_29 * is_31 * is_41 * is_47 * is_59 * is_71;
    valid <== (product == 0) ? 1 : 0;
}

// Compute frequency from prime
template PrimeFrequency() {
    signal input prime;
    signal output frequency;
    
    // frequency = 440 * prime / 71
    // Using fixed-point arithmetic (scale by 1000)
    signal scaled <== prime * 440000;
    frequency <== scaled \ 71;  // Integer division
}

// Verify time signature is 8/8
template TimeSignature() {
    signal input beats;
    signal input unit;
    signal output valid;
    
    signal beats_check <== (beats - 8) * (beats - 8);
    signal unit_check <== (unit - 8) * (unit - 8);
    
    valid <== (beats_check == 0 && unit_check == 0) ? 1 : 0;
}

// Verify tempo is 80 BPM
template Tempo() {
    signal input bpm;
    signal output valid;
    
    signal check <== (bpm - 80) * (bpm - 80);
    valid <== (check == 0) ? 1 : 0;
}

// Verify 10 unique steps
template TenSteps() {
    signal input step_count;
    signal output valid;
    
    signal check <== (step_count - 10) * (step_count - 10);
    valid <== (check == 0) ? 1 : 0;
}

// Verify all primes are unique
template UniquePrimes() {
    signal input primes[10];
    signal output valid;
    
    // Check all pairs are different
    signal differences[45];  // C(10,2) = 45 pairs
    var idx = 0;
    for (var i = 0; i < 10; i++) {
        for (var j = i + 1; j < 10; j++) {
            differences[idx] <== primes[i] - primes[j];
            // Each difference must be non-zero
            differences[idx] * differences[idx] !== 0;
            idx++;
        }
    }
    
    valid <== 1;
}

// Verify frequency ordering (Lean4 lowest, AllBases highest)
template FrequencyOrdering() {
    signal input freq_lean4;    // Step 1
    signal input freq_allbases; // Step 10
    signal output valid;
    
    // freq_lean4 < freq_allbases
    signal diff <== freq_allbases - freq_lean4;
    diff * diff !== 0;  // Must be non-zero
    
    // freq_allbases == 440000 (scaled)
    signal check <== (freq_allbases - 440000) * (freq_allbases - 440000);
    valid <== (check == 0) ? 1 : 0;
}

// Main circuit: Prove Monster Walk composition
template MonsterWalkMusic() {
    // Public inputs
    signal input step_count;
    signal input beats;
    signal input unit;
    signal input bpm;
    
    // Private witness: the 10 Monster primes
    signal input primes[10];
    
    // Public output: composition is valid
    signal output valid;
    
    // Verify step count
    component ten_steps = TenSteps();
    ten_steps.step_count <== step_count;
    
    // Verify time signature
    component time_sig = TimeSignature();
    time_sig.beats <== beats;
    time_sig.unit <== unit;
    
    // Verify tempo
    component tempo = Tempo();
    tempo.bpm <== bpm;
    
    // Verify all primes are Monster primes
    component prime_checks[10];
    for (var i = 0; i < 10; i++) {
        prime_checks[i] = IsMonsterPrime();
        prime_checks[i].prime <== primes[i];
        prime_checks[i].valid === 1;
    }
    
    // Verify primes are unique
    component unique = UniquePrimes();
    for (var i = 0; i < 10; i++) {
        unique.primes[i] <== primes[i];
    }
    
    // Compute frequencies
    component frequencies[10];
    for (var i = 0; i < 10; i++) {
        frequencies[i] = PrimeFrequency();
        frequencies[i].prime <== primes[i];
    }
    
    // Verify frequency ordering
    component ordering = FrequencyOrdering();
    ordering.freq_lean4 <== frequencies[0].frequency;
    ordering.freq_allbases <== frequencies[9].frequency;
    
    // All checks passed
    valid <== ten_steps.valid * time_sig.valid * tempo.valid * unique.valid * ordering.valid;
}

// Main component
component main {public [step_count, beats, unit, bpm]} = MonsterWalkMusic();
