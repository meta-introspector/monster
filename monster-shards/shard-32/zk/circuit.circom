pragma circom 2.0.0;

template MonsterShard32() {
    signal input neurons[100];
    signal output valid;
    
    // Verify all neurons divisible by 32
    var sum = 0;
    for (var i = 0; i < 100; i++) {
        sum += neurons[i] % 32;
    }
    
    valid <== (sum == 0) ? 1 : 0;
}

component main = MonsterShard32();
