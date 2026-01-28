pragma circom 2.0.0;

template MonsterShard17() {
    signal input neurons[100];
    signal output valid;
    
    // Verify all neurons divisible by 17
    var sum = 0;
    for (var i = 0; i < 100; i++) {
        sum += neurons[i] % 17;
    }
    
    valid <== (sum == 0) ? 1 : 0;
}

component main = MonsterShard17();
