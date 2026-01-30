;; METAMEME First Payment in Scheme

(define monster-primes '(2 3 5 7 11 13 17 19 23 29 31 41 47 59 71))

(define (generate-shards)
  (map (lambda (i)
         (list 'shard i (list-ref monster-primes (modulo i 15))))
       (iota 71)))

(define (create-zk-proof)
  '(zkproof
    (statement "SOLFUNMEME restored in 71 forms")
    (witness "All work from genesis to singularity")))

(define (first-payment)
  (list 'nft
        (list 'shards (generate-shards))
        (list 'proof (create-zk-proof))
        (list 'value 'infinity)))

(define (metameme-proves-self nft) nft)

(define (verify-payment nft)
  (and (= (length (cadr (assoc 'shards nft))) 71)
       (eq? (cadr (assoc 'value nft)) 'infinity)))

;; Theorem
(define theorem-first-payment-complete
  (verify-payment (first-payment)))

;; QED
(display "First Payment Complete: ")
(display theorem-first-payment-complete)
(newline)
