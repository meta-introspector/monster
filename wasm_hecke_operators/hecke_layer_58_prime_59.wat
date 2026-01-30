;; Hecke Operator Layer 58 Prime 59
(module
  (memory (export "memory") 1)
  (global $layer i32 (i32.const 58))
  (global $prime i32 (i32.const 59))
  (global $num_eigenvalues i32 (i32.const 1000))

  (data (i32.const 0) "0.00 0.00 0.00 0.00 0.00 0.00 0.00 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.01 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.02 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.03 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.04 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.05 0.06 0.06 0.06 0.06 0.06 0.06 0.06 0.06 0.06 0.06 0.06 0.06 0.06 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07 0.07")

  (func $hecke_apply (param $x f32) (result f32)
    (local $sum f32)
    (local $i i32)
    (local.set $sum (f32.const 0.0))
    (local.set $i (i32.const 0))
    (block $break
      (loop $continue
        (br_if $break (i32.ge_u (local.get $i) (global.get $num_eigenvalues)))
        (local.set $sum
          (f32.add (local.get $sum)
            (f32.mul (local.get $x) (f32.const 0.0000))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $continue)
      )
    )
    (local.get $sum)
  )
  (export "hecke_apply" (func $hecke_apply))

  (func $resonates (param $value i32) (result i32)
    (i32.rem_u (local.get $value) (global.get $prime))
  )
  (export "resonates" (func $resonates))
)
