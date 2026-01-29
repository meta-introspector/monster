-- Analyze extracted MetaCoq terms for Monster structure

data Term 
  = TRel Int
  | TLambda Term Term
  | TApp Term Term
  | TProd Term Term
  deriving Show

-- Measure depth
termDepth :: Term -> Int
termDepth (TRel _) = 1
termDepth (TLambda t1 t2) = 1 + max (termDepth t1) (termDepth t2)
termDepth (TApp t1 t2) = 1 + max (termDepth t1) (termDepth t2)
termDepth (TProd t1 t2) = 1 + max (termDepth t1) (termDepth t2)

-- Test terms
simple = TLambda (TRel 0) (TRel 0)
nested5 = TLambda (TRel 0) (TLambda (TRel 1) (TLambda (TRel 2) (TLambda (TRel 3) (TLambda (TRel 4) (TRel 4)))))

main = do
  putStrLn "🔬 Analyzing MetaCoq Terms"
  putStrLn "=========================="
  putStrLn ""
  putStrLn $ "Simple depth: " ++ show (termDepth simple)
  putStrLn $ "Nested5 depth: " ++ show (termDepth nested5)
  putStrLn ""
  putStrLn "🎯 Looking for depth >= 46 (Monster!)"
  putStrLn $ "Is Monster? " ++ show (termDepth nested5 >= 46)
