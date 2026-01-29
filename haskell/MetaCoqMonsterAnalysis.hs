{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE LambdaCase #-}

module MetaCoqMonsterAnalysis where

import GHC.Generics
import Data.List (intercalate, sort, group)
import qualified Data.Map as M

-- Import the extracted MetaCoq types
data Positive = XI Positive | XO Positive | XH
  deriving (Generic, Show)

data N = N0 | Npos Positive
  deriving (Generic, Show)

data Nat = O | S Nat
  deriving (Generic, Show)

data Byte = X00 | X01 | XFF  -- Simplified
  deriving (Generic, Show)

data MyString = EmptyString | String Byte MyString
  deriving (Generic, Show)

data UniversalLevel = Lzero | Level MyString | Lvar Nat
  deriving (Generic, Show)

-- MetaCoq Term structure
data Term
  = TRel Nat
  | TVar MyString
  | TProd Name Term Term
  | TLambda Name Term Term
  | TApp Term [Term]
  | TConst MyString
  deriving (Generic, Show)

data Name = Anonymous | Named MyString
  deriving (Generic, Show)

-- Monster Primes
monsterPrimes :: [Int]
monsterPrimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

-- Monster Analysis Results (from our scans)
data MonsterData = MonsterData
  { totalFiles :: Int
  , primeDistribution :: M.Map Int Int
  , maxDepth :: Int
  , filesWithPrime71 :: [String]
  } deriving (Show)

-- Our collected data
metacoqMonsterData :: MonsterData
metacoqMonsterData = MonsterData
  { totalFiles = 584
  , primeDistribution = M.fromList
      [ (2, 3781)
      , (3, 668)
      , (5, 121)
      , (7, 34)
      , (11, 20)
      , (13, 28)
      , (17, 8)
      , (19, 8)
      , (23, 3)
      , (29, 1)
      , (31, 1)
      , (41, 1)
      , (47, 2)
      , (59, 1)
      , (71, 1)  -- THE MONSTER!
      ]
  , maxDepth = 7  -- Parenthesis depth (need AST depth!)
  , filesWithPrime71 = ["utils/theories/ByteCompare.v"]
  }

spectralMonsterData :: MonsterData
spectralMonsterData = MonsterData
  { totalFiles = 73
  , primeDistribution = M.fromList
      [ (2, 401)
      , (3, 78)
      , (5, 18)
      , (7, 19)
      , (11, 8)
      , (13, 5)
      , (71, 1)  -- THE MONSTER!
      ]
  , maxDepth = 7
  , filesWithPrime71 = ["algebra/ring.hlean"]
  }

allSourcesData :: MonsterData
allSourcesData = MonsterData
  { totalFiles = 10573
  , primeDistribution = M.fromList
      [ (2, 52197 + 10421 + 14584 + 461 + 420)  -- All sources
      , (3, 4829 + 3540 + 2539 + 75 + 167)
      , (5, 848 + 1987 + 653 + 21 + 29)
      , (7, 228 + 840 + 331 + 5 + 26)
      , (11, 690 + 224 + 76 + 0 + 8)
      , (13, 144 + 91 + 24 + 1 + 4)
      , (17, 142 + 45 + 19 + 0 + 3)
      , (19, 129 + 44 + 19 + 0 + 1)
      , (23, 92 + 78 + 15 + 1 + 2)
      , (29, 165 + 22 + 6 + 0 + 0)
      , (31, 191 + 77 + 4 + 0 + 5)
      , (41, 1 + 13 + 1 + 0 + 0)
      , (47, 2 + 8 + 2 + 0 + 0)
      , (59, 11 + 51 + 3 + 0 + 0)
      , (71, 4 + 4 + 0 + 0 + 0)  -- 8 total Monster files!
      ]
  , maxDepth = 7
  , filesWithPrime71 = 
      [ "Mathlib: Pi/Bounds, ModCases, Distribution, SchwartzZippel"
      , "Spectral: ring.hlean"
      , "Vericoding: 4 files"
      ]
  }

-- Measure AST depth
termDepth :: Term -> Int
termDepth (TRel _) = 1
termDepth (TVar _) = 1
termDepth (TConst _) = 1
termDepth (TProd _ t1 t2) = 1 + max (termDepth t1) (termDepth t2)
termDepth (TLambda _ t1 t2) = 1 + max (termDepth t1) (termDepth t2)
termDepth (TApp t ts) = 1 + maximum (termDepth t : map termDepth ts)

-- Check if term reaches Monster depth
isMonsterTerm :: Term -> Bool
isMonsterTerm t = termDepth t >= 46

-- Determine Monster shell (0-9)
determineShell :: [Int] -> Int
determineShell primes
  | 71 `elem` primes = 9  -- Monster!
  | any (`elem` [31,41,47,59]) primes = 8  -- Deep Resonance
  | any (`elem` [17,19,23,29]) primes = 7  -- Wave Crest
  | 13 `elem` primes = 6
  | 11 `elem` primes = 5
  | 7 `elem` primes = 4
  | 5 `elem` primes = 3
  | 3 `elem` primes = 2
  | 2 `elem` primes = 1
  | otherwise = 0  -- Pure logic

-- Generate GraphQL schema with Monster analysis
generateMonsterGraphQL :: IO ()
generateMonsterGraphQL = do
  putStrLn "# MetaCoq Monster Analysis GraphQL Schema"
  putStrLn ""
  putStrLn "type Query {"
  putStrLn "  # MetaCoq queries"
  putStrLn "  term(id: ID!): Term"
  putStrLn "  allTerms: [Term!]!"
  putStrLn "  "
  putStrLn "  # Monster analysis"
  putStrLn "  monsterAnalysis: MonsterAnalysis!"
  putStrLn "  findMonsterTerms(minDepth: Int = 46): [Term!]!"
  putStrLn "  primeDistribution: [PrimeCount!]!"
  putStrLn "  shellDistribution: [ShellCount!]!"
  putStrLn "  "
  putStrLn "  # Cross-source comparison"
  putStrLn "  compareWithMathlib: Comparison!"
  putStrLn "  compareWithSpectral: Comparison!"
  putStrLn "}"
  putStrLn ""
  putStrLn "type MonsterAnalysis {"
  putStrLn "  totalFiles: Int!"
  putStrLn "  maxDepth: Int!"
  putStrLn "  isMonster: Boolean!  # depth >= 46?"
  putStrLn "  primeDistribution: [PrimeCount!]!"
  putStrLn "  filesWithPrime71: [String!]!"
  putStrLn "  hypothesis: String!"
  putStrLn "}"
  putStrLn ""
  putStrLn "type PrimeCount {"
  putStrLn "  prime: Int!"
  putStrLn "  count: Int!"
  putStrLn "  shell: Int!"
  putStrLn "  emoji: String!"
  putStrLn "}"
  putStrLn ""
  putStrLn "type Comparison {"
  putStrLn "  source1: String!"
  putStrLn "  source2: String!"
  putStrLn "  commonPrimes: [Int!]!"
  putStrLn "  uniquePrimes1: [Int!]!"
  putStrLn "  uniquePrimes2: [Int!]!"
  putStrLn "  correlation: Float!"
  putStrLn "}"

-- Export to CSV (Parquet-compatible)
exportToCSV :: MonsterData -> String -> IO ()
exportToCSV mdata filename = do
  let header = "prime,count,shell,percentage"
  let total = sum $ M.elems (primeDistribution mdata)
  let rows = [ intercalate "," 
                [ show prime
                , show count
                , show (determineShell [prime])
                , show (fromIntegral count / fromIntegral total * 100 :: Double)
                ]
             | (prime, count) <- M.toList (primeDistribution mdata)
             ]
  writeFile filename $ unlines (header : rows)
  putStrLn $ "✅ Exported to " ++ filename

-- Main analysis
main :: IO ()
main = do
  putStrLn "🔬 MetaCoq Monster Analysis"
  putStrLn (replicate 60 '=')
  putStrLn ""
  
  putStrLn "📊 COLLECTED DATA:"
  putStrLn (replicate 60 '-')
  putStrLn $ "MetaCoq files: " ++ show (totalFiles metacoqMonsterData)
  putStrLn $ "Spectral files: " ++ show (totalFiles spectralMonsterData)
  putStrLn $ "All sources: " ++ show (totalFiles allSourcesData)
  putStrLn ""
  
  putStrLn "👹 PRIME 71 LOCATIONS:"
  putStrLn (replicate 60 '-')
  mapM_ putStrLn (filesWithPrime71 allSourcesData)
  putStrLn ""
  
  putStrLn "📊 PRIME DISTRIBUTION (All Sources):"
  putStrLn (replicate 60 '-')
  let dist = M.toList $ primeDistribution allSourcesData
  let total = sum $ map snd dist
  mapM_ (\(p, c) -> putStrLn $ 
    "Prime " ++ show p ++ ": " ++ show c ++ 
    " (" ++ show (fromIntegral c / fromIntegral total * 100 :: Double) ++ "%)")
    dist
  putStrLn ""
  
  putStrLn "🎯 MONSTER HYPOTHESIS:"
  putStrLn (replicate 60 '-')
  putStrLn "If MetaCoq AST depth >= 46:"
  putStrLn "  → Matches 2^46 in Monster order"
  putStrLn "  → Binary tree with 46 levels"
  putStrLn "  → THE STRUCTURE IS THE MONSTER!"
  putStrLn ""
  putStrLn "Current max depth: 7 (parenthesis nesting)"
  putStrLn "Need: Actual AST depth measurement"
  putStrLn ""
  
  putStrLn "📊 Generating GraphQL Schema..."
  generateMonsterGraphQL
  putStrLn ""
  
  putStrLn "📊 Exporting to CSV..."
  exportToCSV allSourcesData "monster_primes_all_sources.csv"
  exportToCSV metacoqMonsterData "monster_primes_metacoq.csv"
  exportToCSV spectralMonsterData "monster_primes_spectral.csv"
  putStrLn ""
  
  putStrLn "✅ Analysis complete!"
  putStrLn ""
  putStrLn "🎯 Next steps:"
  putStrLn "  1. Use MetaCoq.Template.Quote to get actual AST"
  putStrLn "  2. Measure true AST depth (not just parentheses)"
  putStrLn "  3. Find terms with depth >= 46"
  putStrLn "  4. PROVE: MetaCoq IS the Monster!"
