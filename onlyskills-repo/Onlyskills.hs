-- onlyskills.com - zkERDAProlog in Haskell

{-# LANGUAGE DeriveGeneric #-}

module Onlyskills where

import Data.Aeson
import GHC.Generics

data Skill = Skill
  { shardId :: Int
  , prime :: Int
  , skillName :: String
  , skillType :: String
  , command :: String
  , searchCapability :: String
  , zkperfHash :: String
  } deriving (Show, Generic)

instance ToJSON Skill
instance FromJSON Skill

monsterPrimes :: [Int]
monsterPrimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

getPrime :: Int -> Int
getPrime shardId = monsterPrimes !! (shardId `mod` 15)

mkSkill :: Int -> String -> String -> String -> String -> String -> Skill
mkSkill sid name stype cmd cap hash = Skill
  { shardId = sid
  , prime = getPrime sid
  , skillName = name
  , skillType = stype
  , command = cmd
  , searchCapability = cap
  , zkperfHash = hash
  }

toRDF :: Skill -> String
toRDF s = unlines
  [ "<https://onlyskills.com/skill/" ++ skillName s ++ "> rdf:type zkerdfa:SearchSkill ."
  , "<https://onlyskills.com/skill/" ++ skillName s ++ "> zkerdfa:shardId " ++ show (shardId s) ++ " ."
  , "<https://onlyskills.com/skill/" ++ skillName s ++ "> zkerdfa:prime " ++ show (prime s) ++ " ."
  ]

main :: IO ()
main = do
  let skill = mkSkill 29 "expert_system" "search_explicit_search"
                      "cargo run --release --bin expert_system"
                      "explicit_search" "a3f5b2c1d4e6f7a8"
  putStrLn "λ Haskell zkERDAProlog Skill Registry"
  print skill
  putStrLn $ toRDF skill
  putStrLn "∞ 71 Shards in Haskell ∞"
