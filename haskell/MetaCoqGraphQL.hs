{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE DataKinds #-}

module MetaCoqGraphQL where

import GHC.Generics
import Data.Text (Text)
import qualified Data.Text as T

-- Simplified MetaCoq AST types
data Term 
  = TRel Int
  | TVar Text
  | TEvar Int
  | TSort Sort
  | TProd Name Term Term
  | TLambda Name Term Term
  | TLetIn Name Term Term Term
  | TApp Term [Term]
  | TConst Text
  | TInd Text Int
  | TConstruct Text Int Int
  | TCase CaseInfo Term Term [Term]
  | TProj Text Term
  | TFix [Term] Int
  | TCoFix [Term] Int
  deriving (Show, Generic)

data Sort
  = SProp
  | SSet
  | SType Int
  deriving (Show, Generic)

data Name
  = Anonymous
  | Named Text
  deriving (Show, Generic)

data CaseInfo = CaseInfo
  { ciInd :: Text
  , ciNpar :: Int
  , ciCstr :: Int
  } deriving (Show, Generic)

-- GraphQL Schema Generation
generateGraphQLSchema :: IO ()
generateGraphQLSchema = do
  putStrLn "# MetaCoq GraphQL Schema"
  putStrLn ""
  putStrLn "type Query {"
  putStrLn "  term(id: ID!): Term"
  putStrLn "  allTerms: [Term!]!"
  putStrLn "  searchByName(name: String!): [Term!]!"
  putStrLn "  termDepth(id: ID!): Int!"
  putStrLn "}"
  putStrLn ""
  putStrLn "type Term {"
  putStrLn "  id: ID!"
  putStrLn "  kind: TermKind!"
  putStrLn "  children: [Term!]!"
  putStrLn "  depth: Int!"
  putStrLn "  name: String"
  putStrLn "  type: Term"
  putStrLn "}"
  putStrLn ""
  putStrLn "enum TermKind {"
  putStrLn "  REL"
  putStrLn "  VAR"
  putStrLn "  EVAR"
  putStrLn "  SORT"
  putStrLn "  PROD"
  putStrLn "  LAMBDA"
  putStrLn "  LETIN"
  putStrLn "  APP"
  putStrLn "  CONST"
  putStrLn "  IND"
  putStrLn "  CONSTRUCT"
  putStrLn "  CASE"
  putStrLn "  PROJ"
  putStrLn "  FIX"
  putStrLn "  COFIX"
  putStrLn "}"
  putStrLn ""
  putStrLn "type Sort {"
  putStrLn "  kind: SortKind!"
  putStrLn "  level: Int"
  putStrLn "}"
  putStrLn ""
  putStrLn "enum SortKind {"
  putStrLn "  PROP"
  putStrLn "  SET"
  putStrLn "  TYPE"
  putStrLn "}"

-- Measure term depth (looking for 46!)
termDepth :: Term -> Int
termDepth term = case term of
  TRel _ -> 1
  TVar _ -> 1
  TEvar _ -> 1
  TSort _ -> 1
  TConst _ -> 1
  TProd _ t1 t2 -> 1 + max (termDepth t1) (termDepth t2)
  TLambda _ t1 t2 -> 1 + max (termDepth t1) (termDepth t2)
  TLetIn _ t1 t2 t3 -> 1 + maximum [termDepth t1, termDepth t2, termDepth t3]
  TApp t ts -> 1 + maximum (termDepth t : map termDepth ts)
  TInd _ _ -> 1
  TConstruct _ _ _ -> 1
  TCase _ t1 t2 ts -> 1 + maximum (termDepth t1 : termDepth t2 : map termDepth ts)
  TProj _ t -> 1 + termDepth t
  TFix ts _ -> 1 + maximum (map termDepth ts)
  TCoFix ts _ -> 1 + maximum (map termDepth ts)

-- Check if depth reaches Monster level (46)
isMonsterDepth :: Term -> Bool
isMonsterDepth t = termDepth t >= 46

main :: IO ()
main = do
  putStrLn "🔬 MetaCoq GraphQL Schema Generator"
  putStrLn "===================================="
  putStrLn ""
  generateGraphQLSchema
  putStrLn ""
  putStrLn "✅ Schema generated!"
  putStrLn ""
  putStrLn "🎯 To use:"
  putStrLn "  1. Save schema to schema.graphql"
  putStrLn "  2. Implement resolvers for Term queries"
  putStrLn "  3. Measure termDepth to find Monster (depth >= 46)"
