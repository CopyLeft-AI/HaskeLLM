{- ORMOLU_DISABLE -}
-- LLM Trainer, following the book "Build a Large Language Model (From Scratch)".
{-
 - Copyright 2024 Julia Longtin
 -
 - This program is free software: you can redistribute it and/or modify
 - it under the terms of the GNU Affero General Public License as published by
 - the Free Software Foundation, either version 3 of the License, or
 - (at your option) any later version.
 -
 - This program is distributed in the hope that it will be useful,
 - but WITHOUT ANY WARRANTY; without even the implied warranty of
 - MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 - GNU Affero General Public License for more details.
 -
 - You should have received a copy of the GNU Affero General Public License
 - along with this program.  If not, see <http://www.gnu.org/licenses/>.
 -}

{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE OverloadedStrings #-}

import Prelude (Bool(True, False), Char, Float, Int, IO, Maybe(Just, Nothing), Show, String, (<$>), (<*>), (>>=), (<>), (&&), (/=), (==), (<), (>), (.), ($), (*), (+), (/), (-), concat, concatMap, div, error, exp, fromIntegral, getContents, length, mempty, maybe, mod, not, otherwise, print, pure, putStrLn, read, return, sqrt, show, take, zip)

import qualified Prelude as PL (readFile)

import Data.Aeson (Value(Array, Number, Object), FromJSON(parseJSON), ToJSON(toJSON), (.=), eitherDecode, object, withObject)

import qualified Data.Aeson as A (encode)

import Data.Aeson.Key (Key, toText)

import qualified Data.Aeson.Key as AK (fromString, toString)

import Data.Foldable (Foldable)

import BPE.Base (Id, Merges, Pair, Seq, Vocab, mergesToVocab)

import qualified BPE.Regex as BPER (decode, encode)

import BPE.Regex (gpt2pattern)

import qualified Data.Aeson.KeyMap as DAKM (toList)

import Data.Array.Repa (U, D, Z(Z), (*^), (/^), (+^), backpermute, computeS, extend, extent, fromListUnboxed, map, reshape, slice, sumS, transpose)

import qualified Data.Array.Repa as DAR (Array, toList)

import Data.Array.Repa.Index (DIM1, DIM2, DIM3, DIM4, DIM5, (:.)((:.)))

import Data.Array.Repa.Slice (Any(Any), All(All))

import qualified Data.ByteString as BSS (ByteString, pack, singleton, unpack)

import qualified Data.ByteString.Char8 as BSC (unpack)

import qualified Data.ByteString.Lazy as BSL (ByteString, fromStrict, readFile, toStrict)

import qualified Data.ByteString.Lazy.UTF8 as BSLU (break, drop, lines)

import qualified Data.ByteString.UTF8 as BSU (toString, fromString)

import Data.ByteString.Conversion (toByteString)

import Data.Char (digitToInt, isDigit, isHexDigit)

import Data.Either (Either (Left, Right), lefts, rights)

import Data.HashMap.Strict.InsOrd (InsOrdHashMap, empty, insert, lookup, size)

import qualified Data.HashMap.Strict.InsOrd as DHSI (fromList, toRevList, toList, union)

import Data.Tuple (swap)

import Data.List ((++), drop, elem, foldl, foldr1, head, sort, unfoldr)

import Data.List.Extra (replace)

import Data.List.Split (chunksOf, dropBlanks, oneOf, onSublist, split, splitOneOf)

import Data.List.Unique (sortUniq)

import Data.Maybe (fromMaybe)

import Data.Scientific (toBoundedInteger, toRealFloat)

import Data.Text.Encoding (encodeUtf8)

import qualified Data.Vector as DV (toList)

import Data.Word (Word8)

import Options.Applicative (Parser, ReadM, auto, execParser, fullDesc, header, help, helper, info, long, metavar, option, optional, progDesc, short, str, strOption, switch)

import System.Random (StdGen, mkStdGen, uniformR, random)

-- | A type for encoding an example number.
newtype Example = Example (Int, Int)

-- | Read the example number. always in the format "X.YY" where X is a single digit hexidecimal value, and YY is a 1-2 digit integer.
exampleReader :: ReadM Example
exampleReader = do
  v <- str
  let
    exampleDescription :: [Char]
    exampleDescription = "An example must consist of one character for the chapter (0-9A-F), and two characters for the listing number within the chapter (00-99)."
    findExample :: [Char] -> (Int, Int)
    findExample [] = error $ "empty chapter.\n" <> exampleDescription
    findExample [_] = error $ "no listing number.\n" <> exampleDescription
    findExample [_,'.'] = error $ "no listing number.\n" <> exampleDescription
    findExample [_,_] = error $ "two digit chapter number.\n" <> exampleDescription
    findExample [a,'.',b]
      | isHexDigit a && isDigit b = (digitToInt a, digitToInt b)
      | isHexDigit a = error $ "unable to read listing number.\n" <> exampleDescription
      | isDigit b = error $ "unable to read chapter number.\n" <> exampleDescription
      | otherwise = error $ "unable to read chapter or listing number.\n" <> exampleDescription
    findExample [a,'.',b,c]
      | isHexDigit a && isDigit b && isDigit c = (digitToInt a, digitToInt b*10+digitToInt c)
      | not (isDigit c) = error $ "unable to read last digit of listing number.\n" <> exampleDescription
      | not (isDigit b) = error $ "unable to read first digit of listing number.\n" <> exampleDescription
      | not (isHexDigit a) = error $ "unable to read chapter number.\n" <> exampleDescription
    findExample _ = error $ "complete failure to parse chapter and listing numbers.\n" <> exampleDescription
  return $ Example $ findExample v

-- | Store the state of our command line arguments, after parsing.
data TrainRootOpts =
  TrainRootOpts
    {
      inputFileOpt :: Maybe String
    , dictionaryOpt :: Maybe String
    , mergesOpt :: Maybe String
    , exampleOpt :: Example
    , embeddingDimensionsOpt :: Maybe Int
    , attentionWeightDimensionsOpt :: Maybe Int
    , tokenEmbeddingsOpt :: Maybe String
    , attentionWeightsOpt :: Maybe String
    , dropoutMapsOpt :: Maybe String
    , outProjectionWeightsOpt :: Maybe String
    , verboseFlag :: Maybe Bool
    }

-- | Our parser for our command line options.
trainOpts :: Parser TrainRootOpts
trainOpts =
  TrainRootOpts
  <$> optional (
  strOption
    (    short 'i'
      <> long "infile"
      <> metavar "INPUTFILE"
      <> help "load an ASCII text file for tokenization"
    )
  )
  <*> optional (
  strOption
    (    short 'd'
      <> long "dictionary"
      <> metavar "DICTIONARY"
      <> help "load a JSON formatted dictionary"
    )
  )
  <*> optional (
  strOption
    (    short 'm'
      <> long "merges"
      <> metavar "MERGES"
      <> help "load a JSON formatted list of merges"
    )
  )
  <*>
  option exampleReader
    (    long "example"
      <> short 'e'
      <> metavar "EXAMPLE"
      <> help "which example to run"
    )
  <*> optional (
  option auto
    (    long "embeddingDimensions"
      <> short 'c'
      <> help "the number of dimensions each embedding gets"
      <> metavar "DIMENSIONS"
    ) :: Parser Int
  )
  <*> optional (
  option auto
    (    long "attentionWeightDimensions"
      <> short 'w'
      <> help "the number of dimensions each attention weight gets"
      <> metavar "DIMENSIONS"
    ) :: Parser Int
  )
  <*> optional (
  strOption
    (    short 't'
      <> long "tokenEmbeddings"
      <> metavar "TOKENEMBEDDINGS"
      <> help "load a JSON formatted list of token embeddings"
    )
  )
  <*> optional (
  strOption
    (    short 'a'
      <> long "attentionWeights"
      <> metavar "ATTENTIONWEIGHTS"
      <> help "load a JSON formatted list of attention weights"
    )
  )
  <*> optional (
  strOption
    (    short 'm'
      <> long "dropoutMaps"
      <> metavar "DROPOUTMAPS"
      <> help "load a JSON formatted set of dropout maps"
    )
  )
  <*> optional (
  strOption
    (    short 'o'
      <> long "outProjectionWeights"
      <> metavar "OUTPROJECTIONWEIGHTS"
      <> help "load a JSON formatted set of output projection weights"
    )
  )
  <*> optional (
  switch
    (    long "verbose"
      <> short 'v'
      <> help "whether to be verbose"
    )
  )

-- | A Bacov, A Vocab with the key and value swapped. The native form, when coming out of a JSON file.
newtype Bacov = Bacov (InsOrdHashMap BSS.ByteString Int)

-- | Parse a vocabulary file.
instance FromJSON Bacov where
  parseJSON = withObject "Bacov" (pure . findBacov . DAKM.toList)
    where
      findBacov :: [(Key, Value)] -> Bacov
      findBacov maybeTokenMaps = Bacov tokenMaps
        where
          tokenMaps = tokenMaps' maybeTokenMaps empty
          tokenMaps' :: [(Key, Value)] -> InsOrdHashMap BSS.ByteString Int -> InsOrdHashMap BSS.ByteString Int
          tokenMaps' [] myTokenMaps = myTokenMaps
          tokenMaps' [(k,v)] myTokenMaps = insert (encodeUtf8 $ toText k) (numberFromValue v) myTokenMaps
          tokenMaps' ((k,v):xs) myTokenMaps = insert (encodeUtf8 $ toText k) (numberFromValue v) (tokenMaps' xs myTokenMaps)
          numberFromValue :: Value -> Int
          numberFromValue (Number v) = case toBoundedInteger v of
                                         Nothing -> error $ "failed to find bounded integer for: " <> show v <> "\n"
                                         Just a -> a
          numberFromValue a = error $ "failed to parse " <> show a <> " as a Number."

-- | A typeclass for tokenization.
class Tokenable s where
  -- Establish a vocabulary from a set of strings.
  vocabOfText :: s -> Vocab
  -- Use a vocabulary to reconstruct tokens into a string.
  stringFromTokens :: Vocab -> [Int] -> s
  -- Use a vocabulary to tokenize an input.
  tokensFromString :: Vocab -> s -> [Int]

-- | Our tokenization instance, for arrays of characters.
instance Tokenable [Char] where
  vocabOfText = vocabFromText
  stringFromTokens = getStringFromTokens
  tokensFromString v = getTokensFromString v Nothing

-- | Construct a Vocab from a given segment of text. Basically just tokenize | sort -u.
vocabFromText :: [Char] -> Vocab
vocabFromText input = tokenMaps
  where
    tokenMaps :: InsOrdHashMap Int BSS.ByteString
    tokenMaps = tokenMaps' (zip [0,1..] vocab) empty
    tokenMaps' [] tokenMap = tokenMap
    tokenMaps' [(k,v)] tokenMap = insert k v tokenMap
    tokenMaps' ((k,v):xs) tokenMap = insert k v (tokenMaps' xs tokenMap)
    vocab :: [BSS.ByteString]
    vocab = sortUniq $ splitString input

-- | split up a string into tokens. yes, the barriers between tokens are arbitrary, matching the ones used in the book.
splitString :: [Char] -> [BSS.ByteString]
splitString input = BSU.fromString <$> separateDoubleDash
  where
    separateDoubleDash :: [[Char]]
    separateDoubleDash = concatMap (split (dropBlanks $ onSublist "--")) separatePunctuation
    separatePunctuation :: [[Char]]
    separatePunctuation = concatMap (split (oneOf ",.:;?_!\"()'")) words
    words = splitOneOf " \n" input

-- | Use a list of tokens to tokenize a string. optionally accepts an unknown token.
getTokensFromString :: Vocab -> Maybe Int -> [Char] -> [Int]
getTokensFromString rawVocab unk string = findTokenOfString <$> splitString string
  where
    findTokenOfString :: BSS.ByteString -> Int
    findTokenOfString s = case lookup s bacov of
                               Just v -> v
                               Nothing -> case unk of
                                            Just t -> t
                                            Nothing -> error $ "cannot find a token for \"" <> BSU.toString s <> "\"\n"
    -- the vocabulary backwards: a mapping from value to token.
    bacov :: InsOrdHashMap BSS.ByteString Int
    bacov = DHSI.fromList (swap <$> DHSI.toList rawVocab)

-- | Use a vocabulary to reconstruct a string from a list of tokens.
getStringFromTokens :: Vocab -> [Int] -> [Char]
getStringFromTokens rawVocab tokens = maybeIntersperse ' ' $ findStringOfToken <$> tokens
  where
    maybeIntersperse :: Char -> [BSS.ByteString] -> [Char]
    maybeIntersperse _ [] = []
    maybeIntersperse x xs = foldr1 maybeIntersperse' (BSU.toString <$> xs)
      where
        maybeIntersperse' :: [Char] -> [Char] -> [Char]
        maybeIntersperse' a b = if head b `elem` (",.?!\"()'" :: [Char])
                                  then a ++ b
                                  else a ++ x:b
    findStringOfToken t = case lookup t rawVocab of
                            Just s -> s
                            Nothing -> error $ "cannot find a string for token" <> show t <> "\n"

-- | Count the number of characters in the input file.
-- Corresponds to page 22, listing 2.1.
example_2_1 :: [Char] -> [Char]
example_2_1 text = "Total number of character: " <> show (length text) <> "\n" <> take 99 text

-- | Construct a Vocab for the first 51 tokens of the input file.
-- Corresponds to page 25, listing 2.2.
example_2_2 :: [Char] -> Vocab
example_2_2 = DHSI.fromList . take 51 . DHSI.toRevList . vocabOfText

-- | For example 2.3, they use it twice, we just implement this as two functions: one for encoding, and one for decoding.
-- This is the encoding one.
-- Corresponds to page 27, listing 2.3, encode().
example_2_3_1 :: [Char] -> [Char] -> [Int]
example_2_3_1 text = tokensFromString (vocabOfText text)

-- | For example 2.3, they use it twice, we just implement this as two functions: one for encoding, and one for decoding.
-- This is the decoding one.
-- Corresponds to page 27, listing 2.3, decode().
example_2_3_2 :: [Char] -> [Int] -> [Char]
example_2_3_2 text = stringFromTokens (vocabOfText text)

-- | Example 2.4 has several sub examples. This one prints the last 5 tokens in our extended vocabulary.
-- Corresponds to page 30, bottom half of page.
example_2_4_1 :: [Char] -> Vocab
example_2_4_1 text = DHSI.fromList $ drop (length vocab - 5) $ sort $ DHSI.toList vocab
  where
    vocab = extendVocabGPT2Unk $ vocabOfText text

-- | Example 2.4 has several sub examples. This one gives us the tokens at the top of page 32.
example_2_4_2 :: [Char] -> [Char] -> [Int]
example_2_4_2 text = getTokensFromString vocab (Just $ length vocab - 1)
  where
    vocab = extendVocabGPT2Unk $ vocabOfText text

-- | Example 2.4 has several sub examples. This one gives us the reconstituted string on page 32.
example_2_4_3 :: [Char] -> [Int] -> [Char]
example_2_4_3 text = stringFromTokens vocab
  where
    vocab = extendVocabGPT2Unk $ vocabOfText text

-- | Tokenize GPT2 style.
-- When given the GPT2 vocabulary and merges files, along with the string:
-- "Hello, do you like tea?" <> " <|endoftext|> " <> "In the sunlit terraces of someunknownPlace."
-- Produces the sequence of token IDs on page 33.
example_2_5_1 :: [Char] -> InsOrdHashMap Pair Id -> InsOrdHashMap Id BSS.ByteString -> Seq
example_2_5_1 text mergesTXT jsonDictionary
  | mergeDictionary == jsonDictionary = encodeExtensionsGPT2 $ BPER.encode initSeqGPT2 mergesTXT gpt2pattern mempty (BSU.fromString text)
  | otherwise = error $ "Dictionaries not identical:\nTEXT: " <> show (take 100 $ drop 50200 $ sort $ DHSI.toList mergeDictionary) <> "\n"
                     <> "JSON: " <> show (take 100 $ drop 50200 $ sort $ DHSI.toList jsonDictionary) <> "\n"
  where
    -- a dictionary from a merge file.
    mergeDictionary = extendVocabGPT2 $ mergesToVocab mergesTXT initVocabGPT2

-- | De-Tokenize GPT2 style.
-- When given the GPT2 vocabulary and merges files, along with the sequence:
-- [15496,11,466,345,588,8887,30,220,50256,554,262,4252,18250,8812,2114,286,617,34680,27271,13]
-- Produces the same output as: "Hello, do you like tea?" <> " <|endoftext|> " <> "In the sunlit terraces of someunknownPlace."
example_2_5_2 :: Seq -> InsOrdHashMap Pair Id -> InsOrdHashMap Id BSS.ByteString -> BSS.ByteString
example_2_5_2 seq mergesTXT jsonDictionary
  | mergeDictionary == jsonDictionary = respaceGPT2 $ BPER.decode jsonDictionary mempty seq
  | otherwise = error $ "Dictionaries not identical:\nTEXT: " <> show (take 100 $ drop 50200 $ sort $ DHSI.toList mergeDictionary) <> "\n"
                     <> "JSON: " <> show (take 100 $ drop 50200 $ sort $ DHSI.toList jsonDictionary) <> "\n"
  where
    -- a dictionary from a merge file.
    mergeDictionary = extendVocabGPT2 $ mergesToVocab mergesTXT initVocabGPT2

-- | Count the number of tokens in the given text.
-- Implements page 35.
example_2_6_1 :: [Char] -> InsOrdHashMap Pair Id -> Extensions -> Int
example_2_6_1 text mergesTXT extensions = length $ encodeExtensions extensions $ BPER.encode initSeqGPT2 mergesTXT gpt2pattern mempty $ BSU.fromString text

-- | Return tokens 51, 52, 53, and 54.
-- Implements the 'x' result of the top of page 36.
example_2_6_2 :: [Char] -> InsOrdHashMap Pair Id -> Extensions -> Seq
example_2_6_2 text mergesTXT extensions = take 4 $ drop 50 $ encodeExtensions extensions $ BPER.encode initSeqGPT2 mergesTXT gpt2pattern mempty $ BSU.fromString text

-- | Return tokens 52, 53, 54, and 55.
-- Implements the 'y' result of the top of page 36.
example_2_6_3 :: [Char] -> InsOrdHashMap Pair Id -> Extensions -> Seq
example_2_6_3 text mergesTXT extensions = take 4 $ drop 51 $ encodeExtensions extensions $ BPER.encode initSeqGPT2 mergesTXT gpt2pattern mempty $ BSU.fromString text

-- | Produce an example of a next word prediction task training dataset.
-- Implements the output with IDs and arrows in it, in the middle of page 36.
example_2_6_4 :: [Char] -> InsOrdHashMap Pair Id -> Extensions -> [Char]
example_2_6_4 text mergesTXT extensions = rotateShow $ take 5 $ drop 50 $ encodeExtensions extensions $ BPER.encode initSeqGPT2 mergesTXT gpt2pattern mempty $ BSU.fromString text
  where
    rotateShow [] = error "too few."
    rotateShow [_] = error "too few."
    rotateShow xs = rotateShow' [] xs
    rotateShow' _ [] = ""
    rotateShow' _ [_] = ""
    rotateShow' [] [x,y] = show [x] <> " ----> " <> show y <> "\n"
    rotateShow' [] (x:y:xs) = show [x] <> " ----> " <> show y <> "\n" <> rotateShow' [x] (y:xs)
    rotateShow' a [x,y] = show (a <> [x]) <> " ----> " <> show y <> "\n"
    rotateShow' a (x:y:xs) = show (a <> [x]) <> " ----> " <> show y <> "\n" <> rotateShow' (a <> [x]) (y:xs)

-- | Produce a human readable example of a next word prediction task training dataset.
-- Implements the output with words and arrows in it, at the bottom of page 36.
example_2_6_5 :: [Char] -> InsOrdHashMap Pair Id -> Extensions -> [Char]
example_2_6_5 text mergesTXT extensions = BSC.unpack $ rotateShow $ take 5 $ drop 50 $ encodeExtensions extensions $ BPER.encode initSeqGPT2 mergesTXT gpt2pattern mempty $ BSU.fromString text
  where
    rotateShow [] = error "too few."
    rotateShow [_] = error "too few."
    rotateShow xs = rotateShow' [] xs
    rotateShow' _ [] = ""
    rotateShow' _ [_] = ""
    rotateShow' [] [x,y] = respaceGPT2 (BPER.decode mergeDictionary mempty [x]) <> " ----> "
                        <> respaceGPT2 (BPER.decode mergeDictionary mempty [y]) <> "\n"
    rotateShow' [] (x:y:xs) = respaceGPT2 (BPER.decode mergeDictionary mempty [x]) <> " ----> "
                           <> respaceGPT2 (BPER.decode mergeDictionary mempty [y]) <> "\n" <> rotateShow' [x] (y:xs)
    rotateShow' a [x,y] = respaceGPT2 (BPER.decode mergeDictionary mempty (a <> [x])) <> " ----> "
                        <> respaceGPT2 (BPER.decode mergeDictionary mempty [y]) <> "\n"
    rotateShow' a (x:y:xs) = respaceGPT2 (BPER.decode mergeDictionary mempty (a <> [x])) <> " ----> "
                             <> respaceGPT2 (BPER.decode mergeDictionary mempty [y]) <> "\n" <> rotateShow' (a <> [x]) (y:xs)
    -- a dictionary from a merge file.
    mergeDictionary = extendVocabGPT2 $ mergesToVocab mergesTXT initVocabGPT2

-- | A 2D vector of Ints
newtype NVec2I = NVec2I (DAR.Array U DIM2 Int)
  deriving Show

-- | A 3D vector of Ints.
newtype NVec3I = NVec3I (DAR.Array U DIM3 Int)
  deriving Show

-- | Produce the first four tokens, then the second four tokens, where the second consists of the last three of the first, and the 5th token.
-- Produces the same output as the 'first batch' output on page 39.
example_2_6_6 :: [Char] -> InsOrdHashMap Pair Id -> Extensions -> NVec2I
example_2_6_6 text mergesTXT extensions = NVec2I $ fromListUnboxed (Z :. 2 :. 4) $ 
                                          take 4 (encodeExtensions extensions $ BPER.encode initSeqGPT2 mergesTXT gpt2pattern mempty $ BSU.fromString text) ++
                                          take 4 (drop 1 $ encodeExtensions extensions $ BPER.encode initSeqGPT2 mergesTXT gpt2pattern mempty $ BSU.fromString text)


-- | Produce the second four tokens, then the third four tokens, where the second consists of the last three of the first, and the 5th token.
-- Produces the same output as the 'second batch' output on page 39.
example_2_6_7 :: [Char] -> InsOrdHashMap Pair Id -> Extensions -> NVec2I
example_2_6_7 text mergesTXT extensions = NVec2I $ fromListUnboxed (Z :. 2 :. 4) $
                                          take 4 (drop 1 $ encodeExtensions extensions $ BPER.encode initSeqGPT2 mergesTXT gpt2pattern mempty $ BSU.fromString text) ++
                                          take 4 (drop 1 $ drop 1 $ encodeExtensions extensions $ BPER.encode initSeqGPT2 mergesTXT gpt2pattern mempty $ BSU.fromString text)

-- | Produce a more realistic sample, in 3 dimensions.
-- Produces the two tensors spanning pages 40 and 41.
example_2_6_8 :: [Char] -> InsOrdHashMap Pair Id -> Extensions -> NVec3I
example_2_6_8 text mergesTXT extensions = NVec3I $ fromListUnboxed (Z :. 2 :. 8 :. 4) $
                                          take (8*4) (encodeExtensions extensions $ BPER.encode initSeqGPT2 mergesTXT gpt2pattern mempty $ BSU.fromString text) ++
                                          take (8*4) (drop 1 $ encodeExtensions extensions $ BPER.encode initSeqGPT2 mergesTXT gpt2pattern mempty $ BSU.fromString text)

-- | Our Hyperparameters. The configuration settings of the model we are working with.
data HyperParams =
  HyperParams
    {
      embeddingDim :: Int -- How many dimensions our embeddings will be.
    , attentionWeightDim :: Int -- How many dimensions our attention weights will be.
    }
  deriving Show

-- A two dimensional vector of Floats.
newtype NVec2F = NVec2F (DAR.Array U DIM2 Float)
  deriving Show

-- A three dimensional vector of floats.
newtype NVec3F = NVec3F (DAR.Array U DIM3 Float)
  deriving Show

-- We're getting a bit creative in this section, because there is no way we can get our haskell random seed to line up with pytorch.
-- Instead, we're performing serialization / deserialization of the values from the book, constructing and displaying our own random sets, and performing operations on each.

-- | Read from a JSON file and display a set of token embeddings.
-- When given 3d6-token_embeddings.json and 6_token-vocab.json , produces the embedding layer weight matrix on page 42.
example_2_7_1 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> NVec2F
example_2_7_1 (HyperParams embeddingDimensions _) jsonDictionary tokenEmbeddings@(NVec2F rawTokenEmbeddings)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | otherwise = tokenEmbeddings
  where
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | Generate a random set of embeddings.
example_2_7_2 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F
example_2_7_2 = randomEmbeddings

-- | Generate a random set of embeddings.
randomEmbeddings :: HyperParams -> Vocab -> NVec2F
randomEmbeddings (HyperParams embeddingDimensions _) vocab = NVec2F $ fromListUnboxed (Z :. vocabLength :. embeddingDimensions) $ concat [mkRandomEmbedding (mkStdGen v) | v <- [0,1..vocabLength-1]]
    where
      mkRandomEmbedding :: StdGen -> [Float]
      mkRandomEmbedding = take embeddingDimensions . unfoldr (Just . uniformR (-3,3))
      vocabLength = length vocab

-- | Generate a set of embeddings as JSON. So that we can serialize our set, for tracking purposes.
example_2_7_3 :: NVec2F -> BSL.ByteString
example_2_7_3 = embeddingsToJSON

-- | Get the first 32 (8*4) tokens.
-- when given the-verdict.txt, gpt2-vocab.json, gpt2-merges.txt and 3d6-token_embeddings.json, produces the tensor in the middle of page 46.
example_2_8_1 :: [Char] -> InsOrdHashMap Pair Id -> Extensions -> NVec2I
example_2_8_1 text mergesTXT extensions = NVec2I $ fromListUnboxed (Z :. 8 :. 4) $ take (8*4) $ encodeExtensions extensions $ BPER.encode initSeqGPT2 mergesTXT gpt2pattern mempty $ BSU.fromString text

-- | Perform token embedding.
-- for each sequence in the inputs, look up the tokens in the embeddings, and replace the token with the embedding.
example_2_8_2 :: NVec2F -> NVec2I -> NVec3F
example_2_8_2 (NVec2F rawEmbeddings) (NVec2I tokenSeqs) = NVec3F $ fromListUnboxed (Z :. seqCount :. seqLength :. foundEmbeddingsDimensions) $ concat [(\(NVec2F a) -> DAR.toList a) $ embedSeq v | v <- sequences]
  where
    (Z :. seqCount :. seqLength) = extent tokenSeqs
    sequences :: [Seq]
    sequences = chunksOf seqLength $ DAR.toList tokenSeqs
    -- look up all of the items in Seq, generating embeddings for each of them. replaces [Int] with [[Float]]
    embedSeq :: Seq -> NVec2F
    embedSeq seq = NVec2F $ fromListUnboxed (Z :. length seq :. foundEmbeddingsDimensions) $ concat [DAR.toList $ slice rawEmbeddings (Any :. (v::Int) :. All) | v <- seq]
    (Z :. _ :. foundEmbeddingsDimensions) = extent rawEmbeddings

-- | Generate a set of positional embeddings.
example_2_8_3 :: Int -> Int -> NVec2F
example_2_8_3 dimensions positions = NVec2F $ fromListUnboxed (Z :. positions :. dimensions) $ take (positions * dimensions) [0.0 :: Float,1.0..]

-- | Add a set of positional embeddings to a set of token embeddings.
example_2_8_4 ::  NVec3F -> NVec2F -> NVec3F
example_2_8_4 (NVec3F tokenEmbeddings) (NVec2F positionalEmbeddings) =
  do
    let
      -- FIXME: use computeP in main().
      res1 :: DAR.Array U DIM3 Float
      res1 = computeS $ extendedPositionalEmbeddings +^ tokenEmbeddings
      in
      NVec3F res1
      where
        extendedPositionalEmbeddings = extend (Z :. (tokenCount :: Int) :. All :. All) positionalEmbeddings
        (Z :. tokenCount :. _ :. _) = extent tokenEmbeddings

-- | Read from a JSON file and display a set of token embeddings.
-- When given 3d6-token_embeddings-3_3_1.json and 6_token-vocab.json , produces the embeddings for the input sentence, on page 57.
-- Note: code identical to example 2_7_1.
example_3_3_1 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> NVec2F
example_3_3_1 = example_2_7_1

-- A one dimensional vector of Floats.
newtype NVec1F = NVec1F (DAR.Array U DIM1 Float)
  deriving Show

-- | Read a set of token embeddings from a JSON file, and calculate a set of attention results of the second token, vs the rest of the tokens.
-- When given 3d6-token_embeddings-3_3_1.json and 6_token-vocab.json , produces the attention values on page 58.
example_3_3_2 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> NVec1F
example_3_3_2 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot products of all tokens against the second token.
  | otherwise = findDots 1
  where
    -- | For a set of token embeddings, find the dot product of a given token when compared to every other token in the set.
    findDots :: Int -> NVec1F
    findDots itemNo
      | foundEmbeddingsCount < itemNo = error $ "Too few items.\n"
                                             <> "comparison token index: " <> show itemNo <> "\n"
                                             <> "found tokens: " <> show foundEmbeddingsCount <> "\n"
      | otherwise = NVec1F $ sumS $ rawTokenEmbeddings *^ extend (Z :. foundEmbeddingsCount :. All) target
      where
        target = slice rawTokenEmbeddings (Any :. (itemNo :: Int) :. All)
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | Read a set of token embeddings from a JSON file, and calculate a set of attention results of the second token, vs the rest of the tokens.
-- When given 3d6-token_embeddings-3_3_1.json and 6_token-vocab.json , produces the attention values on page 59.
example_3_3_3 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> NVec1F
example_3_3_3 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | attention against the second token.
  | otherwise = findDotAttn 1
  where
    -- | For a set of token embeddings, find the dot product of a given token when compared to every other token in the set. Normalize the output.
    findDotAttn :: Int -> NVec1F
    findDotAttn itemNo
      | foundEmbeddingsCount < itemNo = error $ "Too few items.\n"
                                             <> "comparison token index: " <> show itemNo <> "\n"
                                             <> "found tokens: " <> show foundEmbeddingsCount <> "\n"
      | otherwise = normVec $ NVec1F $ sumS $ rawTokenEmbeddings *^ extend (Z :. foundEmbeddingsCount :. All) target
      where
        normVec :: NVec1F -> NVec1F
        normVec (NVec1F inVec) = NVec1F $ computeS $ inVec /^ extend (Z :. foundItems) (sumS inVec)
          where
            (Z :. foundItems) = extent inVec
        target = slice rawTokenEmbeddings (Any :. (itemNo :: Int) :. All)
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | For a set of token embeddings, find the dot product of a given token when compared to every other token in the set. Normalize the output using softmax.
findAttn :: NVec2F -> Int -> NVec1F
findAttn (NVec2F rawTokenEmbeddings) itemNo
  | foundEmbeddingsCount < itemNo = error $ "Too few items.\n"
                                         <> "comparison token index: " <> show itemNo <> "\n"
                                         <> "found tokens: " <> show foundEmbeddingsCount <> "\n"
  | otherwise = NVec1F $ computeS $ softMax1F $ sumS $ rawTokenEmbeddings *^ extend (Z :. foundEmbeddingsCount :. All) target
  where
    target = slice rawTokenEmbeddings (Any :. (itemNo :: Int) :. All)
    (Z :. foundEmbeddingsCount :. _) = extent rawTokenEmbeddings

-- | Read a set of token embeddings from a JSON file, and calculate a set of attention results of the second token, vs the rest of the tokens.
-- When given 3d6-token_embeddings-3_3_1.json and 6_token-vocab.json , produces the attention values on page 60.
example_3_3_4 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> NVec1F
example_3_3_4 (HyperParams embeddingDimensions _) jsonDictionary tokenEmbeddings@(NVec2F rawTokenEmbeddings)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | softmax attention against the second token.
  | otherwise = findAttn tokenEmbeddings 1
  where
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- When given 3d6-token_embeddings-3_3_1.json and 6_token-vocab.json , produces the context vector at the bottom of page 60.
example_3_3_5 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> NVec1F
example_3_3_5 (HyperParams embeddingDimensions _) jsonDictionary tokenEmbeddings@(NVec2F rawTokenEmbeddings)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | softmax attention against the second token.
  | otherwise = NVec1F $ sumS $ transpose $ rawTokenEmbeddings *^ extend (Z :. All :. (foundEmbeddingsDimensions :: Int)) rawFoundAttention
  where
    (NVec1F rawFoundAttention) = findAttn tokenEmbeddings 1
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | Read a set of token embeddings from a JSON file, and calculate a set of un-normalized attention results.
-- When given 3d6-token_embeddings-3_3_1.json and 6_token-vocab.json , produces the set of attention values on page 62.
example_3_3_6 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> NVec2F
example_3_3_6 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | softmax attention of each item against each item, including itsself.
  | otherwise = findMyAttns
  where
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings
    -- | For a set of token embeddings, find the dot product of each token when compared to every other token in the set, and itsself. Normalize the outputs using softmax.
    findMyAttns :: NVec2F
    findMyAttns = NVec2F $ sumS $ leftSide *^ rightSide
      where
        leftSide = extend (Z :. foundEmbeddingsCount :. All :. All) rawTokenEmbeddings
        rightSide = extend (Z :. All :. foundEmbeddingsCount :. All) rawTokenEmbeddings

-- | Read a set of token embeddings from a JSON file, and calculate a set of attention results of the second token, vs the rest of the tokens.
-- When given 3d6-token_embeddings-3_3_1.json and 6_token-vocab.json , produces the attention values on page 63.
example_3_3_7 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> NVec2F
example_3_3_7 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | softmax attentions.
  | otherwise = NVec2F $ computeS $ findAttns rawTokenEmbeddings
  where
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | Read a set of token embeddings from a JSON file, and calculate a set of attention results of the second token, vs the rest of the tokens.
-- When given 3d6-token_embeddings-3_3_1.json and 6_token-vocab.json , produces the context vectors on page 63.
example_3_3_8 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> NVec2F
example_3_3_8 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | softmax attention against the second token.
  | otherwise = NVec2F $ sumS $ transpose $ extend (Z :. foundEmbeddingsCount :. All :. All) rawTokenEmbeddings *^ extend (Z :. All :. All:. (foundEmbeddingsDimensions :: Int)) rawFoundAttention
  where
    rawFoundAttention = findAttns rawTokenEmbeddings
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | For a set of token embeddings, find the dot product of each token when compared to every other token in the set, and itsself. Normalize the outputs using softmax.
findAttns :: DAR.Array U DIM2 Float -> DAR.Array D DIM2 Float
findAttns rawTokenEmbeddings = softMax $ sumS $ leftSide *^ rightSide
  where
    leftSide = extend (Z :. foundEmbeddingsCount :. All :. All) rawTokenEmbeddings
    rightSide = extend (Z :. All :. foundEmbeddingsCount :. All) rawTokenEmbeddings
    (Z :. foundEmbeddingsCount :. _) = extent rawTokenEmbeddings

-- | A set of Q, K, and V weights.
newtype QKV = QKV (InsOrdHashMap Char NVec2F)
  deriving Show

-- | Attention weights
newtype AttentionWeights = AttentionWeights (InsOrdHashMap Int QKV)
  deriving Show

-- | Parse a file of Attention Weights.
instance FromJSON AttentionWeights where
  parseJSON = withObject "AttentionWeights" (pure . findAttentionWeights . DAKM.toList)
    where
      findAttentionWeights :: [(Key, Value)] -> AttentionWeights
      findAttentionWeights maybeTokenMaps = AttentionWeights weightMaps
        where
          weightMaps :: InsOrdHashMap Int QKV
          weightMaps = weightMaps' maybeTokenMaps empty
          weightMaps' :: [(Key, Value)] -> InsOrdHashMap Int QKV -> InsOrdHashMap Int QKV
          weightMaps' [] myTokenMaps = myTokenMaps
          weightMaps' [(k,v)] myTokenMaps = insert (read $ AK.toString k) (qkvFromValue v) myTokenMaps
          weightMaps' ((k,v):xs) myTokenMaps = insert (read $ AK.toString k) (qkvFromValue v) (weightMaps' xs myTokenMaps)
          qkvFromValue :: Value -> QKV
          qkvFromValue (Object o) = qkvFromObject $ DAKM.toList o
          qkvFromValue v = error $ "missed!\n" <> show v <> "\n"
          qkvFromObject :: [(Key,Value)] -> QKV
          qkvFromObject entries@[_,_,_] = makeQKV $ foldl findQKV (Nothing, Nothing, Nothing) $ cleanEntries <$> entries
          qkvFromObject entries = error $ "wrong number of items\n" <> show (length entries) <> "\n"
          cleanEntries (key, value) = (keyToChar key, cleanValue value)
          keyToChar key = case AK.toString key of
                            "Q" -> 'Q'
                            "K" -> 'K'
                            "V" -> 'V'
                            a   -> error $ "Missed!\n" <> show a <> "\n"
          cleanValue (Object a) = buildSubLists $ findSubLists <$> DAKM.toList a
          cleanValue v = error $ "Missed!\n" <> "V: " <> show v <> "\n"
          buildSubLists :: [(Int, [Float])] -> NVec2F
          buildSubLists sublists@((_, firstFloats):_) = NVec2F $ fromListUnboxed (Z :. length sublists :. length firstFloats) $ concat $ [fromMaybe (error $ "sequence fail: " <> show i <> "\n") $ lookup i $ DHSI.fromList sublists | i <- [0,1.. length sublists-1]]
          buildSubLists [] = error "no sub-lists to build.\n"
          findSubLists :: (Key, Value) -> (Int, [Float])
          findSubLists (key, value) = (read (AK.toString key) :: Int, floatsFromValue value)
          floatsFromValue :: Value -> [Float]
          floatsFromValue (Array vs) = (\(Number a) -> toRealFloat a) <$> DV.toList vs
          floatsFromValue a = error $ "failed to parse " <> show a <> " as a Number.\n"
          makeQKV :: (Maybe (Char, NVec2F), Maybe (Char, NVec2F), Maybe (Char, NVec2F)) -> QKV
          makeQKV (Just q,Just k,Just v) = QKV $ DHSI.fromList [q,k,v]
          makeQKV (a, b, c) = error $ "Missed!\n"
                                   <> "A: " <> show a <> "\n"
                                   <> "B: " <> show b <> "\n"
                                   <> "C: " <> show c <> "\n"
          findQKV :: (Maybe (Char, NVec2F), Maybe (Char, NVec2F), Maybe (Char, NVec2F)) -> (Char, NVec2F) -> (Maybe (Char, NVec2F), Maybe (Char, NVec2F), Maybe (Char, NVec2F))
          findQKV (Nothing, k, v) ('Q', q) = (Just ('Q', q), k, v)
          findQKV (q, Nothing, v) ('K', k) = (q, Just ('K', k), v)
          findQKV (q, k, Nothing) ('V', v) = (q, k, Just ('V', v))
          findQKV (q,k,v) (c, f) = error $ "Missed!\n"
                                        <> "Q: " <> show q <> "\n"
                                        <> "K: " <> show k <> "\n"
                                        <> "V: " <> show v <> "\n"
                                        <> "C: " <> show c <> "\n"
                                        <> "F: " <> show f <> "\n"

-- | Read a set of attention weights from a JSON file, and calculate a set of attention results of the second token, vs the rest of the tokens.
-- When given 3d6-token_embeddings-3_3_1.json, 6_token-vocab.json, and 3d6-weights.json, produces the tensor on page 66.
example_3_4_1 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> AttentionWeights -> NVec1F
example_3_4_1 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | softmax attention against the second token.
  | otherwise = NVec1F $ sumS $ leftSide *^ rightSide
  where
    leftSide = transpose query
    rightSide = extend (Z :. queryEmbeddingsDimensions :. All) input2
    input2 = slice rawTokenEmbeddings (Any :. (1 :: Int) :. All)
    (Z :. _ :. queryEmbeddingsDimensions) = extent query
    (NVec2F query) = fromMaybe (error "no Q?") $ lookup 'Q' weight
    (QKV weight) = fromMaybe (error "no weights?") $ lookup 0 weights
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | Read a set of attention weights from a JSON file, and calculate a set of keys and values for all of the tokens.
-- When given 3d6-token_embeddings-3_3_1.json, 6_token-vocab.json, and 3d6-weights.json, ultimately producing the shapes on page 67.
example_3_4_2 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> AttentionWeights -> NVec2F
example_3_4_2 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | softmax attention against the second token.
  | otherwise = NVec2F $ sumS $ leftSide *^ rightSide
  where
    leftSide = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose query
    rightSide = extend (Z :. All :. queryEmbeddingsDimensions :. All) rawTokenEmbeddings
    (Z :. _ :. queryEmbeddingsDimensions) = extent query
    (NVec2F query) = fromMaybe (error "no Q?") $ lookup 'Q' weight
    (QKV weight) = fromMaybe (error "no weights?") $ lookup 0 weights
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | Read a set of attention weights from a JSON file, and calculate a set of keys and values for all of the tokens.
-- When given 3d6-token_embeddings-3_3_1.json, 6_token-vocab.json, and 3d6-weights.json, ultimately producing an NVec2F with the second shape on page 67.
example_3_4_3 :: HyperParams -> InsOrdHashMap Id BSS.ByteString-> NVec2F -> AttentionWeights -> NVec2F
example_3_4_3 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | softmax attention against the second token.
  | otherwise = NVec2F $ sumS $ leftSide *^ rightSide
  where
    leftSide = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose key
    rightSide = extend (Z :. All :. keyEmbeddingsDimensions :. All) rawTokenEmbeddings
    (Z :. _ :. keyEmbeddingsDimensions) = extent key
    (NVec2F key) = fromMaybe (error "no K?") $ lookup 'K' weight
    (QKV weight) = fromMaybe (error "no weights?") $ lookup 0 weights
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- A zero dimensional vector of Floats, AKA, a single Float.
newtype NVec0F = NVec0F (DAR.Array U Z Float)
  deriving Show

-- | Read a set of attention weights from a JSON file, and calculate a set of keys and values for all of the tokens.
-- When given 3d6-token_embeddings-3_3_1.json, 6_token-vocab.json, and 3d6-weights.json, ultimately producing an NVec0F with the value at the top of page 68.
example_3_4_4 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> AttentionWeights -> NVec0F
example_3_4_4 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | softmax attention against the second token.
  | otherwise = res
  where
    res = NVec0F $ sumS $ query2 *^ key2
    key2 = slice keyRes (Z :. (1::Int) :. All)
    query2 = slice queryRes (Z :. (1::Int) :. All)
    keyRes = sumS $ leftSideKey *^ rightSide
    queryRes = sumS $ leftSideQuery *^ rightSide
    leftSideKey = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose key
    leftSideQuery = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose query
    -- FIXME: this constrains query size == key size.
    rightSide = extend (Z :. All :. keyEmbeddingsDimensions :. All) rawTokenEmbeddings
    (NVec2F query) = fromMaybe (error "no Q?") $ lookup 'Q' weight
    (Z :. _ :. keyEmbeddingsDimensions) = extent key
    (NVec2F key) = fromMaybe (error "no K?") $ lookup 'K' weight
    (QKV weight) = fromMaybe (error "no weights?") $ lookup 0 weights
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | Read a set of attention weights from a JSON file, and calculate a set of keys and values for all of the tokens.
-- When given 3d6-token_embeddings-3_3_1.json, 6_token-vocab.json, and 3d6-weights.json, ultimately producing an NVec1F with the set of six values near the top of page 68.
example_3_4_5 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> AttentionWeights -> NVec1F
example_3_4_5 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | softmax attention against the second token.
  | otherwise = res
  where
    res = NVec1F $ sumS $ keyRes *^ query2Up
    query2Up = extend (Z :. foundEmbeddingsCount :. All) query2
    query2 = slice queryRes (Z :. (1::Int) :. All)
    keyRes = sumS $ leftSideKey *^ rightSide
    queryRes = sumS $ leftSideQuery *^ rightSide
    leftSideKey = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose key
    leftSideQuery = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose query
    -- FIXME: this constrains query size == key size.
    rightSide = extend (Z :. All :. keyEmbeddingsDimensions :. All) rawTokenEmbeddings
    (NVec2F query) = fromMaybe (error "no Q?") $ lookup 'Q' weight
    (Z :. _ :. keyEmbeddingsDimensions) = extent key
    (NVec2F key) = fromMaybe (error "no K?") $ lookup 'K' weight
    (QKV weight) = fromMaybe (error "no weights?") $ lookup 0 weights
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | Read a set of attention weights from a JSON file, and calculate a set of attention weights for all of the given tokens.
-- When given 3d6-token_embeddings-3_3_1.json, 6_token-vocab.json, and 3d6-weights.json, ultimately producing an NVec1F with the set of six values near the top of page 69.
example_3_4_6 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> AttentionWeights -> NVec1F
example_3_4_6 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the modified dot product | softmax attention against the second token.
  | otherwise = res
  where
    -- Which result to calculate. The book calculates the second position,
    index :: Int
    index = 1
    res = NVec1F $ computeS $ softMax1F $ computeS $ slice (sumS $ map (/(sqrt $ fromIntegral keyEmbeddingsDimensions)) $ moreKeyRes *^ moreQueryRes) (Z :. index :. All)
    moreQueryRes = extend (Z :. All :. foundEmbeddingsCount :. All) queryRes
    moreKeyRes = extend (Z :. foundEmbeddingsCount :. All :. All) keyRes
    queryRes = sumS $ leftSideQuery *^ rightSide
    keyRes = sumS $ leftSideKey *^ rightSide
    leftSideQuery = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose query
    leftSideKey = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose key
    -- FIXME: this constrains query size == key size.
    rightSide = extend (Z :. All :. keyEmbeddingsDimensions :. All) rawTokenEmbeddings
    (NVec2F query) = fromMaybe (error "no Q?") $ lookup 'Q' weight
    (Z :. _ :. keyEmbeddingsDimensions) = extent key
    (NVec2F key) = fromMaybe (error "no K?") $ lookup 'K' weight
    (QKV weight) = fromMaybe (error "no weights?") $ lookup 0 weights
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | Read a set of attention weights from a JSON file, and calculate a context vector for the second token.
-- When given 3d6-token_embeddings-3_3_1.json, 6_token-vocab.json, and 3d6-weights.json, ultimately producing an NVec1F with the set of two values near the top of page 70.
example_3_4_7 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> AttentionWeights -> NVec1F
example_3_4_7 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the modified dot product | softmax attention against the second token.
  | otherwise = res
  where
    -- which result to calculate. the book calculates the second position.
    index :: Int
    index = 1
    res = NVec1F $ sumS $ selectedKeyQuery *^ valuesRes
    selectedKeyQuery = extend (Z :. foundEmbeddingsCount :. All) $ slice keyQuery (Z :. index :. All)
    keyQuery = softMax $ sumS $ map (/(sqrt $ fromIntegral keyEmbeddingsDimensions)) $ moreKeyRes *^ moreQueryRes
    moreQueryRes = extend (Z :. All :. foundEmbeddingsCount :. All) queryRes
    moreKeyRes = extend (Z :. foundEmbeddingsCount :. All :. All) keyRes
    queryRes = sumS $ leftSideQuery *^ rightSide
    keyRes = sumS $ leftSideKey *^ rightSide
    valuesRes = transpose $ sumS $ leftSideValues *^ rightSide
    leftSideQuery = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose query
    leftSideKey = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose key
    leftSideValues = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose values
    -- FIXME: this constrains query size == key size == value size.
    rightSide = extend (Z :. All :. keyEmbeddingsDimensions :. All) rawTokenEmbeddings
    (NVec2F query) = fromMaybe (error "no Q?") $ lookup 'Q' weight
    (Z :. _ :. keyEmbeddingsDimensions) = extent key
    (NVec2F key) = fromMaybe (error "no K?") $ lookup 'K' weight
    (NVec2F values) = fromMaybe (error "no V?") $ lookup 'V' weight
    (QKV weight) = fromMaybe (error "no weights?") $ lookup 0 weights
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | Read a set of attention weights from a JSON file, and calculate a context vector for all six tokens.
-- When given 3d6-token_embeddings-3_3_1.json, 6_token-vocab.json, and 3d6-weights.json, ultimately producing the set of 12 values (divided into 6x2) near the middle of page 71.
example_3_4_8 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> AttentionWeights -> NVec2F
example_3_4_8 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the context vectors for all tokens
  | otherwise = res
  where
    res = NVec2F $ sumS $ moreKeyQuery *^ moreValuesRes
    moreValuesRes = extend (Z :. foundEmbeddingsCount :. All :. All) valuesRes
    moreKeyQuery = extend (Z :. All :. foundEmbeddingsCount :. All) keyQuery
    keyQuery = softMax $ sumS $ map (/(sqrt $ fromIntegral keyEmbeddingsDimensions)) $ moreKeyRes *^ moreQueryRes
    moreQueryRes = extend (Z :. All :. foundEmbeddingsCount :. All) queryRes
    moreKeyRes = extend (Z :. foundEmbeddingsCount :. All :. All) keyRes
    queryRes = sumS $ leftSideQuery *^ rightSide
    keyRes = sumS $ leftSideKey *^ rightSide
    valuesRes = transpose $ sumS $ leftSideValues *^ rightSide
    leftSideQuery = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose query
    leftSideKey = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose key
    leftSideValues = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose values
    rightSide = extend (Z :. All :. keyEmbeddingsDimensions :. All) rawTokenEmbeddings
    (NVec2F query) = fromMaybe (error "no Q?") $ lookup 'Q' weight
    -- FIXME: this constrains query size == key size.
    (Z :. _ :. keyEmbeddingsDimensions) = extent key
    (NVec2F key) = fromMaybe (error "no K?") $ lookup 'K' weight
    (NVec2F values) = fromMaybe (error "no V?") $ lookup 'V' weight
    (QKV weight) = fromMaybe (error "no weights?") $ lookup 0 weights
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- We skip example 3_4_9, as that is just proving we can read weights.

-- | Generate a random set of attention weights that has a uniform distribution, similar to the 'nn.linear' python module.
example_3_4_10 :: HyperParams -> AttentionWeights
example_3_4_10 (HyperParams embeddingDims attentionWeightDims) = res
  where
    res = randomAttentionWeight embeddingDims attentionWeightDims 789

-- | Generate a single random set of attention weights that has a uniform distribution, similar to the 'nn.linear' python module.
randomAttentionWeight :: Int -> Int -> Int -> AttentionWeights
randomAttentionWeight inputEmbeddings outputEmbeddings mySeed = AttentionWeights $ DHSI.fromList [(0,makeRandomQKV mySeed)]
  where
    makeRandomQKV :: Int -> QKV
    makeRandomQKV seed = QKV $ insert 'V' (makeRandomEmbedding (mkStdGen seed))
                             $ insert 'K' (makeRandomEmbedding (mkStdGen $ seed+1))
                             $ insert 'Q' (makeRandomEmbedding (mkStdGen $ seed+2))
                               empty
    makeRandomEmbedding :: StdGen -> NVec2F
    makeRandomEmbedding = NVec2F . fromListUnboxed (Z :. inputEmbeddings :. outputEmbeddings) . take (inputEmbeddings * outputEmbeddings) . unfoldr (Just . uniformR (-1,1))

-- | Calculate a context vector for the second token.
-- When given 3d6-token_embeddings-3_3_1.json, 6_token-vocab.json, and 3d6-weights-3_4_10.json, ultimately producing the set of 12 values (divided into 6x2) near the middle of page 73.
example_3_4_11 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> AttentionWeights -> NVec2F
example_3_4_11 = example_3_4_8

-- | Read a set of attention weights from a JSON file, and calculate a query-key tensor using the query and key from the attention weights.
-- When given 3d6-token_embeddings-3_3_1.json, 6_token-vocab.json, and 3d6-weights-3_4_10.json, ultimately producing the set of 36 values (divided into 6x6) toward the bottom of page 75.
example_3_5_1 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> AttentionWeights -> NVec2F
example_3_5_1 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the modified dot product | softmax attention against the second token.
  | otherwise = res
  where
    res = NVec2F $ computeS $ softMax $ sumS $ map (/(sqrt $ fromIntegral keyEmbeddingsDimensions)) $ moreKeyRes *^ moreQueryRes
    moreQueryRes = extend (Z :. All :. foundEmbeddingsCount :. All) queryRes
    moreKeyRes = extend (Z :. foundEmbeddingsCount :. All :. All) keyRes
    queryRes = sumS $ leftSideQuery *^ rightSide
    keyRes = sumS $ leftSideKey *^ rightSide
    leftSideQuery = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose query
    leftSideKey = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose key
    rightSide = extend (Z :. All :. keyEmbeddingsDimensions :. All) rawTokenEmbeddings
    (NVec2F query) = fromMaybe (error "no Q?") $ lookup 'Q' weight
    -- FIXME: this constrains query size == key size.
    (Z :. _ :. keyEmbeddingsDimensions) = extent key
    (NVec2F key) = fromMaybe (error "no K?") $ lookup 'K' weight
    (QKV weight) = fromMaybe (error "no weights?") $ lookup 0 weights
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | Generate an attention weight matrix where the attention weights above the diagonal are zero, and everything else is 1.
-- When given 6_token-vocab.json, produces the tensor at the top of page 76.
example_3_5_2 :: NVec2F -> NVec2F
example_3_5_2 (NVec2F rawTokenEmbeddings) = NVec2F $ fromListUnboxed (Z :. embeddingsCount :. embeddingsCount) $ concat $ [[ if y > x then 0 else 1 | y <- [1,2..embeddingsCount]] | x <- [1,2..embeddingsCount]]
  where
    (Z :. embeddingsCount :. _) = extent rawTokenEmbeddings

-- | Read a set of attention weights from a JSON file, and calculate attention results, without future knowledge making it into the attention calculation.
-- When given 3d6-token_embeddings-3_3_1.json, 6_token-vocab.json, and 3d6-weights-3_4_10.json, ultimately producing the set of 36 values (divided into 6x6) in the middle of page 76.
example_3_5_3 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> AttentionWeights -> NVec2F
example_3_5_3 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the modified dot product | softmax attention against the second token.
  | otherwise = res
  where
    res = NVec2F $ computeS $ keyQuery *^ futureDrop
    keyQuery = softMax $ sumS $ map (/(sqrt $ fromIntegral keyEmbeddingsDimensions)) $ moreKeyRes *^ moreQueryRes
    futureDrop = futureDropOf foundEmbeddingsCount
    moreQueryRes = extend (Z :. All :. foundEmbeddingsCount :. All) queryRes
    moreKeyRes = extend (Z :. foundEmbeddingsCount :. All :. All) keyRes
    queryRes = sumS $ leftSideQuery *^ rightSide
    keyRes = sumS $ leftSideKey *^ rightSide
    leftSideQuery = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose query
    leftSideKey = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose key
    rightSide = transpose $ extend (Z :. All :. All :. keyEmbeddingsDimensions) rawTokenEmbeddings
    (NVec2F query) = fromMaybe (error "no Q?") $ lookup 'Q' weight
    -- FIXME: this constrains query size == key size.
    (Z :. _ :. keyEmbeddingsDimensions) = extent key
    (NVec2F key) = fromMaybe (error "no K?") $ lookup 'K' weight
    (QKV weight) = fromMaybe (error "no weights?") $ lookup 0 weights
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | Read a set of attention weights from a JSON file, and calculate attention results, without future knowledge making it into the attention calculation. normalizes when done.
-- When given 3d6-token_embeddings-3_3_1.json, 6_token-vocab.json, and 3d6-weights-3_4_10.json, ultimately producing the set of 36 values (divided into 6x6) near the bottom of page 76.
example_3_5_4 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> AttentionWeights -> NVec2F
example_3_5_4 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the modified dot product | softmax attention against the second token.
  | otherwise = res
  where
    res = NVec2F $ computeS $ simpleNorm $ keyQuery *^ futureDrop
    keyQuery = softMax $ sumS $ map (/(sqrt $ fromIntegral keyEmbeddingsDimensions)) $ moreKeyRes *^ moreQueryRes
    futureDrop = futureDropOf foundEmbeddingsCount
    moreQueryRes = extend (Z :. All :. foundEmbeddingsCount :. All) queryRes
    moreKeyRes = extend (Z :. foundEmbeddingsCount :. All :. All) keyRes
    queryRes = sumS $ leftSideQuery *^ rightSide
    keyRes = sumS $ leftSideKey *^ rightSide
    leftSideQuery = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose query
    leftSideKey = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose key
    rightSide = transpose $ extend (Z :. All :. All :. keyEmbeddingsDimensions) rawTokenEmbeddings
    (NVec2F query) = fromMaybe (error "no Q?") $ lookup 'Q' weight
    -- FIXME: this constrains query size == key size.
    (Z :. _ :. keyEmbeddingsDimensions) = extent key
    (NVec2F key) = fromMaybe (error "no K?") $ lookup 'K' weight
    (QKV weight) = fromMaybe (error "no weights?") $ lookup 0 weights
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | Generate a dropout mask where the values are either 0 or 2, randomly..
-- When given 6_token-vocab.json and a random integer, produces a random dropout map.
example_3_5_5 :: NVec2F -> Int -> NVec2F
example_3_5_5 (NVec2F rawTokenEmbeddings) mySeed = res
  where
    res = randomDropoutMap embeddingsCount mySeed
    (Z :. embeddingsCount :. _) = extent rawTokenEmbeddings

randomDropoutMap :: Int -> Int -> NVec2F
randomDropoutMap embeddingsCount mySeed = NVec2F $ fromListUnboxed (Z :. embeddingsCount :. embeddingsCount) $ zeroToTwo <$> take (embeddingsCount*embeddingsCount) (yesNo $ mkStdGen mySeed)
  where
    zeroToTwo :: Bool -> Float
    zeroToTwo False = 0
    zeroToTwo True = 2
    yesNo :: StdGen -> [Bool]
    yesNo = unfoldr (Just . random)

-- | Read a dropout map from a JSON file, and calculate an attention weight matrix scaling by the dropout map.
-- When given 3d6-token_embeddings-3_3_1.json, 6_token-vocab.json, 3d6-weights-3_4_10.json, and 3d6-dropout_mask.json, ultimately producing the set of 36 values (divided into 6x6) near the top of page 80.
example_3_5_6 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> AttentionWeights -> NVec2F -> NVec2F
example_3_5_6 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights) (NVec2F rawDropoutMap)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the modified dot product | softmax attention against the second token.
  | otherwise = res
  where
    res = NVec2F $ computeS $ rawDropoutMap *^ droppedKeyQuery
    droppedKeyQuery = simpleNorm $ keyQuery *^ futureDrop
    keyQuery = softMax $ sumS $ map (/(sqrt $ fromIntegral keyEmbeddingsDimensions)) $ moreKeyRes *^ moreQueryRes
    futureDrop = futureDropOf foundEmbeddingsCount
    moreQueryRes = extend (Z :. All :. foundEmbeddingsCount :. All) queryRes
    moreKeyRes = extend (Z :. foundEmbeddingsCount :. All :. All) keyRes
    queryRes = sumS $ leftSideQuery *^ rightSide
    keyRes = sumS $ leftSideKey *^ rightSide
    leftSideQuery = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose query
    leftSideKey = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose key
    rightSide = transpose $ extend (Z :. All :. All :. keyEmbeddingsDimensions) rawTokenEmbeddings
    (NVec2F query) = fromMaybe (error "no Q?") $ lookup 'Q' weight
    -- FIXME: this constrains query size == key size.
    (Z :. _ :. keyEmbeddingsDimensions) = extent key
    (NVec2F key) = fromMaybe (error "no K?") $ lookup 'K' weight
    (QKV weight) = fromMaybe (error "no weights?") $ lookup 0 weights
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | Produce an array with two input texts, each containing 6 tokens, with 3 embeddings per token.
-- When given 3d6-token_embeddings-3_3_1.json, produces a 3 dimentional vector with the shape on the bottom of page 80.
example_3_5_7 :: NVec2F -> NVec3F
example_3_5_7 (NVec2F rawTokenEmbeddings) = NVec3F $ computeS $ extend (Z :. (2::Int) :. All :. All) rawTokenEmbeddings

-- | Read a dropout map from a JSON file, and calculate a set of context vectors scaling by the dropout map during the Q*K step.
-- When given 3d6-token_embeddings-3_3_1.json, 6_token-vocab.json, 3d6-weights-3_4_10.json, and 3d6-dropout_masks.json, ultimately producing the set of 36 values (divided into 6x6) near the top of page 80.
example_3_5_8 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> AttentionWeights -> NVec2F -> NVec2F
example_3_5_8 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights) (NVec2F rawDropoutMap)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- find the dropped out key*query result.
  | otherwise = res
  where
    res = NVec2F $ sumS $ moreKeyQuery *^ transpose moreValuesRes
    moreValuesRes = extend (Z :. foundEmbeddingsCount :. All :. All) valuesRes
    moreKeyQuery = extend (Z :. All :. foundEmbeddingsCount :. All) $ rawDropoutMap *^ droppedKeyQuery
    droppedKeyQuery = simpleNorm $ keyQuery *^ futureDrop
    keyQuery = softMax $ sumS $ map (/(sqrt $ fromIntegral keyEmbeddingsDimensions)) $ moreKeyRes *^ moreQueryRes
    futureDrop = futureDropOf foundEmbeddingsCount
    moreQueryRes = extend (Z :. All :. foundEmbeddingsCount :. All) queryRes
    moreKeyRes = extend (Z :. foundEmbeddingsCount :. All :. All) keyRes
    queryRes = sumS $ leftSideQuery *^ rightSide
    keyRes = sumS $ leftSideKey *^ rightSide
    valuesRes = sumS $ leftSideValues *^ rightSide
    leftSideQuery = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose query
    leftSideKey = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose key
    leftSideValues = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose values
    rightSide = transpose $ extend (Z :. All :. All :. keyEmbeddingsDimensions) rawTokenEmbeddings
    (NVec2F query) = fromMaybe (error "no Q?") $ lookup 'Q' weight
    -- FIXME: this constrains query size == key size.
    (Z :. _ :. keyEmbeddingsDimensions) = extent key
    (NVec2F key) = fromMaybe (error "no K?") $ lookup 'K' weight
    (NVec2F values) = fromMaybe (error "no V?") $ lookup 'V' weight
    (QKV weight) = fromMaybe (error "no weights?") $ lookup 0 weights
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | calculate a set of context vectors using our "random" functions instead of reading from files.
-- When given 3d6-token_embeddings-3_3_1.json, 6_token-vocab.json, returns a result in the appropriate shape (6*2 values).
example_3_5_9 :: HyperParams -> InsOrdHashMap Id BSS.ByteString -> NVec2F -> Int -> NVec2F
example_3_5_9 (HyperParams embeddingDimensions attentionWeightDimensions) jsonDictionary (NVec2F rawTokenEmbeddings) seed
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- find the dropped out key*query result.
  | otherwise = res
  where
    res = NVec2F $ sumS $ moreKeyQuery *^ transpose moreValuesRes
    moreValuesRes = extend (Z :. foundEmbeddingsCount :. All :. All) valuesRes
    moreKeyQuery = extend (Z :. All :. foundEmbeddingsCount :. All) $ rawDropoutMap *^ droppedKeyQuery
    (NVec2F rawDropoutMap) = randomDropoutMap foundEmbeddingsCount seed
    droppedKeyQuery = simpleNorm $ keyQuery *^ futureDrop
    keyQuery = softMax $ sumS $ map (/(sqrt $ fromIntegral keyEmbeddingsDimensions)) $ moreKeyRes *^ moreQueryRes
    futureDrop = futureDropOf foundEmbeddingsCount
    moreQueryRes = extend (Z :. All :. foundEmbeddingsCount :. All) queryRes
    moreKeyRes = extend (Z :. foundEmbeddingsCount :. All :. All) keyRes
    queryRes = sumS $ leftSideQuery *^ rightSide
    keyRes = sumS $ leftSideKey *^ rightSide
    valuesRes = sumS $ leftSideValues *^ rightSide
    leftSideQuery = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose query
    leftSideKey = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose key
    leftSideValues = extend (Z :. foundEmbeddingsCount :. All :. All) $ transpose values
    rightSide = transpose $ extend (Z :. All :. All :. keyEmbeddingsDimensions) rawTokenEmbeddings
    (NVec2F query) = fromMaybe (error "no Q?") $ lookup 'Q' weight
    -- FIXME: this constrains query size == key size.
    (Z :. _ :. keyEmbeddingsDimensions) = extent key
    (NVec2F key) = fromMaybe (error "no K?") $ lookup 'K' weight
    (NVec2F values) = fromMaybe (error "no V?") $ lookup 'V' weight
    (QKV weight) = fromMaybe (error "no weights?") $ lookup 0 weights
    (AttentionWeights weights) = randomAttentionWeight embeddingDimensions attentionWeightDimensions seed
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | calculate two sets of context vectors using our "random" functions instead of reading from files.
-- When given 3d6-token_embeddings-3_3_1.json, 6_token-vocab.json, returns a result in the appropriate shape (2*6*2 values), as seen on the top of page 82.
example_3_5_10 :: Foldable t => HyperParams -> t a -> NVec2F -> Int -> NVec3F
example_3_5_10 (HyperParams embeddingDimensions attentionWeightDimensions) jsonDictionary (NVec2F rawTokenEmbeddings) seed
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- find the dropped out key*query results.
  | otherwise = res tokens qkvs
  where
    qkvs = findQKVs weights
    tokens = NVec3F $ computeS $ extend (Z :. (2::Int) :. All :. All) rawTokenEmbeddings
    res :: NVec3F -> [QKV] -> NVec3F
    res (NVec3F myTokens) myQKVs = NVec3F $ sumS $ moreKeysQueries *^ transpose moreValuesRes
      where
        moreKeysQueries = extend (Z :. All :. All :. foundEmbeddingsCount :. All) $ moreDropoutMaps *^ droppedKeysQueries
          where
            moreDropoutMaps = extend (Z :. foundTokenSets :. All :. All) myRawDropoutMap
            (NVec2F myRawDropoutMap) = randomDropoutMap foundEmbeddingsCount seed
            droppedKeysQueries = simpleNorm3F $ keysQueries *^ futuresDrop
              where
                futuresDrop = extend (Z :. foundTokenSets :. All :. All) $ futureDropOf foundEmbeddingsCount
                keysQueries = softMax3F $ sumS $ map (/(sqrt $ fromIntegral keyEmbeddingsDimensions)) $ moreKeysRes *^ moreQueriesRes
                  where
                    moreKeysRes = extend (Z :. All :. foundEmbeddingsCount :. All :. All) keysRes
                      where
                        keysRes = sumS $ leftSideKeys *^ rightSides
                    moreQueriesRes = extend (Z :. All :. All :. foundEmbeddingsCount :. All) queriesRes
                      where
                        queriesRes = sumS $ leftSideQueries *^ rightSides
        moreValuesRes = extend (Z :. All :. foundEmbeddingsCount :. All :. All) valuesRes
          where
            valuesRes = sumS $ leftSideValues *^ rightSides
        (Z :. foundTokenSets :. _ :. _) = extent myTokens
        leftSideQueries = extend (Z :. foundTokenSets :. foundEmbeddingsCount :. All :. All) $ transpose query
        leftSideKeys = extend (Z :. foundTokenSets :. foundEmbeddingsCount :. All :. All) $ transpose key
        leftSideValues = extend (Z :. foundTokenSets :. foundEmbeddingsCount :. All :. All) $ transpose values
        rightSides = transpose $ extend (Z :. All :. All :. All :. keyEmbeddingsDimensions) myTokens
        (NVec2F query) = fromMaybe (error "no Q?") $ lookup 'Q' weight
      -- FIXME: this constrains query size == key size.
        (Z :. _ :. keyEmbeddingsDimensions) = extent key
        (NVec2F key) = fromMaybe (error "no K?") $ lookup 'K' weight
        (NVec2F values) = fromMaybe (error "no V?") $ lookup 'V' weight
        (QKV weight) = head myQKVs
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings
    (AttentionWeights weights) = randomAttentionWeight embeddingDimensions attentionWeightDimensions seed

-- | A type for a set of dropout maps, as they come out of the JSON file.
newtype DropoutMaps = DropoutMaps (InsOrdHashMap BSS.ByteString [[Float]])
  deriving Show

-- | Our parser for a dropout map.
instance FromJSON DropoutMaps where
  parseJSON = withObject "DropoutMaps" (pure .findDropoutMaps .DAKM.toList)
    where
      findDropoutMaps :: [(Key, Value)] -> DropoutMaps
      findDropoutMaps maybeDropoutMaps = DropoutMaps dropoutmaps
        where
          dropoutmaps = dropoutmaps' maybeDropoutMaps empty
          dropoutmaps' :: [(Key, Value)] -> InsOrdHashMap BSS.ByteString [[Float]] -> InsOrdHashMap BSS.ByteString [[Float]]
          dropoutmaps' [] myDropoutMaps = myDropoutMaps
          dropoutmaps' [(k,v)] myDropoutMaps = insert (encodeUtf8 $ toText k) (dropoutMapFromValue v) myDropoutMaps
          dropoutmaps' ((k,v):xs) myDropoutMaps = insert (encodeUtf8 $ toText k) (dropoutMapFromValue v) (dropoutmaps' xs myDropoutMaps)
          dropoutMapFromValue :: Value -> [[Float]]
          dropoutMapFromValue (Array vs) = numbersFromValue <$> DV.toList vs
          dropoutMapFromValue a = error $ "failed to parse " <> show a <> " as an Array.\n"
          numbersFromValue :: Value -> [Float]
          numbersFromValue (Array vs) = (\(Number a) -> toRealFloat a) <$> DV.toList vs
          numbersFromValue a = error $ "failed to parse " <> show a <> " as an Array.\n"

-- | Our serializer, to produce JSON from a set of DropoutMaps.
instance ToJSON DropoutMaps where
  toJSON (DropoutMaps rawDropoutMaps) = toJSON $ object $ (\(a,b) -> AK.fromString (BSC.unpack a) .= b) <$> DHSI.toList rawDropoutMaps

-- | Fill a ByteString with the JSON formatted form of the given set of dropoutmaps.
{-
dropoutMapsToJSON :: NVec2F -> BSL.ByteString
dropoutMapsToJSON nVec2f
  | otherwise = A.encode $ dropoutmapsFromTensor nVec2f
    where
      dropoutmapsFromTensor (NVec2F rawDropoutMaps) = DropoutMaps $ DHSI.fromList $ zip [BSL.toStrict $ toByteString v | v <- [0,1 .. length sequences-1]] sequences
        where
          sequences = [DAR.toList rawDropoutMaps]
-}

-- | Read a set of dropout maps from a JSON formatted file, each being a map of number to list of N sets of M floats. where N is your vocabulary length, and M is a whole multiple of the vocabulary length.
dropoutMapsFromJSON :: BSL.ByteString -> NVec3F
dropoutMapsFromJSON json = NVec3F $ fromListUnboxed (Z :. dropoutMapCount :. dropoutCount :. firstDropoutLength) dropoutMapsList
  where
    dropoutCount = length (head dropoutMapsList'')
    firstDropoutLength = length (head dropoutMapsList')
    -- By performing lookup from 0-size rawDropoutMaps, we ensure a consistent space, with no gaps.
    dropoutMapsList = concat dropoutMapsList'
    dropoutMapsList' = concat dropoutMapsList''
    dropoutMapsList'' = (\a -> fromMaybe (error $ "could not lookup" <> show a <> "\n") $ lookup (BSL.toStrict $ toByteString a) rawDropoutMaps) <$> [0,1..dropoutMapCount-1]
    dropoutMapCount = size rawDropoutMaps
    (DropoutMaps rawDropoutMaps) = case eitherDecode json :: Either String DropoutMaps of
                                   Left err -> error $ "parse error when reading dropoutmaps:\n" <> err <> "\n" <> show json <> "\n"
                                   Right d -> d

-- | A 4D vector of Floats.
newtype NVec4F = NVec4F (DAR.Array U DIM4 Float)
  deriving Show

-- | Generate a list of QKVs from a hashmap.
findQKVs :: InsOrdHashMap Int QKV -> [QKV]
findQKVs weights = findQKV 0
  where
    findQKV :: Int -> [QKV]
    findQKV v = case lookup v weights of
                  Nothing  -> []
                  Just qkv -> qkv : findQKV (v+1)

-- | calculate two sets of context vectors reading from files.
-- When given 3d6-token_embeddings-3_3_1.json, 3d6-dropout_masks.json, 3d6-weights-3_4_10.json, and 6_token-vocab.json, returns a result in the appropriate shape (2*6*4 values), as seen on page 85. Said result contains the same operations and results as examples 3_5_8 and 3_5_9.
-- When given 3d6-token_embeddings-3_3_1.json, 3d6-dropout_masks-3_5_11.json, 3d6-weights-3_5_11.json, and 6_token-vocab.json, returns (more-or-less) the tensor on page 85.
example_3_5_11 :: Foldable t => HyperParams -> t a -> NVec2F -> AttentionWeights -> NVec3F -> NVec3F
example_3_5_11 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights) (NVec3F rawDropoutMaps)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- find the dropped out key*query results.
  | otherwise = res tokens qkvs
  where
    qkvs = findQKVs weights
    tokens = NVec3F $ computeS $ extend (Z :. (2::Int) :. All :. All) rawTokenEmbeddings
    res :: NVec3F -> [QKV] -> NVec3F
    res (NVec3F myTokens) myQKVs = NVec3F $ sumS myRes
      where
        myRes = moreAttentionWeights *^ moreValuesRes
        moreAttentionWeights = extend (Z :. All :. All :. foundEmbeddingsCount :. All) $ rawDropoutMaps *^ droppedAttentionWeights
          where
            droppedAttentionWeights = simpleNorm3F $ attentionWeights *^ futuresDrop
              where
                futuresDrop = extend (Z :. myQKVCount :. All :. All) $ futureDropOf foundEmbeddingsCount
                -- NOTE: the following sumS and softMax3F is why we cannot calculate two answers by doubling the length of the keyEmbeddingDimensions.
                attentionWeights = softMax3F $ sumS rawAttentionWeights
                  where
                    -- FIXME: RESEARCH: which should the following be, key, query, or valueEmbeddingsDimensions?
                    rawAttentionWeights = map (/(sqrt $ fromIntegral keyEmbeddingsDimensions)) $ moreQueriesRes *^ moreKeysRes
                    moreKeysRes = extend (Z :. All :. foundEmbeddingsCount :. All :. All) keysRes
                      where
                        keysRes = sumS $ transpose $ leftSideKeys *^ rightSideKeys
                    moreQueriesRes = extend (Z :. All :. All :. foundEmbeddingsCount :. All) queriesRes
                      where
                        queriesRes = sumS $ transpose $ leftSideQueries *^ rightSideQueries
        moreValuesRes = extend (Z :. All :. foundEmbeddingsCount :. All :. All) $ transpose valuesRes
          where
            valuesRes = sumS $ transpose rawValueRes
              where
                rawValueRes = leftSideValues *^ rightSideValues
        leftSideQueries, leftSideKeys, leftSideValues :: DAR.Array D DIM4 Float
        leftSideQueries = extend (Z :. All :. foundEmbeddingsCount :. All :. All) queries
        leftSideKeys = extend (Z :. All :. foundEmbeddingsCount :. All :. All) keys
        leftSideValues = extend (Z :. All :. foundEmbeddingsCount :. All :. All) values
        rightSideQueries = extend (Z :. All :. All :. All :. queryEmbeddingsDimensions) myTokens
        rightSideKeys = extend (Z :. All :. All :. All :. keyEmbeddingsDimensions) myTokens
        rightSideValues = extend (Z :. All :. All :. All :. valueEmbeddingsDimensions) myTokens
        queries = fromListUnboxed (Z :. myQKVCount :. queryEmbeddingsCount :. queryEmbeddingsDimensions) $ concatMap queriesFrom myQKVs
          where
            queriesFrom (QKV myQKV) = (\(NVec2F a) -> DAR.toList a) $ fromMaybe (error "no Q?") $ lookup 'Q' myQKV
        keys = fromListUnboxed (Z :. myQKVCount :. keyEmbeddingsCount :. keyEmbeddingsDimensions) $ concatMap keysFrom myQKVs
          where
            keysFrom (QKV myQKV) = (\(NVec2F a) -> DAR.toList a) $ fromMaybe (error "no K?") $ lookup 'K' myQKV
        values = fromListUnboxed (Z :. myQKVCount :. valueEmbeddingsCount :. valueEmbeddingsDimensions) $ concatMap valuesFrom  myQKVs
          where
            valuesFrom (QKV myQKV) = (\(NVec2F a) -> DAR.toList a) $ fromMaybe (error "no V?") $ lookup 'V' myQKV
        (Z :. queryEmbeddingsCount :. queryEmbeddingsDimensions) = extent $ (\(NVec2F query) -> query) $ fromMaybe (error "no Q?") $ lookup 'Q' weight
        (Z :. keyEmbeddingsCount :. keyEmbeddingsDimensions) = extent $ (\(NVec2F key) -> key) $ fromMaybe (error "no K?") $ lookup 'K' weight
        (Z :. valueEmbeddingsCount :. valueEmbeddingsDimensions) = extent $ (\(NVec2F value) -> value) $ fromMaybe (error "no V?") $ lookup 'V' weight
        myQKVCount = length myQKVs
        (QKV weight) = head myQKVs
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | calculate two sets of context vectors reading from files.
-- When given 3d6-token_embeddings-3_3_1.json, 3d6-dropout_masks-3_5_12.json, 3d6-weights-3_5_12.json, 3d6-outProjects-3_5_12.json, and 6_token-vocab.json, returns the result of 3_5_8, then 3_5_9 with 3_5_8's dropoutMask, then 3_5_9, then 3_5_8 with 3_5_9's dropoutmask.
example_3_5_12 :: Foldable t => HyperParams -> t a -> NVec2F -> AttentionWeights -> NVec3F -> NVec4F
example_3_5_12 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights) (NVec3F rawDropoutMaps)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  | mod queryEmbeddingsDimensions attentionHeads /= 0 = error $ "dimensions do not divide into heads evenly.\nFound dimensions: " <> show queryEmbeddingsDimensions <> "\n" <> show (mod attentionHeads queryEmbeddingsDimensions) <> "\n"
  -- find the dropped out key*query results.
  | otherwise = res tokens qkvs
  where
    attentionHeads :: Int
    attentionHeads = 2
    headWidth = div queryEmbeddingsDimensions attentionHeads
    qkvs = findQKVs weights
    tokens = NVec3F $ computeS $ extend (Z :. (2::Int) :. All :. All) rawTokenEmbeddings
    res :: NVec3F -> [QKV] -> NVec4F
    res (NVec3F myTokens) myQKVs = NVec4F $ sumS myRes
      where
        myRes = moreAttentionWeights *^ moreValuesRes
        moreAttentionWeights = extend (Z :. All :. All :. All :. foundEmbeddingsCount :. All) $ moreDropoutMaps *^ droppedAttentionWeights
          where
            moreDropoutMaps = extend (Z :. All :. attentionHeads :. All :. All) rawDropoutMaps
            droppedAttentionWeights = simpleNorm4F $ attentionWeights *^ futuresDrop
              where
                futuresDrop = extend (Z :. myQKVCount :. attentionHeads :. All :. All) $ futureDropOf foundEmbeddingsCount
                -- NOTE: the following softMax3F application is why we cannot calculate two answers by doubling the length of the keyEmbeddingDimensions.
                attentionWeights = softMax4F
                                   $ computeS
                                   $ backpermute (Z :. myQKVCount :. attentionHeads :. foundEmbeddingsCount :. foundEmbeddingsCount)
                                     (\(Z :. a :. b :. c :. d) -> Z :. a :. c :. d :. b)
                                   $ sumS
                                   $ reshape (Z :. myQKVCount :. foundEmbeddingsCount :. foundEmbeddingsCount :. attentionHeads :. headWidth) rawAttentionWeights
                  where
                    -- FIXME: RESEARCH: which should the following embedding dimensions be? key, query, or value?
                    rawAttentionWeights = map (/(sqrt $ fromIntegral (div keyEmbeddingsDimensions attentionHeads))) $ moreQueriesRes *^ moreKeysRes
                    moreQueriesRes = extend (Z :. All :. All :. foundEmbeddingsCount :. All) queriesRes
                      where
                        queriesRes = sumS $ transpose $ leftSideQueries *^ rightSideQueries
                    moreKeysRes = extend (Z :. All :. foundEmbeddingsCount :. All :. All) keysRes
                      where
                        keysRes = sumS $ transpose $ leftSideKeys *^ rightSideKeys
        moreValuesRes = extend (Z :. All :. All :. foundEmbeddingsCount :. All :. All)
                        $ backpermute (Z :. myQKVCount :. attentionHeads :. headWidth :. foundEmbeddingsCount)
                                      (\(Z :. a :. b :. c :. d) -> Z :. a :. d :. b*headWidth+c)
                        valuesRes
          where
            valuesRes = sumS $ transpose rawValueRes
              where
                rawValueRes = leftSideValues *^ rightSideValues
        leftSideQueries = extend (Z :. All :. foundEmbeddingsCount :. All :. All) queries
        leftSideKeys = extend (Z :. All :. foundEmbeddingsCount :. All :. All) keys
        leftSideValues = extend (Z :. All :. foundEmbeddingsCount :. All :. All) values
        rightSideQueries = extend (Z :. All :. All :. All :. queryEmbeddingsDimensions) myTokens
        rightSideKeys = extend (Z :. All :. All :. All :. keyEmbeddingsDimensions) myTokens
        rightSideValues = extend (Z :. All :. All :. All :. valueEmbeddingsDimensions) myTokens
        queries = fromListUnboxed (Z :. myQKVCount :. queryEmbeddingsCount :. queryEmbeddingsDimensions) $ concatMap queriesFrom myQKVs
          where
            queriesFrom (QKV myQKV) = (\(NVec2F a) -> DAR.toList a) $ fromMaybe (error "no Q?") $ lookup 'Q' myQKV
        keys = fromListUnboxed (Z :. myQKVCount :. keyEmbeddingsCount :. keyEmbeddingsDimensions) $ concatMap keysFrom myQKVs
          where
            keysFrom (QKV myQKV) = (\(NVec2F a) -> DAR.toList a) $ fromMaybe (error "no K?") $ lookup 'K' myQKV
        values = fromListUnboxed (Z :. myQKVCount :. valueEmbeddingsCount :. valueEmbeddingsDimensions) $ concatMap valuesFrom myQKVs
          where
            valuesFrom (QKV myQKV) = (\(NVec2F a) -> DAR.toList a) $ fromMaybe (error "no V?") $ lookup 'V' myQKV
        (Z :. keyEmbeddingsCount :. keyEmbeddingsDimensions) = extent $ (\(NVec2F key) -> key) $ fromMaybe (error "no K?") $ lookup 'K' weight
        (Z :. valueEmbeddingsCount :. valueEmbeddingsDimensions) = extent $ (\(NVec2F value) -> value) $ fromMaybe (error "no V?") $ lookup 'V' weight
        myQKVCount = length myQKVs
    (Z :. queryEmbeddingsCount :. queryEmbeddingsDimensions) = extent $ (\(NVec2F query) -> query) $ fromMaybe (error "no Q?") $ lookup 'Q' weight
    (QKV weight) = head qkvs
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | A 5D vector of Floats.
newtype NVec5F = NVec5F (DAR.Array U DIM5 Float)

-- | calculate two sets of context vectors reading from files.
-- When given 3d6-token_embeddings-3_3_1.json, 3d6-dropout_masks-3_5_13.json, 3d6-weights-3_5_12.json, and 6_token-vocab.json, returns the result of 3_5_8, then 3_5_9, then 3_5_9, then 3_5_8.
example_3_5_13 :: Foldable t => HyperParams -> t a -> NVec2F -> AttentionWeights -> NVec3F -> NVec4F
example_3_5_13 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights) (NVec3F rawDropoutMaps)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  | mod queryEmbeddingsDimensions attentionHeads /= 0 = error $ "dimensions do not divide into heads evenly.\nFound dimensions: " <> show queryEmbeddingsDimensions <> "\n" <> show (mod attentionHeads queryEmbeddingsDimensions) <> "\n"
  -- find the dropped out key*query results.
  | otherwise = res tokens qkvs
  where
    attentionHeads :: Int
    attentionHeads = 2
    headWidth = div queryEmbeddingsDimensions attentionHeads
    qkvs = findQKVs weights
    tokens = NVec3F $ computeS $ extend (Z :. (2::Int) :. All :. All) rawTokenEmbeddings
    res :: NVec3F -> [QKV] -> NVec4F
    res (NVec3F myTokens) myQKVs = NVec4F $ sumS myRes
      where
        myRes = moreAttentionWeights *^ moreValuesRes
        moreAttentionWeights = extend (Z :. All :. All :. All :. foundEmbeddingsCount :. All) $ moreDropoutMaps *^ droppedAttentionWeights
          where
            moreDropoutMaps
              | mod dropoutMapRowLength attentionHeads /= 0 = error "dropout map row length not divisible by number of attention heads."
              | attentionHeads == 1 = extend (Z :. All :. attentionHeads :. All :. All) rawDropoutMaps
              | div dropoutMapRowLength attentionHeads == foundEmbeddingsCount = backpermute (Z :. myQKVCount :. attentionHeads :. dropoutMapRowCount :. dropoutMapHeadWidth)
                          (\(Z :. a :. b :. c :. d) -> Z :. a :. c :. b*dropoutMapHeadWidth+d)
                          rawDropoutMaps
              | otherwise = error $ "failed to divide into attention heads properly\n"
                                 <> show (div dropoutMapRowLength attentionHeads) <> "\n"
                                 <> show foundEmbeddingsCount <> "\n"
              where
                dropoutMapHeadWidth = div dropoutMapRowLength attentionHeads
                (Z :. _ :. dropoutMapRowCount :. dropoutMapRowLength) = extent rawDropoutMaps
            droppedAttentionWeights = simpleNorm4F $ attentionWeights *^ futuresDrop
              where
                futuresDrop = extend (Z :. myQKVCount :. attentionHeads :. All :. All) $ futureDropOf foundEmbeddingsCount
                -- NOTE: the following softMax3F application is why we cannot calculate two answers by doubling the length of the keyEmbeddingDimensions.
                attentionWeights = softMax4F
                                   $ computeS
                                   $ backpermute (Z :. myQKVCount :. attentionHeads :. foundEmbeddingsCount :. foundEmbeddingsCount)
                                     (\(Z :. a :. b :. c :. d) -> Z :. a :. c :. d :. b)
                                   $ sumS
                                   $ reshape (Z :. myQKVCount :. foundEmbeddingsCount :. foundEmbeddingsCount :. attentionHeads :. headWidth) rawAttentionWeights
                  where
                    -- FIXME: RESEARCH: which should the following embedding dimensions be? key, query, or value?
                    rawAttentionWeights = map (/(sqrt $ fromIntegral (div keyEmbeddingsDimensions attentionHeads))) $ moreQueriesRes *^ moreKeysRes
                    moreQueriesRes = extend (Z :. All :. All :. foundEmbeddingsCount :. All) queriesRes
                      where
                        queriesRes = sumS $ transpose $ leftSideQueries *^ rightSideQueries
                    moreKeysRes = extend (Z :. All :. foundEmbeddingsCount :. All :. All) keysRes
                      where
                        keysRes = sumS $ transpose $ leftSideKeys *^ rightSideKeys
        moreValuesRes = extend (Z :. All :. All :. foundEmbeddingsCount :. All :. All)
                        $ backpermute (Z :. myQKVCount :. attentionHeads :. headWidth :. foundEmbeddingsCount)
                                      (\(Z :. a :. b :. c :. d) -> Z :. a :. d :. b*headWidth+c)
                        valuesRes
          where
            valuesRes = sumS $ transpose rawValueRes
              where
                rawValueRes = leftSideValues *^ rightSideValues
        leftSideQueries = extend (Z :. All :. foundEmbeddingsCount :. All :. All) queries
        leftSideKeys = extend (Z :. All :. foundEmbeddingsCount :. All :. All) keys
        leftSideValues = extend (Z :. All :. foundEmbeddingsCount :. All :. All) values
        rightSideQueries = extend (Z :. All :. All :. All :. queryEmbeddingsDimensions) myTokens
        rightSideKeys = extend (Z :. All :. All :. All :. keyEmbeddingsDimensions) myTokens
        rightSideValues = extend (Z :. All :. All :. All :. valueEmbeddingsDimensions) myTokens
        queries = fromListUnboxed (Z :. myQKVCount :. queryEmbeddingsCount :. queryEmbeddingsDimensions) $ concatMap queriesFrom myQKVs
          where
            queriesFrom (QKV myQKV) = (\(NVec2F a) -> DAR.toList a) $ fromMaybe (error "no Q?") $ lookup 'Q' myQKV
        keys = fromListUnboxed (Z :. myQKVCount :. keyEmbeddingsCount :. keyEmbeddingsDimensions) $ concatMap keysFrom myQKVs
          where
            keysFrom (QKV myQKV) = (\(NVec2F a) -> DAR.toList a) $ fromMaybe (error "no K?") $ lookup 'K' myQKV
        values = fromListUnboxed (Z :. myQKVCount :. valueEmbeddingsCount :. valueEmbeddingsDimensions) $ concatMap valuesFrom myQKVs
          where
            valuesFrom (QKV myQKV) = (\(NVec2F a) -> DAR.toList a) $ fromMaybe (error "no V?") $ lookup 'V' myQKV
        (Z :. keyEmbeddingsCount :. keyEmbeddingsDimensions) = extent $ (\(NVec2F key) -> key) $ fromMaybe (error "no K?") $ lookup 'K' weight
        (Z :. valueEmbeddingsCount :. valueEmbeddingsDimensions) = extent $ (\(NVec2F value) -> value) $ fromMaybe (error "no V?") $ lookup 'V' weight
        myQKVCount = length myQKVs
    (Z :. queryEmbeddingsCount :. queryEmbeddingsDimensions) = extent $ (\(NVec2F query) -> query) $ fromMaybe (error "no Q?") $ lookup 'Q' weight
    (QKV weight) = head qkvs
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

-- | calculate two sets of context vectors reading from files.
-- When given 3d6-token_embeddings-3_3_1.json, 3d6-dropout_masks-3_5_13.json, 3d6-weights-3_5_12.json, 6_token-vocab.json, and 3d6-out_projections-3_5_14.json, returns the result of 3_5_8, then 3_5_9, then 3_5_9, then 3_5_8.
example_3_5_14 :: Foldable t => HyperParams -> t a -> NVec2F -> AttentionWeights -> NVec3F -> NVec2F -> NVec4F
example_3_5_14 (HyperParams embeddingDimensions _) jsonDictionary (NVec2F rawTokenEmbeddings) (AttentionWeights weights) (NVec3F rawDropoutMaps) (NVec2F outProject)
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show foundEmbeddingsDimensions <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show foundEmbeddingsCount <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  | mod queryEmbeddingsDimensions attentionHeads /= 0 = error $ "dimensions do not divide into heads evenly.\nFound dimensions: " <> show queryEmbeddingsDimensions <> "\n" <> show (mod attentionHeads queryEmbeddingsDimensions) <> "\n"
  -- find the dropped out key*query results.
  | otherwise = res tokens qkvs
  where
    attentionHeads :: Int
    attentionHeads = 2
    headWidth = div queryEmbeddingsDimensions attentionHeads
    qkvs = findQKVs weights
    tokens = NVec3F $ computeS $ extend (Z :. (2::Int) :. All :. All) rawTokenEmbeddings
    res :: NVec3F -> [QKV] -> NVec4F
    res (NVec3F myTokens) myQKVs = NVec4F $ computeS $ unScaledRes *^ moreLinearScale
      where
        moreLinearScale = extend (Z :. All :. attentionHeads :. foundEmbeddingsCount :. All) outProject
        unScaledRes = sumS $ moreAttentionWeights *^ moreValuesRes
        moreAttentionWeights = extend (Z :. All :. All :. All :. foundEmbeddingsCount :. All) $ moreDropoutMaps *^ droppedAttentionWeights
          where
            moreDropoutMaps
              | mod dropoutMapRowLength attentionHeads /= 0 = error "dropout map row length not divisible by number of attention heads."
              | attentionHeads == 1 = extend (Z :. All :. attentionHeads :. All :. All) rawDropoutMaps
              | div dropoutMapRowLength attentionHeads == foundEmbeddingsCount = backpermute (Z :. myQKVCount :. attentionHeads :. dropoutMapRowCount :. dropoutMapHeadWidth)
                          (\(Z :. a :. b :. c :. d) -> Z :. a :. c :. b*dropoutMapHeadWidth+d)
                          rawDropoutMaps
              | otherwise = error $ "failed to divide into attention heads properly\n"
                                 <> show (div dropoutMapRowLength attentionHeads) <> "\n"
                                 <> show foundEmbeddingsCount <> "\n"
              where
                dropoutMapHeadWidth = div dropoutMapRowLength attentionHeads
                (Z :. _ :. dropoutMapRowCount :. dropoutMapRowLength) = extent rawDropoutMaps
            droppedAttentionWeights = simpleNorm4F $ attentionWeights *^ futuresDrop
              where
                futuresDrop = extend (Z :. myQKVCount :. attentionHeads :. All :. All) $ futureDropOf foundEmbeddingsCount
                -- NOTE: the following softMax3F application is why we cannot calculate two answers by doubling the length of the keyEmbeddingDimensions.
                attentionWeights = softMax4F
                                   $ computeS
                                   $ backpermute (Z :. myQKVCount :. attentionHeads :. foundEmbeddingsCount :. foundEmbeddingsCount)
                                     (\(Z :. a :. b :. c :. d) -> Z :. a :. c :. d :. b)
                                   $ sumS
                                   $ reshape (Z :. myQKVCount :. foundEmbeddingsCount :. foundEmbeddingsCount :. attentionHeads :. headWidth) rawAttentionWeights
                  where
                    -- FIXME: RESEARCH: which should the following embedding dimensions be? key, query, or value?
                    rawAttentionWeights = map (/(sqrt $ fromIntegral (div keyEmbeddingsDimensions attentionHeads))) $ moreQueriesRes *^ moreKeysRes
                    moreQueriesRes = extend (Z :. All :. All :. foundEmbeddingsCount :. All) queriesRes
                      where
                        queriesRes = sumS $ transpose $ leftSideQueries *^ rightSideQueries
                    moreKeysRes = extend (Z :. All :. foundEmbeddingsCount :. All :. All) keysRes
                      where
                        keysRes = sumS $ transpose $ leftSideKeys *^ rightSideKeys
        moreValuesRes = extend (Z :. All :. All :. foundEmbeddingsCount :. All :. All)
                        $ backpermute (Z :. myQKVCount :. attentionHeads :. headWidth :. foundEmbeddingsCount)
                                      (\(Z :. a :. b :. c :. d) -> Z :. a :. d :. b*headWidth+c)
                        valuesRes
          where
            valuesRes = sumS $ transpose rawValueRes
              where
                rawValueRes = leftSideValues *^ rightSideValues
        leftSideQueries = extend (Z :. All :. foundEmbeddingsCount :. All :. All) queries
        leftSideKeys = extend (Z :. All :. foundEmbeddingsCount :. All :. All) keys
        leftSideValues = extend (Z :. All :. foundEmbeddingsCount :. All :. All) values
        rightSideQueries = extend (Z :. All :. All :. All :. queryEmbeddingsDimensions) myTokens
        rightSideKeys = extend (Z :. All :. All :. All :. keyEmbeddingsDimensions) myTokens
        rightSideValues = extend (Z :. All :. All :. All :. valueEmbeddingsDimensions) myTokens
        queries = fromListUnboxed (Z :. myQKVCount :. queryEmbeddingsCount :. queryEmbeddingsDimensions) $ concatMap queriesFrom myQKVs
          where
            queriesFrom (QKV myQKV) = (\(NVec2F a) -> DAR.toList a) $ fromMaybe (error "no Q?") $ lookup 'Q' myQKV
        keys = fromListUnboxed (Z :. myQKVCount :. keyEmbeddingsCount :. keyEmbeddingsDimensions) $ concatMap keysFrom myQKVs
          where
            keysFrom (QKV myQKV) = (\(NVec2F a) -> DAR.toList a) $ fromMaybe (error "no K?") $ lookup 'K' myQKV
        values = fromListUnboxed (Z :. myQKVCount :. valueEmbeddingsCount :. valueEmbeddingsDimensions) $ concatMap valuesFrom myQKVs
          where
            valuesFrom (QKV myQKV) = (\(NVec2F a) -> DAR.toList a) $ fromMaybe (error "no V?") $ lookup 'V' myQKV
        (Z :. keyEmbeddingsCount :. keyEmbeddingsDimensions) = extent $ (\(NVec2F key) -> key) $ fromMaybe (error "no K?") $ lookup 'K' weight
        (Z :. valueEmbeddingsCount :. valueEmbeddingsDimensions) = extent $ (\(NVec2F value) -> value) $ fromMaybe (error "no V?") $ lookup 'V' weight
        myQKVCount = length myQKVs
    (Z :. queryEmbeddingsCount :. queryEmbeddingsDimensions) = extent $ (\(NVec2F query) -> query) $ fromMaybe (error "no Q?") $ lookup 'Q' weight
    (QKV weight) = head qkvs
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings

simpleNorm :: DAR.Array D DIM2 Float -> DAR.Array D DIM2 Float
simpleNorm inRawVec = inRawVec /^ extend (Z :. All :. foundItems) (sumS inRawVec)
  where
    (Z :. _ :. foundItems) = extent inRawVec

simpleNorm3F :: DAR.Array D DIM3 Float -> DAR.Array D DIM3 Float
simpleNorm3F inRawVec = inRawVec /^ extend (Z :. All :. All :. foundItems) (sumS inRawVec)
  where
    (Z :. _ :. _ :. foundItems) = extent inRawVec

simpleNorm4F :: DAR.Array D DIM4 Float -> DAR.Array D DIM4 Float
simpleNorm4F inRawVec = inRawVec /^ extend (Z :. All :. All :. All :. foundItems) (sumS inRawVec)
  where
    (Z :. _ :. _ :. _ :. foundItems) = extent inRawVec

softMax1F :: DAR.Array U DIM1 Float -> DAR.Array D DIM1 Float
softMax1F inRawVec = map exp inRawVec /^ extend (Z :. foundItems) (sumS $ map exp inRawVec)
  where
    (Z :. foundItems) = extent inRawVec

softMax :: DAR.Array U DIM2 Float -> DAR.Array D DIM2 Float
softMax inRawVec = map exp inRawVec /^ extend (Z :. All :. foundItems) (sumS $ map exp inRawVec)
  where
    (Z :. _ :. foundItems) = extent inRawVec

softMax3F :: DAR.Array U DIM3 Float -> DAR.Array D DIM3 Float
softMax3F inRawVec = map exp inRawVec /^ extend (Z :. All :. All :. foundItems) (sumS $ map exp inRawVec)
  where
    (Z :. _ :. _ :. foundItems) = extent inRawVec

softMax4F :: DAR.Array U DIM4 Float -> DAR.Array D DIM4 Float
softMax4F inRawVec = map exp inRawVec /^ extend (Z :. All :. All :. All :. foundItems) (sumS $ map exp inRawVec)
  where
    (Z :. _ :. _ :. _ :. foundItems) = extent inRawVec

futureDropOf :: Int -> DAR.Array U DIM2 Float
futureDropOf foundEmbeddingsCount = fromListUnboxed (Z :. foundEmbeddingsCount :. foundEmbeddingsCount) $ concat $ [[ if y > x then 0 else 1 | y <- [1,2..foundEmbeddingsCount]] | x <- [1,2..foundEmbeddingsCount]]

-- | A type for Embeddings, as they come out of the JSON file.
newtype Embeddings = Embeddings (InsOrdHashMap BSS.ByteString [Float])

-- | Our parser for an embeddings file.
instance FromJSON Embeddings where
  parseJSON = withObject "Embeddings" (pure . findEmbeddings . DAKM.toList)
    where
      findEmbeddings :: [(Key, Value)] -> Embeddings
      findEmbeddings maybeEmbeddings = Embeddings embeddings
        where
          embeddings = embeddings' maybeEmbeddings empty
          embeddings' :: [(Key, Value)] -> InsOrdHashMap BSS.ByteString [Float] -> InsOrdHashMap BSS.ByteString [Float]
          embeddings' [] myEmbeddings = myEmbeddings
          embeddings' [(k,v)] myEmbeddings = insert (encodeUtf8 $ toText k) (numbersFromValue v) myEmbeddings
          embeddings' ((k,v):xs) myEmbeddings = insert (encodeUtf8 $ toText k) (numbersFromValue v) (embeddings' xs myEmbeddings)
          numbersFromValue :: Value -> [Float]
          numbersFromValue (Array vs) = (\(Number a) -> toRealFloat a) <$> DV.toList vs
          numbersFromValue a = error $ "failed to parse " <> show a <> " as an Array.\n"

-- | Our serializer, to produce JSON from a set of Embeddings.
instance ToJSON Embeddings where
  toJSON (Embeddings rawEmbeddings) = toJSON $ object $ (\(a,b) -> AK.fromString (BSC.unpack a) .= b) <$> DHSI.toList rawEmbeddings

-- | Fill a ByteString with the JSON formatted form of the given set of embeddings.
embeddingsToJSON :: NVec2F -> BSL.ByteString
embeddingsToJSON nVec2f = A.encode $ embeddingsFromTensor nVec2f
    where
      embeddingsFromTensor (NVec2F rawEmbeddings) = Embeddings $ DHSI.fromList $ zip [BSL.toStrict $ toByteString v | v <- [0,1 .. length sequences-1]] sequences
        where
          sequences = [DAR.toList rawEmbeddings]

-- | Read a set of embeddings from a JSON formatted map of number to list of N sets of D floats. where N is your vocabulary length, and D is your embeddings dimensions.
embeddingsFromJSON :: BSL.ByteString -> NVec2F
embeddingsFromJSON json = NVec2F $ fromListUnboxed (Z :. size rawEmbeddings :. firstEmbeddingLength) embeddingsList
  where
    (Embeddings rawEmbeddings) = case eitherDecode json :: Either String Embeddings of
                                   Left err -> error $ "parse error when reading embeddings:\n" <> err <> "\n" <> show json <> "\n"
                                   Right d -> d
    -- By performing lookup from 0-size rawEmbeddings, we ensure a consistent space, with no gaps.
    embeddingsList = concatMap (\a -> fromMaybe (error $ "could not lookup" <> show a <> "\n") $ lookup (BSL.toStrict $ toByteString a) rawEmbeddings) [0,1..size rawEmbeddings-1]
    firstEmbeddingLength = length $ fromMaybe (error "failed to lookup first embedding (0)." ) $ lookup "0" rawEmbeddings

-- | A type for ProjectionWeights, as they come out of the JSON file.
newtype ProjectionWeights = ProjectionWeights (InsOrdHashMap BSS.ByteString [Float])
 deriving Show

-- | Our parser for a file containing projection Weights.
instance FromJSON ProjectionWeights where
  parseJSON = withObject "ProjectionWeights" (pure . findProjectionWeights . DAKM.toList)
    where
      findProjectionWeights :: [(Key, Value)] -> ProjectionWeights
      findProjectionWeights maybeProjectionWeights = ProjectionWeights projectionWeights
        where
          projectionWeights = projectionWeights' maybeProjectionWeights empty
          projectionWeights' :: [(Key, Value)] -> InsOrdHashMap BSS.ByteString [Float] -> InsOrdHashMap BSS.ByteString [Float]
          projectionWeights' [] myProjectionWeights = myProjectionWeights
          projectionWeights' [(k,v)] myProjectionWeights = insert (encodeUtf8 $ toText k) (numbersFromValue v) myProjectionWeights
          projectionWeights' ((k,v):xs) myProjectionWeights = insert (encodeUtf8 $ toText k) (numbersFromValue v) (projectionWeights' xs myProjectionWeights)
          numbersFromValue :: Value -> [Float]
          numbersFromValue (Array vs) = (\(Number a) -> toRealFloat a) <$> DV.toList vs
          numbersFromValue a = error $ "failed to parse " <> show a <> " as an Array.\n"

-- | Our serializer, to produce JSON from a set of ProjectionWeights.
instance ToJSON ProjectionWeights where
  toJSON (ProjectionWeights rawProjectionWeights) = toJSON $ object $ (\(a,b) -> AK.fromString (BSC.unpack a) .= b) <$> DHSI.toList rawProjectionWeights

-- | Fill a ByteString with the JSON formatted form of the given set of projectionWeights.
projectionWeightsToJSON :: NVec2F -> BSL.ByteString
projectionWeightsToJSON nVec2f = A.encode $ projectionWeightsFromTensor nVec2f
    where
      projectionWeightsFromTensor (NVec2F rawProjectionWeights) = ProjectionWeights $ DHSI.fromList $ zip [BSL.toStrict $ toByteString v | v <- [0,1 .. length sequences-1]] sequences
        where
          sequences = [DAR.toList rawProjectionWeights]

-- | Read a set of projection weights from a JSON formatted map of number to list of N sets of D floats. where N is your vocabulary length, and D is your projectionWeights dimensions.
projectionWeightsFromJSON :: BSL.ByteString -> NVec2F
projectionWeightsFromJSON json = NVec2F $ fromListUnboxed (Z :. size rawProjectionWeights :. firstEmbeddingLength) projectionWeightsList
  where
    (ProjectionWeights rawProjectionWeights) = case eitherDecode json :: Either String ProjectionWeights of
                                   Left err -> error $ "parse error when reading projectionWeights:\n" <> err <> "\n" <> show json <> "\n"
                                   Right d -> d
    -- By performing lookup from 0-size rawProjectionWeights, we ensure a consistent space, with no gaps.
    projectionWeightsList = concatMap (\a -> fromMaybe (error $ "could not lookup" <> show a <> "\n") $ lookup (BSL.toStrict $ toByteString a) rawProjectionWeights) [0,1..size rawProjectionWeights-1]
    firstEmbeddingLength = length $ fromMaybe (error "failed to lookup first embedding (0).") $ lookup "0" rawProjectionWeights

-- | Read a dictionary from a JSON formatted map.
dictionaryFromJSON :: BSL.ByteString -> Vocab
dictionaryFromJSON json = case eitherDecode json :: Either String Bacov of
                            Left err -> error $ "parse error when reading dictionary:\n" <> err <> "\n" <> show json <> "\n"
                            Right d -> flipBacov d

-- | The default starting vocabulary, taken from the first 256 tokens of gpt2.
initVocabGPT2 :: Vocab
initVocabGPT2 = flipBacov defaultBacov


flipBacov ::  Bacov -> Vocab
flipBacov vk = DHSI.fromList (swap <$> DHSI.toList ((\(Bacov v) -> v) vk))

-- | The default starting vocabulary, taken from the first 256 tokens of gpt2.
-- An initial (reverse) vocabulary, consisting of....
defaultBacov :: Bacov
defaultBacov = Bacov $ DHSI.fromList $ zip (BSS.singleton <$> [33, 34..]) [0,1..93]                        -- The first 94 characters of ascii, after 33 control signals.
                                     <> zip ((\a -> BSS.pack [194,128+a]) <$> [33,34..44]) [94, 95..105]   -- UTF8 characters U+00A1-U+00AB
                                     <> zip ((\a -> BSS.pack [194,128+a]) <$> [46,47..63]) [106, 107..123] -- UTF8 characters U+00AD-U+00BF
                                     <> zip ((\a -> BSS.pack [195,128+a]) <$> [0,1..63]) [124, 125..187]   -- UTF8 characters U+00C0-U+00FF
                                     <> zip ((\a -> BSS.pack [196,128+a]) <$> [0,1..63]) [188, 189..251]   -- UTF8 characters U+0100-U+013F
                                     <> zip ((\a -> BSS.pack [197,128+a]) <$> [0,1..63]) [252, 253..255]   -- UTF8 characters U+0140-U+017F

-- | Convert an ASCII string into a sequence of tokens in the initVocabGPT2 token space.
initSeqGPT2 :: BSS.ByteString -> Seq
initSeqGPT2 text = conv <$> BSS.unpack text
  where
    conv :: Word8 -> Id
    conv chr
      -- FIXME: 10 is known to be a carriage return, 32 is space.. the rest is a guess.
      | chr > 9 && chr < 34 = fromIntegral $ chr + 188
      -- ASCII 34-128 -> Int 1-94
      | (chr > 33) && (chr < (94 + 33)) = fromIntegral $ chr - 33
      | otherwise = error $ "whoops: " <> show chr <> "\n"

-- | Perform any changes we need before presenting a string to the user. basically, the opposite of 'initSeqGP2'
respaceGPT2 :: BSS.ByteString -> BSS.ByteString
respaceGPT2 bs = BSS.pack $ respaceGPT2' (BSS.unpack bs)
  where
    respaceGPT2' [] = []
    respaceGPT2' [x] = [x]
    respaceGPT2' (x:y:xs)
      -- 33(?) special characters, starting at ascii 10, and ending at 33.
      -- Yes, we are converting them to unicode escape sequences, per defaultBacov above.
      -- For GPT2, a special character representing a single space is translated to 'Ġ', UTF8 character U+120 .
      | x == 196 && y > (9 + 128) && y < (34 + 128) = y - 128 : respaceGPT2' xs
      -- Regular characters. No, I do not know why we don't have to shift them like we do in initSeqGPT2.
      | otherwise = x : respaceGPT2' (y:xs)

-- | A type, for expressing which sequences need to be swapped out after tokenization, to insert a new token into the stream.
type Extensions = [(Seq, Seq)]

-- | After-the-fact, accept a fully encoded sequence, and a list of extensions, and apply those extensions.
-- FIXME: should be done during the fact, after the first Seq?
encodeExtensions :: Extensions -> Seq -> Seq
encodeExtensions extensions inSeq = case extensions of
                                      [] -> inSeq
                                      [(find,replaceWith)] -> replace find replaceWith inSeq
                                      ((find,replaceWith):xs) -> replace find replaceWith $ encodeExtensions xs inSeq

-- | A set of filters, to add additional tokens to our encoded text.
-- note: since these are applied after the fact.. that we will end up with many permutations for one token.
extensionsGPT2 :: Extensions
extensionsGPT2 = [([1279,91,437,1659,5239,91,29],[220,50256])] -- " <|endoftext|>"

-- | Additional filters, to handle encoding extensions.
encodeExtensionsGPT2 :: Seq -> Seq
encodeExtensionsGPT2 = encodeExtensions extensionsGPT2

-- | Add the extra vocab that is in the JSON file, but not the merges TXT file. GPT2 vocabulary version.
extendVocabGPT2 :: Vocab -> Vocab
extendVocabGPT2 v = insert (size v) "<|endoftext|>" v

-- | Add the extra vocab that is in the JSON file, but not the merges TXT file. Also, add a token for encoding unknown values. GPT2 vocabulary version.
extendVocabGPT2Unk :: Vocab -> Vocab
extendVocabGPT2Unk v = insert (size v) "<|endoftext|>" (insert (size v +1) "<|unk|>" v)

-- | Read a set of Merges from a TXT file.
-- Expects:
-- First line: #version 0.2\n
-- All other lines: <string1><space><string2>
-- each line coresponds to a token of <string1><string2>, starting from token 256, in a vocabulary.
-- In strings, a special character stands in for a single space.
mergesFromTXT :: BSL.ByteString -> Merges
mergesFromTXT text = DHSI.fromList (splitLineRecurse defaultBacov [] $ zip [256,257..] (Left <$> drop 1 (BSLU.lines mergeLines)))
  where
    -- Discard the first line of the file.
    (_,mergeLines) = BSLU.break (== '\n') text
    splitLineRecurse :: Bacov -> [((Int, Int), Int)] -> [(Int, Either BSL.ByteString (Either BSS.ByteString Int, BSS.ByteString))] -> [((Int, Int), Int)]
    splitLineRecurse bacovIn mergesDone mergesTodo = case lefts res of
                                                       [] -> mergesDone <> rights res
                                                       _ -> splitLineRecurse newBacov (mergesDone <> rights res) (lefts res)
      where
        -- new entries, from this layer.
        foundBacov :: Bacov
        foundBacov = Bacov $ DHSI.fromList ((\(a,b) -> (BSL.toStrict (byteStringFrom a),b)) <$> rights res)
          where
           byteStringFrom :: (Int, Int) -> BSL.ByteString
           byteStringFrom (tok1, tok2) =  case lookup tok1 vocabIn of
                                            Nothing -> error $ show (lookup tok1 vocabIn) <> "\n" <> show tok2 <> "\n"
                                            (Just pre) -> case lookup tok2 vocabIn of
                                                            Nothing -> error $ show (lookup tok2 vocabIn) <> "\n" <> show tok2 <> "\n"
                                                            (Just post) -> BSL.fromStrict $ pre <> post
             where
               vocabIn = flipBacov bacovIn
        newBacov :: Bacov
        newBacov = unionBacovs bacovIn foundBacov
          where
            unionBacovs :: Bacov -> Bacov -> Bacov
            unionBacovs (Bacov rawBacov1) (Bacov rawBacov2) = Bacov $ DHSI.union rawBacov1 rawBacov2
        res :: [Either (Int, Either BSL.ByteString (Either BSS.ByteString Int, BSS.ByteString)) ((Int, Int), Int)]
        res = splitLine bacovIn <$> mergesTodo
    splitLine :: Bacov -> (Int, Either BSL.ByteString (Either BSS.ByteString Int, BSS.ByteString)) -> Either (Int, Either BSL.ByteString (Either BSS.ByteString Int, BSS.ByteString)) ((Int, Int), Int)
    splitLine (Bacov rawBacov) tokenMap@(tokenNumber, Right (Right token1, post)) =
      case lookup post rawBacov of
        Nothing -> Left tokenMap
        Just token2 -> Right ((token1, token2), tokenNumber)
    splitLine (Bacov rawBacov) tokenMap@(tokenNumber, Right (Left pre, post)) =
      case lookup pre rawBacov of
        Nothing -> Left tokenMap
        Just token1 -> case lookup post rawBacov of
                         Nothing -> Left (tokenNumber, Right (Right token1, post))
                         Just token2 -> Right ((token1, token2), tokenNumber)
    splitLine (Bacov rawBacov) (tokenNumber, Left string) =
      case lookup pre rawBacov of
        Nothing -> Left (tokenNumber, Right (Left pre, post))
        Just token1 -> case lookup post rawBacov of
                         Nothing -> Left (tokenNumber, Right (Right token1, post))
                         Just token2 -> Right ((token1, token2), tokenNumber)
      where
        (pre, post) = (\(a,b) -> (BSL.toStrict a, BSL.toStrict $ BSLU.drop 1 b)) $ BSLU.break (== ' ') string

-- | select which example to run.
run :: TrainRootOpts -> IO ()
run rawArgs =
  let
    readInput :: IO String
    readInput = maybe getContents PL.readFile (inputFileOpt rawArgs)
    readDictionary :: IO (InsOrdHashMap Id BSS.ByteString)
    readDictionary = do
      input <- case dictionaryOpt rawArgs of
                 Nothing -> error "This example requires you to pass in your own dictionary, in JSON format."
                 Just inFile -> BSL.readFile inFile
      return (dictionaryFromJSON input)
    readMerges :: IO (InsOrdHashMap Pair Id)
    readMerges = do
      input <- case mergesOpt rawArgs of
                 Nothing -> error "This example requires you to pass in your own merges file, in text format."
                 Just inFile -> BSL.readFile inFile
      return (mergesFromTXT input)
    readEmbeddings :: IO NVec2F
    readEmbeddings = do
      input <- case tokenEmbeddingsOpt rawArgs of
                 Nothing -> error "This example requires you to pass in a set of token embeddings, in JSON format."
                 Just inFile -> BSL.readFile inFile
      return (embeddingsFromJSON input)
    readWeights :: IO AttentionWeights
    readWeights = do
      input <- case attentionWeightsOpt rawArgs of
                 Nothing -> error "This example requires you to pass in a set of attention weights, in JSON format."
                 Just inFile -> BSL.readFile inFile
      return (getWeights input)
        where
          getWeights input = case eitherDecode input :: Either String AttentionWeights of
                               (Left s) -> error $ show s <> "\n"
                               (Right w) -> w
    readDropoutMaps :: IO NVec3F
    readDropoutMaps = do
      input <- case dropoutMapsOpt rawArgs of
                 Nothing -> error "This example requires you to pass in a set of dropout maps, in JSON format."
                 Just inFile -> BSL.readFile inFile
      return (dropoutMapsFromJSON input)
    readOutProjectionWeights :: IO NVec2F
    readOutProjectionWeights = do
      input <- case outProjectionWeightsOpt rawArgs of
                 Nothing -> error "This example requires you to pass in a set of output projection weights, in JSON format."
                 Just inFile -> BSL.readFile inFile
      return (embeddingsFromJSON input)
    beVerbose = fromMaybe False (verboseFlag rawArgs)
    hyperParams :: HyperParams
    hyperParams = HyperParams maybeEmbeddingDimensions maybeAttentionWeightDimensions
      where
        maybeEmbeddingDimensions = case embeddingDimensionsOpt rawArgs of
                    Nothing -> error "This example requires you to specify a number of dimensions each embedding recieves."
                    Just a -> if a < 1
                              then error "You must specify a positive number of dimensions for your embeddings."
                              else a
        maybeAttentionWeightDimensions = case attentionWeightDimensionsOpt rawArgs of
                    Nothing -> error "This example requires you to specify a number of dimensions each attention weight recieves."
                    Just a -> if a < 1
                              then error "You must specify a positive number of dimensions for your attention weights."
                              else a
  in
    case exampleOpt rawArgs of
      Example (2,1) -> do
        input <- readInput
        if beVerbose
          then putStrLn $ "running listing 2.1 with input:\n<input>\n" <> input <> "</input>\n<result>" <> example_2_1 input <> "</result>\n"
          else putStrLn $ example_2_1 input
      Example (2,2) -> do
        input <- readInput
        print (example_2_2 input)
      Example (2,3) -> do
        input <- readInput
        putStrLn $  show (example_2_3_1 input example_2_3_String) <> "\n"
                 <> show (example_2_3_2 input $ example_2_3_1 input example_2_3_String) <> "\n"
      Example (2,4) -> do
        input <- readInput
        putStrLn $  show (example_2_4_1 input) <> "\n"
                 <> show example_2_4_String <> "\n"
                 <> show (example_2_4_2 input example_2_4_String) <> "\n"
                 <> show (example_2_4_3 input $ example_2_4_2 input example_2_4_String) <> "\n"
      Example (2,5) -> do
        dictionary <- readDictionary
        merges <- readMerges
        putStrLn $ "Encode:\n" <> show (example_2_5_1 example_2_5_String merges dictionary) <> "\n"
                <> "Decode:\n" <> show (respaceGPT2 $ example_2_5_2 example_2_5_Seq merges dictionary) <> "\n"
                <> "Decode(Encode):\n" <> show (respaceGPT2 $ example_2_5_2 (example_2_5_1 example_2_5_String merges dictionary) merges dictionary) <> "\n"
      Example (2,6) -> do
        input <- readInput
        merges <- readMerges
        putStrLn $ show (example_2_6_1 input merges extensionsGPT2) <> "\n"
                <> "x: " <> show (example_2_6_2 input merges extensionsGPT2) <> "\n"
                <> "y:     " <> show (example_2_6_3 input merges extensionsGPT2) <> "\n" <> "\n"
                <> example_2_6_4 input merges extensionsGPT2 <> "\n" <> "\n"
                <> example_2_6_5 input merges extensionsGPT2 <> "\n" <> "\n"
                <> show (example_2_6_6 input merges extensionsGPT2) <> "\n" <> "\n"
                <> show (example_2_6_7 input merges extensionsGPT2) <> "\n" <> "\n"
                <> show (example_2_6_8 input merges extensionsGPT2) <> "\n" <> "\n"
      Example (2,7) -> do
        dictionary <- readDictionary
        embeddings <- readEmbeddings
        putStrLn $ show hyperParams <> "\n"
                <> show (example_2_7_1 hyperParams dictionary embeddings) <> "\n"
                <> show (example_2_7_2 hyperParams dictionary) <> "\n"
                <> BSC.unpack (BSL.toStrict $ example_2_7_3 $ example_2_7_2 hyperParams dictionary) <> "\n"
                -- Perform the lookup at the bottom of page 42, returning the tensor at the top of page 43.
                <> show [DAR.toList $ (\(NVec2F a) -> slice a (Any :. (v:: Int) :. All)) $ example_2_7_1 hyperParams dictionary embeddings| v <- [3]] <> "\n"
                -- Perform the lookup in the middle of page 43, returning the 4 tensors given there.
                <> show [DAR.toList $ (\(NVec2F a) -> slice a (Any :. (v:: Int) :. All)) $ example_2_7_1 hyperParams dictionary embeddings| v <- [2,3,5,1]] <> "\n"
      Example (2,8) -> do
        dictionary <- readDictionary
        input <- readInput
        merges <- readMerges
        let
          res_2_8_1@(NVec2I rawRes_2_8_1) = example_2_8_1 input merges extensionsGPT2
          (Z :. res_2_8_1_H :. res_2_8_1_W) = extent rawRes_2_8_1
          res_2_8_2 = example_2_8_2 (randomEmbeddings hyperParams dictionary) res_2_8_1
          (Z :. res_2_8_2_H :. res_2_8_2_W :. res_2_8_2_D) = extent ((\(NVec3F a) -> a) res_2_8_2)
          res_2_8_3 = example_2_8_3 ((\(HyperParams v _) -> v) hyperParams) 4
          (Z :. res_2_8_3_H :. res_2_8_3_W) = extent ((\(NVec2F a) -> a) res_2_8_3)
          res_2_8_4 = example_2_8_4 res_2_8_2 res_2_8_3
          (Z :. res_2_8_4_H :. res_2_8_4_W :. res_2_8_4_D) = extent ((\(NVec3F a) -> a) res_2_8_4)
          in
          putStrLn $ show hyperParams <> "\n"
                  <> show res_2_8_1 <> "\nInputs shape: [" <> show res_2_8_1_H <> "," <> show res_2_8_1_W <> "]\n"
                  <> show res_2_8_2 <> "\nInputs shape: [" <> show res_2_8_2_H <> "," <> show res_2_8_2_W <> "," <> show res_2_8_2_D <> "]\n"
                  <> show res_2_8_3 <> "\nInputs shape: [" <> show res_2_8_3_H <> "," <> show res_2_8_3_W <> "]\n"
                  <> show res_2_8_4 <> "\nInputs shape: [" <> show res_2_8_4_H <> "," <> show res_2_8_4_W <> "," <> show res_2_8_4_D <> "]\n"
      Example (3,3) -> do
        dictionary <- readDictionary
        embeddings <- readEmbeddings
        putStrLn $ show hyperParams <> "\n"
                <> show (example_3_3_1 hyperParams dictionary embeddings) <> "\n"
                <> show (example_3_3_2 hyperParams dictionary embeddings) <> "\n"
                <> show (example_3_3_3 hyperParams dictionary embeddings) <> "\n"
                <> show (sumS ((\(NVec1F a) -> a) $ example_3_3_3 hyperParams dictionary embeddings)) <> "\n"
                <> show (example_3_3_4 hyperParams dictionary embeddings) <> "\n"
                <> show (example_3_3_5 hyperParams dictionary embeddings) <> "\n"
                <> show (example_3_3_6 hyperParams dictionary embeddings) <> "\n"
                <> show (example_3_3_7 hyperParams dictionary embeddings) <> "\n"
                <> show (example_3_3_8 hyperParams dictionary embeddings) <> "\n"
      Example (3,4) -> do
        dictionary <- readDictionary
        embeddings <- readEmbeddings
        attentionWeights <- readWeights
        putStrLn $ show (example_3_4_1 hyperParams dictionary embeddings attentionWeights) <> "\n"
                <> "keys.shape: "
                <> show ((\(NVec2F a) -> extent a) $ example_3_4_2 hyperParams dictionary embeddings attentionWeights) <> "\n"
                <> "keys.shape: "
                <> show ((\(NVec2F a) -> extent a) $ example_3_4_3 hyperParams dictionary embeddings attentionWeights) <> "\n"
                <> show (example_3_4_4 hyperParams dictionary embeddings attentionWeights) <> "\n"
                <> show (example_3_4_5 hyperParams dictionary embeddings attentionWeights) <> "\n"
                <> show (example_3_4_6 hyperParams dictionary embeddings attentionWeights) <> "\n"
                <> show (example_3_4_7 hyperParams dictionary embeddings attentionWeights) <> "\n"
                <> show (example_3_4_8 hyperParams dictionary embeddings attentionWeights) <> "\n"
                <> show attentionWeights <> "\n"
                <> show (example_3_4_10 hyperParams) <> "\n"
                <> show (example_3_4_11 hyperParams dictionary embeddings attentionWeights) <> "\n"
      Example (3,5) -> do
        dictionary <- readDictionary
        embeddings <- readEmbeddings
        attentionWeights <- readWeights
        dropoutMaps@(NVec3F rawDropoutMaps) <- readDropoutMaps
        outProjectionWeights <- readOutProjectionWeights
        let
          dropoutMap = NVec2F $ computeS $ slice rawDropoutMaps (Z :. (0::Int) :. All :. All)
        putStrLn $ show (example_3_5_1 hyperParams dictionary embeddings attentionWeights) <> "\n"
                <> show (example_3_5_2 embeddings) <> "\n"
                <> show (example_3_5_3 hyperParams dictionary embeddings attentionWeights) <> "\n"
                <> show (example_3_5_4 hyperParams dictionary embeddings attentionWeights) <> "\n"
                <> show dropoutMap <> "\n"
                <> show (example_3_5_5 embeddings 123) <> "\n"
                <> show (example_3_5_6 hyperParams dictionary embeddings attentionWeights dropoutMap) <> "\n"
                <> show (example_3_5_7 embeddings) <> "\n"
                <> show (example_3_5_8 hyperParams dictionary embeddings attentionWeights dropoutMap) <> "\n"
                <> show (example_3_5_9 hyperParams dictionary embeddings 123) <> "\n"
                <> show (example_3_5_10 hyperParams dictionary embeddings 123) <> "\n"
                <> show (example_3_5_11 hyperParams dictionary embeddings attentionWeights dropoutMaps) <> "\n"
                <> show (example_3_5_12 hyperParams dictionary embeddings attentionWeights dropoutMaps) <> "\n"
                <> show (example_3_5_13 hyperParams dictionary embeddings attentionWeights dropoutMaps) <> "\n"
                <> show (example_3_5_14 hyperParams dictionary embeddings attentionWeights dropoutMaps outProjectionWeights) <> "\n"
      Example (a,b) -> error $ "unknown listing: " <> show a <> "." <> show b <> "\n"
  where
    example_2_3_String, example_2_4_String, example_2_5_String :: [Char]
    example_2_3_String = "\"It's the last he painted, you know,\" Mrs. Gisburn said with pardonable pride."
    example_2_4_String = "Hello, do you like tea?" <> " <|endoftext|> " <> "In the sunlit terraces of the palace."
    example_2_5_String = "Hello, do you like tea?" <> " <|endoftext|> " <> "In the sunlit terraces of someunknownPlace."
    -- our tokenizer was not handling <|endoftext|> properly. it was recognising it as 1279,91,437,1659,5239,91,29, rather than 220,50256.
    example_2_5_Seq :: Seq
    example_2_5_Seq = [15496,11,466,345,588,8887,30,
                       220, 50256 -- " <|endoftext|>~
                      ,554,262,4252,18250,8812,2114,286,617,34680,27271,13]

-- | The entry point. Use the option parser then run the trainer.
main :: IO ()
main = execParser opts >>= run
    where
      opts = info (helper <*> trainOpts)
             ( fullDesc
               <> progDesc "TrainLLM: An implementation of the examples from 'Build a Large Language Model (From Scratch)'."
               <> header "trainLLM - LLM Training in haskell."
             )
