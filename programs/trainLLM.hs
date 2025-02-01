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

import Prelude (Bool(True, False), Char, Float, Int, IO, Maybe(Just, Nothing), Show, String, (<$>), (<*>), (>>=), (<>), (&&), (/=), (==), (<), (>), (.), ($), (*), (+), (-), concat, error, exp, fromIntegral, getContents, length, mempty, not, otherwise, pure, putStrLn, return, show, take, zip)

import qualified Prelude as PL (readFile)

import Data.Aeson (Value(Array, Number), FromJSON(parseJSON), ToJSON(toJSON), (.=), eitherDecode, object, withObject)

import qualified Data.Aeson as A (encode)

import Data.Aeson.Key (Key, toText)

import qualified Data.Aeson.Key as AK (fromString)

import BPE.Base (Id, Merges, Seq, Vocab, mergesToVocab)

import qualified BPE.Regex as BPER (decode, encode)

import BPE.Regex (gpt2pattern)

import qualified Data.Aeson.KeyMap as DAKM (toList)

import Data.Array.Repa (U, Z(Z), (*^), (/^), (+^), computeS, extend, extent, fromListUnboxed, map, slice, sumS, transpose)

import qualified Data.Array.Repa as DAR (Array, toList)

import Data.Array.Repa.Index (DIM1, DIM2, DIM3, (:.)((:.)))

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

import Data.List ((++), drop, elem, foldr1, head, unfoldr, sort)

import Data.List.Extra (replace)

import Data.List.Split (chunksOf, dropBlanks, oneOf, onSublist, split, splitOneOf)

import Data.List.Unique (sortUniq)

import Data.Maybe (fromMaybe)

import Data.Scientific (toBoundedInteger, toRealFloat)

import Data.Text.Encoding (encodeUtf8)

import qualified Data.Vector as DV (toList)

import Data.Word (Word8)

import Options.Applicative (Parser, ReadM, auto, execParser, fullDesc, header, help, helper, info, long, metavar, option, optional, progDesc, short, str, strOption, switch)

import System.Random (StdGen, mkStdGen, uniformR)

-- | A type for encoding an example number.
data Example =
  Example (Int, Int)

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
    findExample (_:'.':[]) = error $ "no listing number.\n" <> exampleDescription
    findExample (_:_:[]) = error $ "two digit chapter number.\n" <> exampleDescription
    findExample (a:'.':b:[])
      | isHexDigit a && isDigit b = (digitToInt a, digitToInt b)
      | isHexDigit a = error $ "unable to read listing number.\n" <> exampleDescription
      | isDigit b = error $ "unable to read chapter number.\n" <> exampleDescription
      | otherwise = error $ "unable to read chapter or listing number.\n" <> exampleDescription
    findExample (a:'.':b:c:[])
      | isHexDigit a && isDigit b && isDigit c = (digitToInt a, (digitToInt b)*10+digitToInt c)
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
    , tokenEmbeddingsOpt :: Maybe String
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
  <*> (
  option exampleReader
    (    long "example"
      <> short 'e'
      <> metavar "EXAMPLE"
      <> help "which example to run"
    )
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
  strOption
    (    short 't'
      <> long "tokenEmbeddings"
      <> metavar "TOKENEMBEDDINGS"
      <> help "load a JSON formatted list of token embeddings"
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
data Bacov = Bacov (InsOrdHashMap BSS.ByteString Int)

-- | Parse a vocabulary file.
instance FromJSON Bacov where
  parseJSON = withObject "Bacov" (\v -> pure $ findBacov $ DAKM.toList v)
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
  vocabOfText s = vocabFromText s
  stringFromTokens v t = getStringFromTokens v t
  tokensFromString v s = getTokensFromString v Nothing s

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
    separateDoubleDash = concat $ split (dropBlanks $ onSublist "--") <$> separatePunctuation
    separatePunctuation :: [[Char]]
    separatePunctuation = concat $ split (oneOf ",.:;?_!\"()'") <$> words
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
                                            Nothing -> error $ "cannot find a token for \"" <> (BSU.toString s) <> "\"\n"
    -- the vocabulary backwards: a mapping from value to token.
    bacov :: InsOrdHashMap BSS.ByteString Int
    bacov = DHSI.fromList (swap <$> DHSI.toList rawVocab)
    swap (a,b) = (b,a)

-- | Use a vocabulary to reconstruct a string from a list of tokens.
getStringFromTokens :: Vocab -> [Int] -> [Char]
getStringFromTokens rawVocab tokens = maybeIntersperse ' ' $ findStringOfToken <$> tokens
  where
    maybeIntersperse :: Char -> [BSS.ByteString] -> [Char]
    maybeIntersperse _ [] = []
    maybeIntersperse x xs = foldr1 maybeIntersperse' (BSU.toString <$> xs)
      where
        maybeIntersperse' :: [Char] -> [Char] -> [Char]
        maybeIntersperse' a b = case (head b) `elem` (",.?!\"()'" :: [Char]) of
                                  False -> a ++ x:b
                                  True -> a ++ b
    findStringOfToken t = case lookup t rawVocab of
                            Just s -> s
                            Nothing -> error $ "cannot find a string for token" <> (show t) <> "\n"

-- | Count the number of characters in the input file.
-- Corresponds to page 22, listing 2.1.
example_2_1 :: [Char] -> [Char]
example_2_1 text = "Total number of character: " <> show (length text) <> "\n" <> take 99 text

-- | Construct a Vocab for the first 51 tokens of the input file.
-- Corresponds to page 25, listing 2.2.
example_2_2 :: [Char] -> Vocab
example_2_2 text = DHSI.fromList $ take 51 $ DHSI.toRevList $ vocabOfText text

-- | For example 2.3, they use it twice, we just implement this as two functions: one for encoding, and one for decoding.
-- This is the encoding one.
-- Corresponds to page 27, listing 2.3, encode().
example_2_3_1 :: [Char] -> [Char] -> [Int]
example_2_3_1 text string = tokensFromString (vocabOfText text) string

-- | For example 2.3, they use it twice, we just implement this as two functions: one for encoding, and one for decoding.
-- This is the decoding one.
-- Corresponds to page 27, listing 2.3, decode().
example_2_3_2 :: [Char] -> [Int] -> [Char]
example_2_3_2 text tokens = stringFromTokens (vocabOfText text) tokens

-- | Example 2.4 has several sub examples. This one prints the last 5 tokens in our extended vocabulary.
-- Corresponds to page 30, bottom half of page.
example_2_4_1 :: [Char] -> Vocab
example_2_4_1 text = DHSI.fromList $ drop (length vocab - 5) $ sort $ DHSI.toList vocab
  where
    vocab = extendVocabGPT2Unk $ vocabOfText text

-- | Example 2.4 has several sub examples. This one gives us the tokens at the top of page 32.
example_2_4_2 :: [Char] -> [Char] -> [Int]
example_2_4_2 text string = getTokensFromString vocab (Just $ length vocab - 1) string
  where
    vocab = extendVocabGPT2Unk $ vocabOfText text

-- | Example 2.4 has several sub examples. This one gives us the reconstituted string on page 32.
example_2_4_3 :: [Char] -> [Int] -> [Char]
example_2_4_3 text tokens = stringFromTokens vocab tokens
  where
    vocab = extendVocabGPT2Unk $ vocabOfText text

-- | Tokenize GPT2 style.
-- When given the GPT2 vocabulary and merges files, along with the string:
-- "Hello, do you like tea?" <> " <|endoftext|> " <> "In the sunlit terraces of someunknownPlace."
-- Produces the sequence of token IDs on page 33.
example_2_5_1 :: [Char] -> BSL.ByteString -> BSL.ByteString -> Seq
example_2_5_1 text merges dictionary
  | mergeDictionary == jsonDictionary = encodeExtensionsGPT2 $ BPER.encode initSeqGPT2 (mergesFromTXT merges) gpt2pattern mempty (BSU.fromString text)
  | otherwise = error $ "Dictionaries not identical:\nTEXT: " <> (show $ take 100 $ drop 50200 $ sort $ DHSI.toList $ mergeDictionary) <> "\n"
                     <> "JSON: " <> (show $ take 100 $ drop 50200 $ sort $ DHSI.toList $ jsonDictionary) <> "\n"
  where
    -- a dictionary from a merge file.
    mergeDictionary = extendVocabGPT2 $ mergesToVocab (mergesFromTXT merges) initVocabGPT2
    -- a dictionary from a dictionary file.
    jsonDictionary = dictionaryFromJSON dictionary

-- | De-Tokenize GPT2 style.
-- When given the GPT2 vocabulary and merges files, along with the sequence:
-- [15496,11,466,345,588,8887,30,220,50256,554,262,4252,18250,8812,2114,286,617,34680,27271,13]
-- Produces the same output as: "Hello, do you like tea?" <> " <|endoftext|> " <> "In the sunlit terraces of someunknownPlace."
example_2_5_2 :: Seq -> BSL.ByteString -> BSL.ByteString -> BSS.ByteString
example_2_5_2 seq merges dictionary
  | mergeDictionary == jsonDictionary = respaceGPT2 $ BPER.decode jsonDictionary mempty seq
  | otherwise = error $ "Dictionaries not identical:\nTEXT: " <> (show $ take 100 $ drop 50200 $ sort $ DHSI.toList $ mergeDictionary) <> "\n"
                     <> "JSON: " <> (show $ take 100 $ drop 50200 $ sort $ DHSI.toList $ jsonDictionary) <> "\n"
  where
    -- a dictionary from a merge file.
    mergeDictionary = extendVocabGPT2 $ mergesToVocab (mergesFromTXT merges) initVocabGPT2
    -- a dictionary from a dictionary file.
    jsonDictionary = dictionaryFromJSON dictionary

-- | Count the number of tokens in the given text.
-- Implements page 35.
example_2_6_1 :: [Char] -> BSL.ByteString -> Extensions -> Int
example_2_6_1 text merges extensions = length (encodeExtensions extensions $ BPER.encode initSeqGPT2 (mergesFromTXT merges) gpt2pattern mempty (BSU.fromString text))

-- | Return tokens 51, 52, 53, and 54.
-- Implements the 'x' result of the top of page 36.
example_2_6_2 :: [Char] -> BSL.ByteString -> Extensions -> Seq
example_2_6_2 text merges extensions = take 4 $ drop 50 (encodeExtensions extensions $ BPER.encode initSeqGPT2 (mergesFromTXT merges) gpt2pattern mempty (BSU.fromString text))

-- | Return tokens 52, 53, 54, and 55.
-- Implements the 'y' result of the top of page 36.
example_2_6_3 :: [Char] -> BSL.ByteString -> Extensions -> Seq
example_2_6_3 text merges extensions = take 4 $ drop 51 (encodeExtensions extensions $ BPER.encode initSeqGPT2 (mergesFromTXT merges) gpt2pattern mempty (BSU.fromString text))

-- | Produce an example of a next word prediction task training dataset.
-- Implements the output with IDs and arrows in it, in the middle of page 36.
example_2_6_4 :: [Char] -> BSL.ByteString -> Extensions -> [Char]
example_2_6_4 text merges extensions =  rotateShow $ take 5 $ drop 50 (encodeExtensions extensions $ BPER.encode initSeqGPT2 (mergesFromTXT merges) gpt2pattern mempty (BSU.fromString text))
  where
    rotateShow [] = error "too few."
    rotateShow [_] = error "too few."
    rotateShow (xs) = rotateShow' [] xs
    rotateShow' _ [] = ""
    rotateShow' _ [_] = ""
    rotateShow' [] [x,y] = show [x] <> " ----> " <> show y <> "\n"
    rotateShow' [] (x:y:xs) = show [x] <> " ----> " <> show y <> "\n" <> rotateShow' [x] (y:xs)
    rotateShow' a [x,y] = show (a <> [x]) <> " ----> " <> show y <> "\n"
    rotateShow' a (x:y:xs) = show (a <> [x]) <> " ----> " <> show y <> "\n" <> rotateShow' (a <> [x]) (y:xs)

-- | Produce a human readable example of a next word prediction task training dataset.
-- Implements the output with words and arrows in it, at the bottom of page 36.
example_2_6_5 :: [Char] -> BSL.ByteString -> Extensions -> [Char]
example_2_6_5 text merges extensions = BSC.unpack $ rotateShow $ take 5 $ drop 50 (encodeExtensions extensions $ BPER.encode initSeqGPT2 (mergesFromTXT merges) gpt2pattern mempty (BSU.fromString text))
  where
    rotateShow [] = error "too few."
    rotateShow [_] = error "too few."
    rotateShow (xs) = rotateShow' [] xs
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
    mergeDictionary = extendVocabGPT2 $ mergesToVocab (mergesFromTXT merges) initVocabGPT2

-- | A 2D vector of Ints
data NVec2I = NVec2I (DAR.Array U DIM2 Int)
  deriving Show

-- | A 3D vector of Ints.
data NVec3I = NVec3I (DAR.Array U DIM3 Int)
  deriving Show

-- | Produce the first four tokens, then the second four tokens, where the second consists of the last three of the first, and the 5th token.
-- Produces the same output as the 'first batch' output on page 39.
example_2_6_6 :: [Char] -> BSL.ByteString -> Extensions -> NVec2I
example_2_6_6 text merges extensions = NVec2I $ fromListUnboxed (Z :. 2 :. 4) $ concat $ [
                                        take 4 $ encodeExtensions extensions $ BPER.encode initSeqGPT2 (mergesFromTXT merges) gpt2pattern mempty (BSU.fromString text)
                                       ,take 4 $ drop 1 $ encodeExtensions extensions $ BPER.encode initSeqGPT2 (mergesFromTXT merges) gpt2pattern mempty (BSU.fromString text)
                                       ]

-- | Produce the second four tokens, then the third four tokens, where the second consists of the last three of the first, and the 5th token.
-- Produces the same output as the 'second batch' output on page 39.
example_2_6_7 :: [Char] -> BSL.ByteString -> Extensions -> NVec2I
example_2_6_7 text merges extensions = NVec2I $ fromListUnboxed (Z :. 2 :. 4) $ concat $ [
                                        take 4 $ drop 1 $ encodeExtensions extensions $ BPER.encode initSeqGPT2 (mergesFromTXT merges) gpt2pattern mempty (BSU.fromString text)
                                       ,take 4 $ drop 1 $ drop 1 $ encodeExtensions extensions $ BPER.encode initSeqGPT2 (mergesFromTXT merges) gpt2pattern mempty (BSU.fromString text)
                                       ]

-- | Produce a more realistic sample, in 3 dimensions.
-- Produces the two tensors spanning pages 40 and 41.
example_2_6_8 :: [Char] -> BSL.ByteString -> Extensions -> NVec3I
example_2_6_8 text merges extensions = NVec3I $ fromListUnboxed (Z :. 2 :. 8 :. 4) $ concat $ [
                                        take (8*4) $ encodeExtensions extensions $ BPER.encode initSeqGPT2 (mergesFromTXT merges) gpt2pattern mempty (BSU.fromString text)
                                       ,take (8*4) $ drop 1 $ encodeExtensions extensions $ BPER.encode initSeqGPT2 (mergesFromTXT merges) gpt2pattern mempty (BSU.fromString text)
                                       ]

-- | Our Hyperparameters. The configuration settings of the model we are working with.
data HyperParams =
  HyperParams
    {
      embeddingDim :: Int -- How many dimensions our embeddings will be.
    }
  deriving Show

-- A two dimensional vector of Floats.
data NVec2F = NVec2F (DAR.Array U DIM2 Float)
  deriving Show

-- A three dimensional vector of floats.
data NVec3F = NVec3F (DAR.Array U DIM3 Float)
  deriving Show

-- We're getting a bit creative in this section, because there is no way we can get our haskell random seed to line up with pytorch.
-- Instead, we're performing serialization / deserialization of the values from the book, constructing and displaying our own random sets, and performing operations on each.

-- | Read from a JSON file and display a set of token embeddings.
-- When given 3d6-token_embeddings.json and 6_token-vocab.json , produces the embedding layer weight matrix on page 42.
example_2_7_1 :: HyperParams -> BSL.ByteString -> BSL.ByteString -> NVec2F
example_2_7_1 (HyperParams embeddingDimensions) dictionary tokenEmbeddingsByteStream
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show (foundEmbeddingsDimensions) <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show (foundEmbeddingsCount) <> "\n"
  | otherwise = tokenEmbeddings
  where
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings
    tokenEmbeddings@(NVec2F rawTokenEmbeddings) = embeddingsFromJSON tokenEmbeddingsByteStream
    -- a dictionary from a dictionary file.
    jsonDictionary = dictionaryFromJSON dictionary

-- | Generate a random set of embeddings.
example_2_7_2 :: HyperParams -> BSL.ByteString -> NVec2F
example_2_7_2 hyperParams dictionary = randomEmbeddings hyperParams jsonDictionary
  where
    -- a dictionary from a dictionary file.
    jsonDictionary = dictionaryFromJSON dictionary

-- | Generate a set of embeddings as JSON. So that we can serialize our set, for tracking purposes.
example_2_7_3 :: NVec2F -> BSL.ByteString
example_2_7_3 embeddings = embeddingsToJSON embeddings

-- | Get the first 32 (8*4) tokens.
-- when given the-verdict.txt, gpt2-vocab.json, gpt2-merges.txt and 3d6-token_embeddings.json, produces the tensor in the middle of page 46.
example_2_8_1 :: [Char] -> BSL.ByteString -> Extensions -> NVec2I
example_2_8_1 text merges extensions = NVec2I $ fromListUnboxed (Z :. 8 :. 4) $ take (8*4) $ encodeExtensions extensions $ BPER.encode initSeqGPT2 (mergesFromTXT merges) gpt2pattern mempty (BSU.fromString text)

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
    embedSeq seq = NVec2F $ fromListUnboxed (Z :. (length seq) :. foundEmbeddingsDimensions) $ concat [DAR.toList $ slice rawEmbeddings (Any :. (v::Int) :. All) | v <- seq]
    (Z :. _ :. foundEmbeddingsDimensions) = extent rawEmbeddings

-- | Generate a set of positional embeddings.
example_2_8_3 :: Int -> Int -> NVec2F
example_2_8_3 dimensions positions = NVec2F $ fromListUnboxed (Z :. positions :. dimensions) $ take (positions * dimensions) $ [0.0 :: Float,1.0..]

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
example_3_3_1 :: HyperParams -> BSL.ByteString -> BSL.ByteString -> NVec2F
example_3_3_1 = example_2_7_1

-- A one dimensional vector of Floats.
data NVec1F = NVec1F (DAR.Array U DIM1 Float)
  deriving Show

-- | Read a set of token embeddings from a JSON file, and calculate a set of attention results of the second token, vs the rest of the tokens.
-- When given 3d6-token_embeddings-3_3_1.json and 6_token-vocab.json , produces the attention values on page 58.
example_3_3_2 :: HyperParams -> BSL.ByteString -> BSL.ByteString -> NVec1F
example_3_3_2 (HyperParams embeddingDimensions) dictionary tokenEmbeddingsByteStream
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show (foundEmbeddingsDimensions) <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show (foundEmbeddingsCount) <> "\n"
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
      | otherwise = NVec1F $ sumS $ rawTokenEmbeddings *^ (extend (Z :. (foundEmbeddingsCount) :. All) target)
      where
        target = slice rawTokenEmbeddings (Any :. (itemNo :: Int) :. All)
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings
    (NVec2F rawTokenEmbeddings) = embeddingsFromJSON tokenEmbeddingsByteStream
    -- a dictionary from a dictionary file.
    jsonDictionary = dictionaryFromJSON dictionary

-- | Read a set of token embeddings from a JSON file, and calculate a set of attention results of the second token, vs the rest of the tokens.
-- When given 3d6-token_embeddings-3_3_1.json and 6_token-vocab.json , produces the attention values on page 59.
example_3_3_3 :: HyperParams -> BSL.ByteString -> BSL.ByteString -> NVec1F
example_3_3_3 (HyperParams embeddingDimensions) dictionary tokenEmbeddingsByteStream
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show (foundEmbeddingsDimensions) <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show (foundEmbeddingsCount) <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | softmax attention against the second token.
  | otherwise = findDotAttn 1
  where
    -- | For a set of token embeddings, find the dot product of a given token when compared to every other token in the set. Normalize the output.
    findDotAttn :: Int -> NVec1F
    findDotAttn itemNo
      | foundEmbeddingsCount < itemNo = error $ "Too few items.\n"
                                             <> "comparison token index: " <> show itemNo <> "\n"
                                             <> "found tokens: " <> show foundEmbeddingsCount <> "\n"
      | otherwise = normVec $ NVec1F $ sumS $ rawTokenEmbeddings *^ (extend (Z :. (foundEmbeddingsCount) :. All) target)
      where
        normVec :: NVec1F -> NVec1F
        normVec (NVec1F inVec) = NVec1F $ computeS $ inVec /^ (extend (Z :. foundItems) $ sumS inVec)
          where
            (Z :. foundItems) = extent inVec
        target = slice rawTokenEmbeddings (Any :. (itemNo :: Int) :. All)
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings
    (NVec2F rawTokenEmbeddings) = embeddingsFromJSON tokenEmbeddingsByteStream
    -- a dictionary from a dictionary file.
    jsonDictionary = dictionaryFromJSON dictionary

-- | For a set of token embeddings, find the dot product of a given token when compared to every other token in the set. Normalize the output using softmax.
findAttn :: NVec2F -> Int -> NVec1F
findAttn (NVec2F rawTokenEmbeddings) itemNo
  | foundEmbeddingsCount < itemNo = error $ "Too few items.\n"
                                         <> "comparison token index: " <> show itemNo <> "\n"
                                         <> "found tokens: " <> show foundEmbeddingsCount <> "\n"
  | otherwise = softMax $ NVec1F $ sumS $ rawTokenEmbeddings *^ (extend (Z :. (foundEmbeddingsCount) :. All) target)
  where
    softMax :: NVec1F -> NVec1F
    softMax (NVec1F inVec) = NVec1F $ computeS $ (map exp inVec) /^ (extend (Z :. foundItems) $ sumS $ map exp inVec)
      where
        (Z :. foundItems) = extent inVec
    target = slice rawTokenEmbeddings (Any :. (itemNo :: Int) :. All)
    (Z :. foundEmbeddingsCount :. _) = extent rawTokenEmbeddings

-- | Read a set of token embeddings from a JSON file, and calculate a set of attention results of the second token, vs the rest of the tokens.
-- When given 3d6-token_embeddings-3_3_1.json and 6_token-vocab.json , produces the attention values on page 60.
example_3_3_4 :: HyperParams -> BSL.ByteString -> BSL.ByteString -> NVec1F
example_3_3_4 (HyperParams embeddingDimensions) dictionary tokenEmbeddingsByteStream
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show (foundEmbeddingsDimensions) <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show (foundEmbeddingsCount) <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | softmax attention against the second token.
  | otherwise = findAttn tokenEmbeddings 1
  where
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings
    tokenEmbeddings@(NVec2F rawTokenEmbeddings) = embeddingsFromJSON tokenEmbeddingsByteStream
    -- a dictionary from a dictionary file.
    jsonDictionary = dictionaryFromJSON dictionary

-- When given 3d6-token_embeddings-3_3_1.json and 6_token-vocab.json , produces the context vector at the bottom of page 60.
example_3_3_5 :: HyperParams -> BSL.ByteString -> BSL.ByteString -> NVec1F
example_3_3_5 (HyperParams embeddingDimensions) dictionary tokenEmbeddingsByteStream
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show (foundEmbeddingsDimensions) <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show (foundEmbeddingsCount) <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | softmax attention against the second token.
  | otherwise = NVec1F $ sumS $ transpose $ rawTokenEmbeddings *^ extend (Z :. All :. (foundEmbeddingsDimensions :: Int)) rawFoundAttention
  where
    (NVec1F rawFoundAttention) = findAttn tokenEmbeddings 1
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings
    tokenEmbeddings@(NVec2F rawTokenEmbeddings) = embeddingsFromJSON tokenEmbeddingsByteStream
    -- a dictionary from a dictionary file.
    jsonDictionary = dictionaryFromJSON dictionary

-- | Read a set of token embeddings from a JSON file, and calculate a set of un-normalized attention results.
-- When given 3d6-token_embeddings-3_3_1.json and 6_token-vocab.json , produces the set of attention values on page 62.
example_3_3_6 :: HyperParams -> BSL.ByteString -> BSL.ByteString -> NVec2F
example_3_3_6 (HyperParams embeddingDimensions) dictionary tokenEmbeddingsByteStream
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show (foundEmbeddingsDimensions) <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show (foundEmbeddingsCount) <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | softmax attention of each item against each item, including itsself.
  | otherwise = findMyAttns
  where
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings
    (NVec2F rawTokenEmbeddings) = embeddingsFromJSON tokenEmbeddingsByteStream
    -- a dictionary from a dictionary file.
    jsonDictionary = dictionaryFromJSON dictionary
    -- | For a set of token embeddings, find the dot product of each token when compared to every other token in the set, and itsself. Normalize the outputs using softmax.
    findMyAttns :: NVec2F
    findMyAttns = NVec2F $ sumS $ leftSide *^ rightSide
      where
        leftSide = extend (Z :. (foundEmbeddingsCount) :. All :. All) rawTokenEmbeddings
        rightSide = extend (Z :. All :. (foundEmbeddingsCount) :. All) rawTokenEmbeddings

-- | Read a set of token embeddings from a JSON file, and calculate a set of attention results of the second token, vs the rest of the tokens.
-- When given 3d6-token_embeddings-3_3_1.json and 6_token-vocab.json , produces the attention values on page 63.
example_3_3_7 :: HyperParams -> BSL.ByteString -> BSL.ByteString -> NVec2F
example_3_3_7 (HyperParams embeddingDimensions) dictionary tokenEmbeddingsByteStream
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show (foundEmbeddingsDimensions) <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show (foundEmbeddingsCount) <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | softmax attentions.
  | otherwise = findAttns tokenEmbeddings
  where
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings
    tokenEmbeddings@(NVec2F rawTokenEmbeddings) = embeddingsFromJSON tokenEmbeddingsByteStream
    -- a dictionary from a dictionary file.
    jsonDictionary = dictionaryFromJSON dictionary

-- | Read a set of token embeddings from a JSON file, and calculate a set of attention results of the second token, vs the rest of the tokens.
-- When given 3d6-token_embeddings-3_3_1.json and 6_token-vocab.json , produces the context vectors on page 63.
example_3_3_8 :: HyperParams -> BSL.ByteString -> BSL.ByteString -> NVec2F
example_3_3_8 (HyperParams embeddingDimensions) dictionary tokenEmbeddingsByteStream
  -- Check our expected embedding dimensions, compared to the found one.
  | embeddingDimensions /= foundEmbeddingsDimensions = error $ "mismatch in count of dimensions in first token, and embedding dimensions\nDimensions expected(via HyperParams): " <> show embeddingDimensions <> "\nFound dimensions: " <> show (foundEmbeddingsDimensions) <> "\n"
  -- Check our expected embedding count, compared to the found one.
  | length jsonDictionary /= foundEmbeddingsCount = error $ "mismatch in count of embeddings, versus number of items in dictionary.\nDictionary items: " <> show (length jsonDictionary) <> "\nEmbeddings: " <> show (foundEmbeddingsCount) <> "\n"
  | foundEmbeddingsCount < 2 = error "There is no second token in our stream of embedded tokens.\n"
  -- Find the dot product | softmax attention against the second token.
  | otherwise = NVec2F $ sumS $ transpose $ extend (Z :. foundEmbeddingsCount :. All :. All) rawTokenEmbeddings *^ extend (Z :. All :. All:. (foundEmbeddingsDimensions :: Int)) rawFoundAttention
  where
    (NVec2F rawFoundAttention) = findAttns tokenEmbeddings
    (Z :. foundEmbeddingsCount :. foundEmbeddingsDimensions) = extent rawTokenEmbeddings
    tokenEmbeddings@(NVec2F rawTokenEmbeddings) = embeddingsFromJSON tokenEmbeddingsByteStream
    -- a dictionary from a dictionary file.
    jsonDictionary = dictionaryFromJSON dictionary

-- | For a set of token embeddings, find the dot product of each token when compared to every other token in the set, and itsself. Normalize the outputs using softmax.
findAttns :: NVec2F -> NVec2F
findAttns (NVec2F rawTokenEmbeddings) = softMax $ NVec2F $ sumS $ leftSide *^ rightSide
  where
    leftSide = extend (Z :. (foundEmbeddingsCount) :. All :. All) rawTokenEmbeddings
    rightSide = extend (Z :. All :. (foundEmbeddingsCount) :. All) rawTokenEmbeddings
    (Z :. foundEmbeddingsCount :. _) = extent rawTokenEmbeddings
    softMax :: NVec2F -> NVec2F
    softMax (NVec2F inRawVec) = NVec2F $ computeS $ (map exp inRawVec) /^ (extend (Z :. All :. foundItems) $ sumS $ map exp inRawVec)
      where
        (Z :. foundItems :. _) = extent inRawVec

-- | Generate a random set of embeddings.
randomEmbeddings :: HyperParams -> Vocab -> NVec2F
randomEmbeddings (HyperParams embeddingDimensions) vocab
  | otherwise = NVec2F $ fromListUnboxed (Z :. vocabLength :. embeddingDimensions) $ concat [mkRandomEmbedding (mkStdGen v) | v <- [0,1..vocabLength-1]]
    where
      mkRandomEmbedding :: StdGen -> [Float]
      mkRandomEmbedding = take embeddingDimensions . unfoldr (Just . uniformR (-3,3))
      vocabLength = length vocab

-- | A type for Embeddings, as they come out of the JSON file.
data Embeddings = Embeddings (InsOrdHashMap BSS.ByteString [Float])
  deriving Show

-- | Our parser for an embeddings file.
instance FromJSON Embeddings where
  parseJSON = withObject "Embeddings" (\v -> pure $ findEmbeddings $ DAKM.toList v)
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
          numbersFromValue (Array (vs)) = (\(Number a) -> toRealFloat a) <$> DV.toList vs
          numbersFromValue a = error $ "failed to parse " <> show a <> " as an Array.\n"

-- | Our serializer, to produce JSON from a set of Embeddings.
instance ToJSON Embeddings where
  toJSON (Embeddings rawEmbeddings) = toJSON $ object $ (\(a,b) -> AK.fromString (BSC.unpack a) .= b) <$> DHSI.toList rawEmbeddings

-- | Fill a ByteString with the JSON formatted form of the given set of embeddings.
embeddingsToJSON :: NVec2F -> BSL.ByteString
embeddingsToJSON nVec2f
  | otherwise = A.encode $ embeddingsFromTensor nVec2f
    where
      embeddingsFromTensor (NVec2F rawEmbeddings) = Embeddings $ DHSI.fromList $ zip [BSL.toStrict $ toByteString v | v <- [0,1 .. length sequences-1]] sequences
        where
          sequences = [DAR.toList rawEmbeddings]

-- | Read a set of embeddings from a JSON formatted map of number to list of N sets of D floats. where N is your vocabulary length, and D is your embeddings dimensions.
embeddingsFromJSON :: BSL.ByteString -> NVec2F
embeddingsFromJSON json = NVec2F $ fromListUnboxed (Z :. (size rawEmbeddings) :. firstEmbeddingLength) embeddingsList
  where
    (Embeddings rawEmbeddings) = case eitherDecode json :: Either String Embeddings of
                                   Left err -> error $ "parse error when reading embeddings:\n" <> err <> "\n" <> show json <> "\n"
                                   Right d -> d
    -- By performing lookup from 0-size rawEmbeddings, we ensure a consistent space, with no gaps.
    embeddingsList = concat $ (\a -> fromMaybe (error $ "could not lookup" <> show a <> "\n") $ lookup (BSL.toStrict $ toByteString a) rawEmbeddings) <$> [0,1..size rawEmbeddings-1]
    firstEmbeddingLength = length $ fromMaybe (error $ "failed to lookup first embedding (0)." ) $ lookup "0" rawEmbeddings

-- | Read a dictionary from a JSON formatted map.
dictionaryFromJSON :: BSL.ByteString -> Vocab
dictionaryFromJSON json = case eitherDecode json :: Either String Bacov of
                            Left err -> error $ "parse error when reading dictionary:\n" <> err <> "\n" <> show json <> "\n"
                            Right d -> flip d
                              where
                                flip ::  Bacov -> Vocab
                                flip vk = DHSI.fromList (swap <$> (DHSI.toList $ (\(Bacov v) -> v) vk))
                                  where
                                    swap (a,b) = (b,a)

-- | The default starting vocabulary, taken from the first 256 tokens of gpt2.
initVocabGPT2 :: Vocab
initVocabGPT2 = flip defaultBacov
  where
    flip ::  Bacov -> Vocab
    flip vk = DHSI.fromList (swap <$> (DHSI.toList $ (\(Bacov v) -> v) vk))
      where
        swap (a,b) = (b,a)

-- | The default starting vocabulary, taken from the first 256 tokens of gpt2.
-- An initial (reverse) vocabulary, consisting of....
defaultBacov :: Bacov
defaultBacov = Bacov $ DHSI.fromList $ (zip (BSS.singleton <$> [33, 34..]) [0,1..93]) -- the first 94 characters of ascii, after 33 control signals.
                                     <> (zip ((\a -> BSS.pack [194,128+a]) <$> [33,34..44]) [94, 95..105] )    -- UTF8 characters U+00A1-U+00AB
                                     <> (zip ((\a -> BSS.pack [194,128+a]) <$> [46,47..63]) [106, 107..123] )  -- UTF8 characters U+00AD-U+00BF
                                     <> (zip ((\a -> BSS.pack [195,128+a]) <$> [0,1..63]) [124, 125..187] )    -- UTF8 characters U+00C0-U+00FF
                                     <> (zip ((\a -> BSS.pack [196,128+a]) <$> [0,1..63]) [188, 189..251] )    -- UTF8 characters U+0100-U+013F
                                     <> (zip ((\a -> BSS.pack [197,128+a]) <$> [0,1..63]) [252, 253..255] )    -- UTF8 characters U+0140-U+017F

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
encodeExtensionsGPT2 inSeq = encodeExtensions extensionsGPT2 inSeq

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
mergesFromTXT text = DHSI.fromList (splitLineRecurse defaultBacov [] $ zip [256,257..] (Left <$> (drop 1 $ BSLU.lines mergeLines)))
  where
    -- Discard the first line of the file.
    (_,mergeLines) = BSLU.break (== '\n') text
    swap (a,b) = (b,a)
    splitLineRecurse :: Bacov -> [((Int, Int), Int)] -> [(Int, Either BSL.ByteString (Either BSS.ByteString Int, BSS.ByteString))] -> [((Int, Int), Int)]
    splitLineRecurse bacovIn mergesDone mergesTodo = case lefts res of
                                                       [] -> mergesDone <> rights res
                                                       _ -> splitLineRecurse newBacov (mergesDone <> rights res) (lefts res)
      where
        -- new entries, from this layer.
        foundBacov :: Bacov
        foundBacov = Bacov $ DHSI.fromList $ ((\(a,b) -> (BSL.toStrict (byteStringFrom a),b)) <$> rights res)
          where
           byteStringFrom :: (Int, Int) -> BSL.ByteString
           byteStringFrom (tok1, tok2) =  case lookup tok1 (flip bacovIn) of
                                            Nothing -> error $ show (lookup tok1 (flip bacovIn)) <> "\n" <> show tok2 <> "\n"
                                            (Just pre) -> case lookup tok2 (flip bacovIn) of
                                                            Nothing -> error $ show (lookup tok2 (flip bacovIn)) <> "\n" <> show tok2 <> "\n"
                                                            (Just post) -> BSL.fromStrict $ pre <> post

        newBacov :: Bacov
        newBacov = unionBacovs bacovIn foundBacov
          where
            unionBacovs :: Bacov -> Bacov -> Bacov
            unionBacovs (Bacov rawBacov1) (Bacov rawBacov2) = Bacov $ DHSI.union rawBacov1 rawBacov2
        flip ::  Bacov -> Vocab
        flip vk = DHSI.fromList (swap <$> (DHSI.toList $ (\(Bacov v) -> v) vk))
        res :: [Either (Int, Either BSL.ByteString (Either BSS.ByteString Int, BSS.ByteString)) ((Int, Int), Int)]
        res = splitLine bacovIn <$> mergesTodo
    splitLine :: Bacov -> (Int, Either BSL.ByteString (Either BSS.ByteString Int, BSS.ByteString)) -> Either (Int, Either BSL.ByteString (Either BSS.ByteString Int, BSS.ByteString)) ((Int, Int), Int)
    splitLine (Bacov rawBacov) tokenMap@(tokenNumber, (Right (Right token1, post))) =
      case lookup post rawBacov of
        Nothing -> Left tokenMap
        Just token2 -> Right ((token1, token2), tokenNumber)
    splitLine (Bacov rawBacov) tokenMap@(tokenNumber, (Right (Left pre, post))) =
      case lookup pre rawBacov of
        Nothing -> Left tokenMap
        Just token1 -> case lookup post rawBacov of
                         Nothing -> Left (tokenNumber, (Right (Right token1, post)))
                         Just token2 -> Right ((token1, token2), tokenNumber)
    splitLine (Bacov rawBacov) (tokenNumber, (Left string)) =
      case lookup pre rawBacov of
        Nothing -> Left (tokenNumber, (Right (Left pre, post)))
        Just token1 -> case lookup post rawBacov of
                         Nothing -> Left (tokenNumber, (Right (Right token1, post)))
                         Just token2 -> Right ((token1, token2), tokenNumber)
      where
        (pre, post) = (\(a,b) -> (BSL.toStrict a, BSL.toStrict $ BSLU.drop 1 b)) $ BSLU.break (== ' ') string

-- | select which example to run.
run :: TrainRootOpts -> IO ()
run rawArgs =
  let
    readInput :: IO String
    readInput = do
      input <- case inputFileOpt rawArgs of
                 Nothing -> getContents
                 Just inFile -> PL.readFile inFile
      return input
    readDictionary :: IO BSL.ByteString
    readDictionary = do
      input <- case dictionaryOpt rawArgs of
                 Nothing -> error "This example requires you to pass in your own dictionary, in JSON format."
                 Just inFile -> BSL.readFile inFile
      return input
    readMerges :: IO BSL.ByteString
    readMerges = do
      input <- case mergesOpt rawArgs of
                 Nothing -> error "This example requires you to pass in your own merges file, in text format."
                 Just inFile -> BSL.readFile inFile
      return input
    readEmbeddings :: IO BSL.ByteString
    readEmbeddings = do
      input <- case tokenEmbeddingsOpt rawArgs of
                 Nothing -> error "This example requires you to pass in a set of token embeddings, in JSON format."
                 Just inFile -> BSL.readFile inFile
      return input
    beVerbose = case verboseFlag rawArgs of
                  Nothing -> False
                  Just a -> a
    hyperParams :: HyperParams
    hyperParams = case embeddingDimensionsOpt rawArgs of
                    Nothing -> error "This example requires you to specify a number of dimensions each embedding recieves."
                    Just a -> if a < 1
                              then error "You must specify a positive number of dimensions for your embeddings."
                              else HyperParams a
  in
    case exampleOpt rawArgs of
      Example (2,1) -> do
        input <- readInput
        case beVerbose of
          True -> putStrLn $ "running listing 2.1 with input:\n<input>\n" <> input <> "</input>\n<result>" <> example_2_1 input <> "</result>\n"
          _ -> putStrLn $ example_2_1 input
      Example (2,2) -> do
        input <- readInput
        putStrLn $ show $ example_2_2 input
      Example (2,3) -> do
        input <- readInput
        putStrLn $ (show $ example_2_3_1 input example_2_3_String) <> "\n"
                 <> (show $ example_2_3_2 input $ example_2_3_1 input example_2_3_String) <> "\n"
      Example (2,4) -> do
        input <- readInput
        putStrLn $ (show $ example_2_4_1 input) <> "\n"
                 <> (show example_2_4_String) <> "\n"
                 <> (show $ example_2_4_2 input example_2_4_String) <> "\n"
                 <> (show $ example_2_4_3 input $ example_2_4_2 input example_2_4_String) <> "\n"
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
          res_2_8_2 = example_2_8_2 (randomEmbeddings hyperParams (dictionaryFromJSON dictionary)) (res_2_8_1)
          (Z :. res_2_8_2_H :. res_2_8_2_W :. res_2_8_2_D) = extent ((\(NVec3F a) -> a) res_2_8_2)
          res_2_8_3 = example_2_8_3 ((\(HyperParams v) -> v) hyperParams) 4
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
