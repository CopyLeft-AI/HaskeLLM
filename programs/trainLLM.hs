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

import Prelude (Bool(True, False), Char, Int, IO, Maybe(Just, Nothing), Show, String, (<$>), (<*>), (>>=), (<>), (&&), ($), (*), (+), (-), concat, error, getContents, length, lookup, not, otherwise, pure, putStrLn, return, show, take, zip)

import qualified Prelude as PL (readFile)

import Data.Aeson (Value(Array, Number), FromJSON(parseJSON), ToJSON(toJSON), (.=), decodeStrict, withObject, object)

import Data.Aeson.Key (Key, toText)

import qualified Data.Aeson.Key as DAK (fromText)

import Data.Aeson.KeyMap (toList)

import Data.ByteString (ByteString)

import qualified Data.ByteString.UTF8 as BSU (toString, fromString)

import Data.Char (digitToInt, isDigit, isHexDigit)

import Data.List ((++), drop, elem, foldr1, head)

import Data.List.Split (dropBlanks, oneOf, onSublist, split, splitOneOf)

import Data.List.Unique (sortUniq)

import Data.Scientific (toBoundedInteger)

import Data.Text.Encoding (decodeUtf8, encodeUtf8)

import Data.Vector (fromList)

import Options.Applicative (Parser, ReadM, execParser, fullDesc, header, help, helper, info, long, metavar, option, optional, progDesc, short, str, strOption, switch)

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

data TrainRootOpts =
  TrainRootOpts
    {
      inputFileOpt :: Maybe String
    , dictionaryOpt :: Maybe String
    , exampleOpt :: Example
    , verboseFlag :: Maybe Bool
    }

trainOpts :: Parser TrainRootOpts
trainOpts =
  TrainRootOpts
  <$> optional (
  strOption
    (    short 'i'
      <> long "infile"
      <> metavar "INPUTFILE"
      <> help "load an ASCII text file"
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
  <*> (
  option exampleReader
    (    long "example"
      <> short 'e'
      <> metavar "EXAMPLE"
      <> help "which example to run"
    )
  )
  <*> optional (
  switch
    (    long "verbose"
      <> short 'v'
      <> help "whether to be verbose"
    )
  )

data Vocabulary = Vocabulary [TokenMap]
  deriving Show

findVocabulary :: [(Key, Value)] -> Vocabulary
findVocabulary maybeTokenMaps = Vocabulary $ findTokenMap <$> maybeTokenMaps

instance ToJSON Vocabulary where
  toJSON (Vocabulary tokenMaps) = Array $ fromList $ toJSON <$> tokenMaps

instance FromJSON Vocabulary where
  parseJSON = withObject "Vocabulary" (\v -> pure $ findVocabulary $ toList v)
    where

data TokenMap =
  TokenMap { tm_token :: ByteString
           , tm_value :: Int
           }
  deriving Show

findTokenMap :: (Key, Value) -> TokenMap
findTokenMap (k,v) = case (k,v) of
                       (a, Number b) ->  case toBoundedInteger b of
                                           Just c -> TokenMap (encodeUtf8 $ toText a) (c)
                                           Nothing -> error $ "value out of bounds: " <> show b
                       (_,b) -> error $ "failed to parse " <> show b <> " as a Number."

instance ToJSON TokenMap where
  toJSON (TokenMap token value) = object [DAK.fromText (decodeUtf8 token) .= value]

instance FromJSON TokenMap where
  parseJSON = withObject "TokenMap" (\v -> pure $ onlyOne $ findTokenMap <$> toList v)
    where
      onlyOne :: (Show a) => [a] -> a
      onlyOne [] = error $ "no item!\n"
      onlyOne [a] = a
      onlyOne xs = error $ "too many items." <> show xs <> "\n"

-- A typeclass for tokenization.
class Tokenable s where
  -- | Establish a vocabulary from a set of strings.
  vocabOfText :: s -> Vocabulary
  -- Use a vocabulary to reconstruct tokens into a string.
  stringFromTokens :: Vocabulary -> [Int] -> s
  -- Use a vocabulary to tokenize an input.
  tokensFromString :: Vocabulary -> s -> [Int] 

instance Tokenable [Char] where
  vocabOfText s = vocabFromText s
  stringFromTokens v t = getStringFromTokens v t
  tokensFromString v s = getTokensFromString v Nothing s

vocabFromText :: [Char] -> Vocabulary
vocabFromText input = Vocabulary $ (\(t, v) -> TokenMap t v) <$> zip vocab [0,1..]
  where
    vocab :: [ByteString]
    vocab = sortUniq $ splitString input

-- | split up a string into tokens. yes, the barriers between tokens are arbitrary, matching the ones used in the book.
splitString :: [Char] -> [ByteString]
splitString input = BSU.fromString <$> separateDoubleDash
  where
    separateDoubleDash :: [[Char]]
    separateDoubleDash = concat $ split (dropBlanks $ onSublist "--") <$> separatePunctuation
    separatePunctuation :: [[Char]]
    separatePunctuation = concat $ split (oneOf ",.:;?_!\"()'") <$> words
    words = splitOneOf " \n" input

-- | Use a list of tokens to tokenize a string. optionally accepts an unknown token.
getTokensFromString :: Vocabulary -> Maybe Int -> [Char] -> [Int]
getTokensFromString (Vocabulary rawVocab) unk string = findTokenOfString <$> splitString string
  where
    findTokenOfString :: ByteString -> Int
    findTokenOfString s = case lookup s vocab of
                               Just t -> t
                               Nothing -> case unk of
                                            Just t -> t
                                            Nothing -> error $ "cannot find a token for \"" <> (BSU.toString s) <> "\"\n"
    vocab = (\(TokenMap t v) -> (t, v)) <$> rawVocab


getStringFromTokens :: Vocabulary -> [Int] -> [Char]
getStringFromTokens (Vocabulary rawVocab) tokens = maybeIntersperse ' ' $ findStringOfToken <$> tokens
  where
    maybeIntersperse :: Char -> [ByteString] -> [Char]
    maybeIntersperse _ [] = []
    maybeIntersperse x xs = foldr1 maybeIntersperse' (BSU.toString <$> xs)
      where
        maybeIntersperse' :: [Char] -> [Char] -> [Char]
        maybeIntersperse' a b = case (head b) `elem` (",.?!\"()'" :: [Char]) of
                                  False -> a ++ x:b
                                  True -> a ++ b
    findStringOfToken t = case lookup t bacov of
                            Just s -> s
                            Nothing -> error $ "cannot find a string for token" <> (show t) <> "\n"
    -- the vocabulary backwards: a mapping from value to token.
    bacov = (\(TokenMap t v) -> (v, t)) <$> rawVocab

example_2_1 :: [Char] -> [Char]
example_2_1 text = "Total number of character: " <> show (length text) <> "\n" <> take 99 text

example_2_2 :: [Char] -> Vocabulary
example_2_2 text = Vocabulary $ take 51 $ (\(Vocabulary a) -> a) $ vocabOfText text

-- | For example 2.3, they use it twice, we just implement this as two functions: one for encoding, and one for decoding.
-- This is the encoding one.
example_2_3_1 :: [Char] -> [Char] -> [Int]
example_2_3_1 text string = tokensFromString vocab string
  where
    vocab = vocabOfText text

-- | For example 2.3, they use it twice, we just implement this as two functions: one for encoding, and one for decoding.
-- This is the decoding one.
example_2_3_2 :: [Char] -> [Int] -> [Char]
example_2_3_2 text tokens = stringFromTokens vocab tokens
  where
    vocab = vocabOfText text

-- | Example 2.4 has several sub examples. This one prints the last 5 tokens in our extended vocabulary.
example_2_4_1 :: [Char] -> Vocabulary
example_2_4_1 text = Vocabulary $ drop (vocabLength vocab - 3) $ rawExtendedVocab vocab
  where
    vocab = vocabOfText text
    vocabLength (Vocabulary v) = length v
    rawExtendedVocab (Vocabulary v) = v ++ [TokenMap "<|endoftext|>" (length v), TokenMap "<|unk|>" (length v + 1)]

-- Example 2.4 has several sub examples. This one gives us the tokens at the top of page 32.
example_2_4_2 :: [Char] -> [Char] -> [Int]
example_2_4_2 text string = getTokensFromString (extendedVocab vocab) (Just $ vocabLength (extendedVocab vocab) - 1) string
  where
    vocab = vocabOfText text
    vocabLength (Vocabulary v) = length v
    extendedVocab (Vocabulary v) = Vocabulary $ v ++ [TokenMap "<|endoftext|>" (length v), TokenMap "<|unk|>" (length v + 1)]

-- Example 2.4 has several sub examples. This one gives us the reconstituted string on page 32.
example_2_4_3 :: [Char] -> [Int] -> [Char]
example_2_4_3 text tokens = stringFromTokens (extendedVocab vocab) tokens
  where
    vocab = vocabOfText text
    extendedVocab (Vocabulary v) = Vocabulary $ v ++ [TokenMap "<|endoftext|>" (length v), TokenMap "<|unk|>" (length v + 1)]

example_2_5_1 :: [Char] -> ByteString -> [Int]
example_2_5_1 text dictionaryFile = tokensFromString vocab text
  where
    vocab :: Vocabulary
    vocab = case decodeStrict dictionaryFile of
              Just v -> v
              Nothing -> error "no vocabulary?"

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
    readDictionary :: IO ByteString
    readDictionary = do
      input <- case dictionaryOpt rawArgs of
                 Nothing -> error "This example requires you to pass in your own dictionary, in JSON format."
                 Just inFile -> PL.readFile inFile
      return $ BSU.fromString input
    beVerbose = case verboseFlag rawArgs of
                  Nothing -> False
                  Just a -> a
  in
    case exampleOpt rawArgs of
      Example (2,1) -> do
        input <- readInput
        case beVerbose of
          True -> putStrLn $ "running listing 2.1 with input:\n<input>\n" <> input <> "<\\input>\n<result>" <> example_2_1 input <> "<\result>\n"
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
        putStrLn $ (show (example_2_5_1 example_2_5_String dictionary :: [Int])) <> "\n"
      Example (a,b) -> error $ "unknown listing: " <> show a <> "." <> show b <> "\n"
  where
    example_2_3_String, example_2_4_String, example_2_5_String :: [Char]
    example_2_3_String = "\"It's the last he painted, you know,\" Mrs. Gisburn said with pardonable pride."
    example_2_4_String = "Hello, do you like tea?" <> " <|endoftext|> " <> "In the sunlit terraces of the palace."
    example_2_5_String = "Hello, do you like" --  tea?" <> " <|endoftext|> " <> "In the sunlit terraces of someunknownPlace."

-- | The entry point. Use the option parser then run the trainer.
main :: IO ()
main = execParser opts >>= run
    where
      opts = info (helper <*> trainOpts)
             ( fullDesc
               <> progDesc "TrainLLM: An implementation of the examples from 'Build a Large Language Model (From Scratch)'."
               <> header "trainLLM - LLM Training in haskell."
             )

