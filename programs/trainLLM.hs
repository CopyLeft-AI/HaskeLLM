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

import Prelude (Bool(True, False), Char, Int, IO, Maybe(Just, Nothing), String, (<$>), (<*>), (>>=), (<>), (&&), (==), (<), (>), ($), (*), (+), (-), concat, error, fromIntegral, getContents, length, not, otherwise, pure, putStrLn, return, show, take, zip)

import qualified Prelude as PL (readFile)

import Data.Aeson (Value(Number), FromJSON(parseJSON), eitherDecode, withObject)

import Data.Aeson.Key (Key, toText)

import BPE.Base (Id, Merges, Seq, Vocab, mergesToVocab)

import BPE.Basic (decode, encode)

import BPE.Regex (encodeOrdinary, gpt2pattern)

import qualified Data.Aeson.KeyMap as DAKM (toList)

import qualified Data.ByteString as BSS (ByteString, pack, singleton, unpack)

import qualified Data.ByteString.Lazy as BSL (ByteString, fromStrict, readFile, toStrict)

import qualified Data.ByteString.Lazy.UTF8 as BSLU (break, drop, lines)

import qualified Data.ByteString.UTF8 as BSU (toString, fromString)

import Data.Char (digitToInt, isDigit, isHexDigit)

import Data.Either (Either (Left, Right), lefts, rights)

import Data.HashMap.Strict.InsOrd (InsOrdHashMap, empty, insert, lookup, size)

import qualified Data.HashMap.Strict.InsOrd as DHSI (fromList, toRevList, toList, union)

import Data.List ((++), drop, elem, foldr1, head, sort)

import Data.List.Split (dropBlanks, oneOf, onSublist, split, splitOneOf)

import Data.List.Unique (sortUniq)

import Data.Scientific (toBoundedInteger)

import Data.Text.Encoding (encodeUtf8)

import Data.Word (Word8)

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
    , mergesOpt :: Maybe String
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
  switch
    (    long "verbose"
      <> short 'v'
      <> help "whether to be verbose"
    )
  )

-- A Bacov, a Vocab with the key and value swapped. native form, when coming out of our file.
data Bacov = Bacov (InsOrdHashMap BSS.ByteString Int)

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

instance FromJSON Bacov where
  parseJSON = withObject "Bacov" (\v -> pure $ findBacov $ DAKM.toList v)

-- A typeclass for tokenization.
class Tokenable s where
  -- | Establish a vocabulary from a set of strings.
  vocabOfText :: s -> Vocab
  -- Use a vocabulary to reconstruct tokens into a string.
  stringFromTokens :: Vocab -> [Int] -> s
  -- Use a vocabulary to tokenize an input.
  tokensFromString :: Vocab -> s -> [Int]

instance Tokenable [Char] where
  vocabOfText s = vocabFromText s
  stringFromTokens v t = getStringFromTokens v t
  tokensFromString v s = getTokensFromString v Nothing s

vocabFromText :: [Char] -> Vocab
vocabFromText input = tokenMaps
  where
    tokenMaps :: InsOrdHashMap Int BSS.ByteString
    tokenMaps = tokenMaps' (zip [0,1..] vocab) empty
    tokenMaps' [] map = map
    tokenMaps' [(k,v)] map = insert k v map
    tokenMaps' ((k,v):xs) map = insert k v (tokenMaps' xs map)
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

example_2_1 :: [Char] -> [Char]
example_2_1 text = "Total number of character: " <> show (length text) <> "\n" <> take 99 text

example_2_2 :: [Char] -> Vocab
example_2_2 text = DHSI.fromList $ take 51 $ DHSI.toRevList $ vocabOfText text

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
example_2_4_1 :: [Char] -> Vocab
example_2_4_1 text = DHSI.fromList $ drop (vocabLength vocab - 3) $ DHSI.toList $ extendedVocab vocab
  where
    vocab = vocabOfText text
    vocabLength v = length v
    extendedVocab :: Vocab -> InsOrdHashMap Int BSS.ByteString
    extendedVocab v = insert (size v) "<|endoftext|>" (insert (size v+1) "<|unk|>" v)

-- Example 2.4 has several sub examples. This one gives us the tokens at the top of page 32.
example_2_4_2 :: [Char] -> [Char] -> [Int]
example_2_4_2 text string = getTokensFromString (extendedVocab vocab) (Just $ vocabLength (extendedVocab vocab) - 1) string
  where
    vocab = vocabOfText text
    vocabLength v = length v
    extendedVocab v = insert (size v) "<|endoftext|>" (insert (size v+1) "<|unk|>" v)

-- Example 2.4 has several sub examples. This one gives us the reconstituted string on page 32.
example_2_4_3 :: [Char] -> [Int] -> [Char]
example_2_4_3 text tokens = stringFromTokens (extendedVocab vocab) tokens
  where
    vocab = vocabOfText text
    extendedVocab v = insert (size v) "<|endoftext|>" (insert (size v+1) "<|unk|>" v)

example_2_5_1 :: [Char] -> BSL.ByteString -> BSL.ByteString -> Seq
example_2_5_1 text merges dictionary
  | mergeDictionary == jsonDictionary = encode initVocabGPT2 (mergesFromTXT merges) initSeqGPT2 (BSU.fromString text) -- encodeOrdinary (mergesFromTXT merges) gpt2pattern (BSU.fromString text)
  | otherwise = error $ "Dictionaries not identical:\nTEXT: " <> (show $ take 100 $ drop 50200 $ sort $ DHSI.toList $ mergeDictionary) <> "\n"
                     <> "JSON: " <> (show $ take 100 $ drop 50200 $ sort $ DHSI.toList $ jsonDictionary) <> "\n"
  where
    -- a dictionary from a merge file.
    mergeDictionary = extendedVocab $ mergesToVocab (mergesFromTXT merges) initVocabGPT2
    extendedVocab v = insert (size v) "<|endoftext|>" v
    -- a dictionary from a dictionary file.
    jsonDictionary = dictionaryFromJSON dictionary

example_2_5_2 :: Seq -> BSL.ByteString -> BSL.ByteString -> BSS.ByteString
example_2_5_2 seq merges dictionary
  | mergeDictionary == jsonDictionary = respaceGPT2 $ decode jsonDictionary seq -- encodeOrdinary (mergesFromTXT merges) gpt2pattern (BSU.fromString text)
  | otherwise = error $ "Dictionaries not identical:\nTEXT: " <> (show $ take 100 $ drop 50200 $ sort $ DHSI.toList $ mergeDictionary) <> "\n"
                     <> "JSON: " <> (show $ take 100 $ drop 50200 $ sort $ DHSI.toList $ jsonDictionary) <> "\n"
  where
    -- a dictionary from a merge file.
    mergeDictionary = extendedVocab $ mergesToVocab (mergesFromTXT merges) initVocabGPT2
    extendedVocab v = insert (size v) "<|endoftext|>" v
    -- a dictionary from a dictionary file.
    jsonDictionary = dictionaryFromJSON dictionary
    respaceGPT2 :: BSS.ByteString -> BSS.ByteString
    respaceGPT2 bs = BSS.pack $ respaceGPT2' (BSS.unpack bs)
    respaceGPT2' [] = []
    respaceGPT2' [x] = [x]
    respaceGPT2' (x:y:xs)
      -- a special character representing a single space, for GPT2, it's 'Ġ', UTF8 character U+120
      | x == 196 && y == 160 = 32 : respaceGPT2' xs
      -- regular characters.
      | otherwise = x : respaceGPT2' (y:xs)

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

-- The default starting vocabulary, taken from the first 256 tokens of gpt2.
initVocabGPT2 :: Vocab
initVocabGPT2 = flip defaultBacov
  where
    flip ::  Bacov -> Vocab
    flip vk = DHSI.fromList (swap <$> (DHSI.toList $ (\(Bacov v) -> v) vk))
      where
        swap (a,b) = (b,a)

-- The default starting vocabulary, taken from the first 256 tokens of gpt2.
-- An initial (reverse) vocabulary, consisting of....
defaultBacov :: Bacov
defaultBacov = Bacov $ DHSI.fromList $ (zip (BSS.singleton <$> [33, 34..]) [0,1..93]) -- the first 94 characters of ascii, after 33 control signals.
                                     <> (zip ((\a -> BSS.pack [194,128+a]) <$> [33,34..44]) [94, 95..105] )    -- UTF8 characters U+00A1-U+00AB
                                     <> (zip ((\a -> BSS.pack [194,128+a]) <$> [46,47..63]) [106, 107..123] )  -- UTF8 characters U+00AD-U+00BF
                                     <> (zip ((\a -> BSS.pack [195,128+a]) <$> [0,1..63]) [124, 125..187] )    -- UTF8 characters U+00C0-U+00FF
                                     <> (zip ((\a -> BSS.pack [196,128+a]) <$> [0,1..63]) [188, 189..251] )    -- UTF8 characters U+0100-U+013F
                                     <> (zip ((\a -> BSS.pack [197,128+a]) <$> [0,1..63]) [252, 253..255] )    -- UTF8 characters U+0140-U+017F

-- convert an ascii string into a sequence of tokens in the initVocabGPT2 token space.
initSeqGPT2 :: BSS.ByteString -> Seq
initSeqGPT2 text = conv <$> BSS.unpack text
  where
    conv :: Word8 -> Id
    conv chr
      -- the special character, standing in for space.
      | chr == 32 = 220
      -- ASCII 34-128 -> Int 1-94
      | (chr > 33) && (chr < (94 + 33)) = fromIntegral $ chr - 33
      | otherwise = error $ "whoops: " <> show chr <> "\n"

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
        merges <- readMerges
        putStrLn $ (show (example_2_5_1 example_2_5_String merges dictionary)) <> "\n"
                <> show (example_2_5_2 example_2_5_Seq_Given merges dictionary) <> "\n"
                <> show (example_2_5_2 example_2_5_Seq_Found merges dictionary) <> "\n"
                <> show (example_2_5_2 (example_2_5_1 example_2_5_String merges dictionary) merges dictionary) <> "\n"
      Example (a,b) -> error $ "unknown listing: " <> show a <> "." <> show b <> "\n"
  where
    example_2_3_String, example_2_4_String, example_2_5_String :: [Char]
    example_2_3_String = "\"It's the last he painted, you know,\" Mrs. Gisburn said with pardonable pride."
    example_2_4_String = "Hello, do you like tea?" <> " <|endoftext|> " <> "In the sunlit terraces of the palace."
    example_2_5_String = "Hello, do you like tea?" <> " <|endoftext|> " <> "In the sunlit terraces of someunknownPlace."
    example_2_5_Seq_Found, example_2_5_Seq_Given :: Seq
    example_2_5_Seq_Found = [15496,11,466,345,588,8887,30
                            ,1279,91,437,1659,5239,91,29
                            ,554,262,4252,18250,8812,2114,286,617,34680,27271,13]
    example_2_5_Seq_Given = [15496,11,466,345,588,8887,30,
                             220, 50256
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
