{- ORMOLU_DISABLE -}
-- LLM Trainer, following the book "Build a Large Language Model (From Scratch)".
{-
 - Copyright 2019 Julia Longtin
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

import Prelude (Integer, IO, Maybe(Nothing), (<$>), (<*>), (>>=), (<>), pure, putStrLn)

import Options.Applicative (Parser, execParser, fullDesc, header, helper, info, progDesc)

data TrainRootOpts =
  TrainRootOpts
    {
      example :: Maybe (Integer, Integer)
    }

trainOpts :: Parser TrainRootOpts
trainOpts = TrainRootOpts <$> pure Nothing

run :: TrainRootOpts -> IO ()
run rawArgs = putStrLn "Hello World"

-- | The entry point. Use the option parser then run the trainer.
main :: IO ()
main = execParser opts >>= run
    where
      opts = info (helper <*> trainOpts)
             ( fullDesc
               <> progDesc "TrainLLM: An implementation of the examples from 'Build a Large Language Model (From Scratch)'."
               <> header "trainLLM - LLM Training in haskell."
             )

