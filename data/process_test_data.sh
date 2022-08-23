#!/bin/bash

BenchIE=evaluation_data/BenchIE
Carb=evaluation_data/carb
Wire57=evaluation_data/wire57


python process.py --source_file $BenchIE/sample300_en.txt --target_file $BenchIE/benchIE_test.json --conjunctions_file $BenchIE/benchIE_conjunctions.txt
python process.py --source_file $Carb/data/test.txt --target_file $Carb/carb_test.json --conjunctions_file $Carb/carb_test_conjunctions.txt
python process.py --source_file $Wire57/gold_data/wire57_test_sentences.txt --target_file $Wire57/wire57_test.json --conjunctions_file $Wire57/wire57_conjunctions.txt