## Dataset Processing

### Our Benchmark (processed OIE2016)

Firstly, download our benchmark tailored for compact extractions provided [`here`](https://zenodo.org/record/7014032#.YwQQ0OzMJb8) and put it under [`data/OIE2016(processed)`](https://github.com/FarimaFatahi/CompactIE/tree/master/data/OIE2016(processed)).
Secondly, split out the train, development, test set for the constituent extraction model by running:
``` 
cd OIE2016(processed)/constituent_model
python process_constituent_data.py
```
Lastly, split out the train, development, test set for the constituent linking model by running:
``` 
cd OIE2016(processed)/relation_model
python process_linking_data.py
```
Note that the data folders for training each model are set to the ones mentioned above.

### Evaluation Benchmarks

Three evaluation benchmarks (**BenchIE**, **CaRB**, and **Wire57**) are used for evaluating CompactIE's performance. Note that since these datasets are not targeted for compact triples, we exclude triples that have at least one clause within a constituent.
To get the final data (json format) for these benchmarks, run: 

```bash
./process_test_data.sh
```

### Other files
Since the schema design of the table filling model does not support conjunctions inside constituents, we use the conjunction module developed by [`OpenIE6`](https://github.com/dair-iitd/openie6) to break sentences into smaller conjunction-free sentences before passing them to the system.
Therefore, to input new test files (`source_file.txt`), produce the conjunction file (`conjunctions.txt`) and then run:
```
python process.py --source_file source_file.txt --target_file output.json --conjunctions_file conjunctions.txt
```


