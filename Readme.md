# CompactIE
Source code for NAACL 2022 paper ["CompactIE: Compact Facts in Open Information Extraction"](https://aclanthology.org/2022.naacl-main.65/), which extracts compact facts for raw input sentences. Our pipelined approach consists of two models, Constituent Extraction and Constituent Linking models. First, Constituent Extraction model extracts the triple slots (constituents) and pass them to Constituent Linking models to link the constituents and form triples.
<p align="center">
<img src="https://github.com/FarimaFatahi/CompactIE/blob/main/example.png" width="750">
</p> 

## Requirements
* `python`: 3.6
* `pytorch`: 1.9.0
* `transformers`: 4.2.2
* `configargparse`: 1.2.3
* `bidict`: 0.20.0

## Datasets
The scripts and instructions for downloading and processing our benchmark are provided in [`data/`](https://github.com/FarimaFatahi/CompactIE/tree/master/data).

## Training
### Constituent Extraction Model
```python 
python constituent_model.py \
    --config_file constituent_model_config.yml \
    --device 0
```

### Constituent Linking Model
```python 
python linking_model.py \
    --config_file linking_model_config.yml \
    --device 0
```

Note that first the data for these models should be provided and processed (as described in [`data/`](https://github.com/FarimaFatahi/CompactIE/tree/master/data)) and then each model can be trained individually.  
If **OOM** occurs, we suggest that reducing `train_batch_size` and increasing `gradient_accumulation_steps` (`gradient_accumulation_steps` is used to perform *Gradient Accumulation*). 

## Inference
After training and saving the Constituent Extraction and Constituent Linking models, test the pipelined CompactIE on evaluation benchmarks. Three popular evaluation benchmarks (BenchIE, CaRB, Wire57) are provided to examine CompactIE's performance.
```python 
python test.py --config_file config.yml
```
### Pre-trained Models
Models checkpoint are available in [Zenodo](https://drive.google.com/drive/folders/1b7JD419DBJv2BrNduBYOs8floP1JgO0-?usp=sharing). Download the Constituent Extraction (`ce_model`) model and put in under `save_results/models/constituent/` folder. Download the Constituent Linking (`cl_model`) model and put in under `save_results/models/relation/` folder.

## Cite
If you find our code is useful, please cite:
```
@inproceedings{fatahi-bayat-etal-2022-compactie,
    title = "{C}ompact{IE}: Compact Facts in Open Information Extraction",
    author = "Fatahi Bayat, Farima  and
      Bhutani, Nikita  and
      Jagadish, H.",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.65",
    doi = "10.18653/v1/2022.naacl-main.65",
    pages = "900--910",
    abstract = "A major drawback of modern neural OpenIE systems and benchmarks is that they prioritize high coverage of information in extractions over compactness of their constituents. This severely limits the usefulness of OpenIE extractions in many downstream tasks. The utility of extractions can be improved if extractions are compact and share constituents. To this end, we study the problem of identifying compact extractions with neural-based methods. We propose CompactIE, an OpenIE system that uses a novel pipelined approach to produce compact extractions with overlapping constituents. It first detects constituents of the extractions and then links them to build extractions. We train our system on compact extractions obtained by processing existing benchmarks. Our experiments on CaRB and Wire57 datasets indicate that CompactIE finds 1.5x-2x more compact extractions than previous systems, with high precision, establishing a new state-of-the-art performance in OpenIE.",
}
```
