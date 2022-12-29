# SD-based-on-AIEN-and-ICA
Sarcasm Detection Based on Adaptive Incongruity Extraction Network and Incongruity Cross-Attention
## Requirements
* PyTorch 1.12
* Transformers

## Training
The script for training is:
```
PYTHONENCODING=utf-8 python main.py --data_dir ./data/Ptacek \ 
--output_dir ./output/Ptacek_KL-Bert_output/ --do_train --do_test --model_select KL-Bert
```
where 
* `--data_dir` can be set as `./data/SARC_politics`, `./data/Ghosh`, and `./data/Ptacek`
* `--output_dir` should keep up with `data_dir` and `model_select` to be `./output/DATASETNAME_MODELNAME_output/`
* `--know_strategy` is for different knowledge selecting strategies, which can be `common_know.txt`, `major_sent_know.txt`, `contrast_sent_know.txt`, and `minor_sent_know.txt`. Our model uses all the obtained commonsense knowledge--`"common_know.txt"`, and the other three knowledge selection strategies are used for comparison experiments. 
* `--know_num` is to choose how many items of knowledge are used for each sentence, which is set to `'5'`, `'4'`, `'3'`, `'2'`, `'1'`

The script for testing is:
```
PYTHONENCODING=utf-8 python run_classifier.py --data_dir ./data/Ptacek \ 
--output_dir ./output/Ptacek_KL-Bert_output/ --do_test --model_select KL-Bert
```