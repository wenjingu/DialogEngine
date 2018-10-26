# NLU Benchmark

1.	NLU benchmark dataset published by Snips in 2017 https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines
7 intents with roughly 2500 utterances for each intent. I transformed data into csv and benchmarked with splits between training and testing in 80-20 and 20-80 portions respectively.
 
In 20-80 test:
Rasa NLU scores 97.29% with 344 false predictions in total 12708 samples
Snips NLU scores 96.69% with 421 false predictions in total 12708 samples
My DNN Elmo based NLU beats both of them scores 97.94% with 262 false predictions in total 12708 samples
 
In 80-20 test: 
Rasa NLU scores 98.08% with 61 false predictions in total 3177 samples
Snips NLU scores 97.70% with 73 false predictions in total 3177 samples

 
2.	Google smalltalk samples
86 intents  with roughly 2500 utterances in total. The distribution is unbalanced. Each intent has 20 -100 utterances.  I split the training and testing dataset in 80-20 portions and used stratified train/test-split. 
 
Rasa NLU scores 82.26% with 69 false predictions in total 389 samples
Snips NLU scores 72.49% with 107 false predictions in total 389 samples
DNN NLU scores 68.12% with 124 false predictions in total 389 samples
   
