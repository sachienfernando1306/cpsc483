# CPSC 483 Final Project

This is an implementation of a Simplified Graph Convolutional Network for Text Classification. This project takes heavily from the [TextGCN](https://github.com/yao8839836/text_gcn)and the [SGC](https://github.com/Tiiiger/SGC/) repos. It has been modified for use for Python 3.11, since those projects were specifically meant for previous Python versions. 


Requirements can be found from the _requirements.txt_ file. 

Once the environment has been found below, run the following lines. 

**Procuring Data**

For some reason, I was running into difficulties with upolading the data directly to Github. Instead, please follow [this link](https://github.com/yao8839836/text_gcn/tree/master/data) link directly to the original TextGCN. This will provide you with the data to replicate the TextGCN and SGC experiments.

To equip yourself with the trec dataset, simply run the _fast.py_ file, and change the label to _coarse_label_ and _fine_label_ for the 8 classifier and the 50 classifier respectively. This code should download the test and train splits into your data file. From then, proceed as all the other data. 

**Code Usage**
1) Remove Words: This code removes stopwords from the dataset and preprocesses the corpus for use in the graph building. 
```
python3.11 remove_words.py --dataset <dt> # <dt> can be any of R8, MR, ohsumed, trec, R52, 20ng
```

2) Building Graph: This code takes the generated words and builds the corpus-level graph.
```
python3.11 build_graph.py --dataset <dt> # <dt> can be any of R8, MR, ohsumed, trec, R52, 20ng
```

3) Train: This line of code trains the model and returns the test accuracy result
```
python3.11 train.py --dataset <dt> # <dt> can be any of R8, MR, ohsumed, trec, R52, 20ng
```

Enjoy and thank you :)
