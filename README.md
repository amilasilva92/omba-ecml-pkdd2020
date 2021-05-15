# OMBA: User-Guided Product Representations for Online Market Basket Analysis
[![GitHub](https://img.shields.io/github/license/amilasilva92/multilingual-communities-by-code-switching?style=plastic)](https://opensource.org/licenses/MIT)

This repository provides the source codes and the formats of the data files to reproduce the results in the following paper:

```
OMBA: User-Guided Product Representations for Online Market Basket Analysis
Amila Silva, Ling Luo, Shanika Karunasekera, Christopher Leckie
In Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases, 2021 (ECML-PKDD2020)
```

##### Datasets 
A toy dataset (from TF dataset) is included in the /data/ directory due to the large file sizes of the full datasets. All three datasets (CJ, TF, and IC) used in this paper can be downloaded via the following links.

Complete Journey Dataset (CJ) :
[https://www.dunnhumby.com/careers/engineering/sourcefiles](https://www.dunnhumby.com/careers/engineering/sourcefiles)

Ta-Feng Dataset (TF) : 
[http://www.bigdatalab.ac.cn/benchmark/bm/dd?data=Ta-Feng](http://www.bigdatalab.ac.cn/benchmark/bm/dd?data=Ta-Feng) 

InstaCart Dataset (IC) : 
[https://www.instacart.com/datasets/grocery-shopping-2017](https://www.instacart.com/datasets/grocery-shopping-2017)

Please follow the below steps to run the evaluation. 

##### Instructions to Run 
1. Install required libraries (Note: The code is written in python 2.7)
```shell=
pip install -r requirements.txt
```
2. Go to the directory "code\"
```shell=
cd code/
```

3. Run the evaluation script
```shell=
python train_eval.py [your_yaml_file]
```
One example yaml file is shown in "scripts/toy.yaml"

[your_yaml_file] will be used to specify:
1) the paths of your input tweets and output models;
2) your personalized parameters. 

##### Input Data Format
Each line in the input file is an instance to the model, which consists of 6 fields separated by "\x01":
1) shopping basket id
2) user id
3) timestamp
4) product list (seperated by space)
5) values of the products (seperated by space)
6) quantitiy of the products (seperated by space)
 
One example input file is shown in "data/TF/input/transactions.txt"

##### Hyperparameters
Here we describe the hyperparameters of OMBA, with some advice in how to set them.

[voca_min] 
What is it: The products will be ranked based on their frequencies (high to low). The [voca_min] most frequent products will be ignored in the training process.

[voca_max]
What is it: Similar to [voca_min], The products less frequent than the [voca_max]th ranked products will be ignored in the training process.
How to set it: It can be used to control your time and space complexity. A smaller [voca_max] will save you both memory and time. Quantitative evaluation results are not so sensitive to [voca_max] as long as it is larger than a few thousand, but qualitative results may be affected if you care about low-frequency products. 

[update_tweets]
What is it: Setting this to 0, the model will ignore the parameters set by [voca_min] and [voca_max] by default.


[dim]
What is it: The dimension of the learnt embeddings. 
How to set it: It is an important parameter, affecting the trade-off between efficiency and effectiveness. The time and space cost can scale almost linearly with [dim], but an insufficiently large [dim] will largely sacrifice the effectiveness. In our study, we found setting [dim] to a few hundreds can be a good choice for the mrr to be plateaued with it. 

[negative]
What is it: The number of negative samples for each positive sample.
How to set it: The time complexity linearly increases with [negative].

[alpha]
What is it: The standard learning rate.
How to set it: In our study, we found 0.01-0.1 can be a reasonable range for [alpha]. A too large [alpha] will lead to diverge, while a too small [alpha] will make converge very slow. 

[epoch]
What is it: The number of epochs. 
How to set it: Similar as [dim], it is another important parameter affecting the trade-off between efficiency and effectiveness. The time and space cost can scale almost linearly with [epoch], but an insufficiently large [epoch] will largely sacrifice the effectiveness. In our study, we found setting [epoch] to a few dozens can be a good choice for the mrr to be plateaued with it. 

[nt_list]
What is it: The list of "node types" to embed. Possible "node type" can be "w" (denoting products), "u" (denoting users).
How to set it: It's better to include and embed all of these types (users, and products) of information. But you can also choose to only retain some of them to check the performance weakend variants of OMBA.

[predict_type]
What is it: The list of "node types", on which to perform our quantitative evaluation and report mr and mrr.
How to set it: set as 'w' to get the resuts for the product retrieval task.

[load_existing_model]
What is it: Will determine whether to train the model from scratch or load the model from a previously trained model 
How to set it: Please set it to 0 if you are training the model using the specified parameters for the first time, otherwise you could set it to 1.

[test_batch_num]
What is it: The number of randomly sampled one-day windows.
How to set it: Set it to at least 10 to obtain reliable results.

[regu_weight]
What is it: The tau value in the proposed 'adaptive-optimization stratergy"
How to set it: Somewhere around 1e-1 to 1.
