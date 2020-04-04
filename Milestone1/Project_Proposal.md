
# Project proposal -- Jigsaw Multilingual Toxic Comment Classification

## Introduction:

We would like to do toxic comments classification task, which is more like a sentiment analysis task in nature, that we will train neural models to learn about what words/phrases may be insulting, threatening or hurtful to others. The models will take in a sequence of text, and output a binary classification of whether a "toxic" element is found in the sequence, and if positive, also output a multi-label classification on the toxic comment.

## Motivation and Contributions

The project is primarily socially-motivated. The number of people chatting and commenting and just writing content on the Internet have increased exponentially over the past decade. A lot of people do use toxic language which should be dealt by first detecting if it is toxic and then dealing with it some way like hiding it or just deleting it. The goal of this project is to detect such toxic comments/messages and also classify the type of toxicity.

We hope that our system will be a fine-tuned version of one of the popular pretrained models thatworks really well for this specific task. We want to make an end to end library out of it as well where a sentence can be passed along and the labels can be easily assigned to it. If we have the time we would also want to explore the model interpretability and try to analyse why the model is assigning different labels to different sentences.


## Data

We will be using the dataset from the Kaggle Toxic Comment Classification

The training set has 159571 comments and every comment has 6 labels associated with it namely `toxic`, `server_toxic`, `obscene`, `threat`, `insult` and `identity_hate`. This is a multi-label dataset which means that a comment can have more than one class assigned to it. The assignments are represented by a 0 or 1 for each which class. 0 means the comment is not attributed to that class and 1 means that the comment is attributed to that specific task.

Even though there are 159571 comments in total there is a high level of class imbalance and most of the sentences cannot be attributed to any of the classes which is why we might have to use negative sampling to deal with the problem.

The data is in English language and it is just stored in CSV format.


## Engineering

We are planning to use Google Colab to run the experiments since none of us have access to GPU and it is not feasible to iterate on experiments that are run on a CPU.

We want to do multi-label text classification with the latest transformer architectures and pretrained models like `BERT`, `GPT` and other spin offs of the BERT models. We plan to iterate on the model and choose the best one along with the best hyperparameters.

We will be use the transformers library by Hugging Face which is based off PyTorch.

Link to the transformers library - [Click Here](https://huggingface.co/transformers/)

## Previous Works

There are a few papers published in 2018 which tested different machine learning models on toxic comment classfication that can serve as our basis for further exploration.

- In `"Detecting and Classifying Toxic Comments"` by Kevin Khieu and Neha Narwal from Stanford ([link]( https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1184/reports/6837517.pdf)), authors explored the use of Support Vector Machine(SVM), Long Short-Term Memory Networks(LSTM), Convolutional Neural Networks(CNN) and Multilayer Perceptron(MLP) with both word-level and character-level embeddings on the same Kaggle toxic comment classification challenge dataset. Authors achieved best test accuracy(0.889) and highest F1 score(0.886) on the binary classification task on a `word-level LSTM model` that has 3 layers with 32 output units at each layer. A CNN model with kernel size 3 and dropout ratio 0.2 achieved a F1 score of 0.871. Character-level neural models performed much worse. On much more fine-grained multiple-label toxic comment classification task, LSTM model also won best F1 score(0.706). The LSTM word-level model surprisingly could detect toxicity despite spelling mistakes in the toxic comment.

- In `"Challenges for Toxic Comment Classification: An In-Depth Error Analysis"` by Betty van Aken et al.([link](https://www.researchgate.net/publication/327345300_Challenges_for_Toxic_Comment_Classification_An_In-Depth_Error_Analysis)), authors point out that main challenges of toxic comment classification include `long-range dependencies`, `(intentionally) misspelled and idiosyncratic out-of-vocabulary words`, `class imbalance problem` and `high variance in data/inconsistency in labeling`. Authors applied an ensemble approach, combining strong classifiers of Logistic Regression, bidirectional RNN, bidirectional GRU with Attention layer and CNN, with pretrained word embeddings from Glove and sub-word embeddings from FastText. `Bidirectional GRU with Attention` outperformed other models but ensemble approach achieved even higher F1 scores(0.791 on wikipedia comments and 0.793 on another tweets dataset). For multi-label comment classification task, authors found that ensembling is especially effective on the sparse classes "threat" and "hate". In the follow-up error analysis, this paper discusses the remaining major prediction errors came from a few areas: `incorrect original labeling`; `toxicity without swear words`; `toxic comments framed as rhetorical questions, subtle metaphors and comparisons` that require more real world knowledge/context.
 

## Evaluation:

For this text classification task, accuracy, precision, recall and macro F1 score should be the most appropriate evaluation metrics. We will also visualize numbers of prediction errors in a confusion matrix, and list out some examples of false negatives and false positives to explore what can be further worked on to reduce model error. 
