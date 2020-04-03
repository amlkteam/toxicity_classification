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
