## Installing Required Packages

Tested on python version 3.8.1

Suggested way: `pip install -r requirements.txt`
or you can do `pip install transformers sklearn torch`

## Running the code

Simply use `python main.py` to run the code.

## Parameters to adjust

1. Change `corpus_name` to change corpora to train on. Available corpus are: `AIMed` and `BioInfer`.
2. Change `model_name` to change model. `bert-base-uncased` is a large model, you may want to change it to smaller ones to save memory. But expect a performance drop when you use a smaller model. Suggested smaller models, like `prajjwal1/bert-tiny`, are in the comments of the code. Visit [huggingface website](https://huggingface.co/) for more available models. You can probably change it to other non Bert based models, like GPT2, for example.
3. Training parameters. You may want to adjust `learning_rate` and `num_epochs` to change the learning rate of stochastic gradient descent, and the number of training epochs, respectively.

## GPU support

This code automatically supports the use of GPU. See [this article](https://discuss.pytorch.org/t/how-to-change-the-default-device-of-gpu-device-ids-0/1041/22), for example, on how to select which GPU to use when you have multiple.

## Understading the code

Below are some website links for you to understand the code, if you wish to:

[Official pytorch tutorial](https://pytorch.org/tutorials/)

[Transformer package's example on fine-tuning](https://huggingface.co/transformers/custom_datasets.html#sequence-classification-with-imdb-reviews)

[Using dataloader in pytorch](https://pytorch.org/docs/stable/data.html)