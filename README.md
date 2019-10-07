# AI seq2seq ChatBot
This is an attempt at building a ChatBot using [Seq2Seq](https://www.geeksforgeeks.org/seq2seq-model-in-machine-learning/) model. This model is based on 2 [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) Layers. Seq2Seq mainly consists of 2 components i.e Encoder and Decoder, hence sometimes it is also called Encoder-Decoder network.

Diagram:
![Encoder & Decoder Network](https://miro.medium.com/proxy/1*sO-SP58T4brE9EHazHSeGA.png)

## Dependencies
1. Install [Anaconda](https://www.anaconda.com/distribution/)
2. Download the [Cornell Movie-Dialogs Corpus](http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) Dataset
3. Install Python 3.5
4. Install Tensorflow 1.0.0

## Getting Started (Windows)
### Install Anaconda
```https://www.anaconda.com/distribution/```

### Installing Python 3.5 and Creating Virtual Environment
Please note that, when you're running the below line of code, you do so in the anaconda shell to avoid any issues. "chatbot" is the name of the virtual environment.

```conda create -n chatbot python=3.5 anaconda ```

Press 'y' and enter when it promts you for y/n

### Activate Virtual Environment
```activate chatbot```

To deactivate the virtual environment, simply use **deactivate**

### Install Tensorflow 1.0.0
```pip install tensorflow==1.0.0``` 



## Status
Need to train model
