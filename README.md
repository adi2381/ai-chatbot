# AI seq2seq ChatBot
This is an attempt at building a ChatBot using [Seq2Seq](https://www.geeksforgeeks.org/seq2seq-model-in-machine-learning/) model. This model is based on 2 [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) Layers. Seq2Seq mainly consists of 2 components i.e Encoder and Decoder, hence sometimes it is also called Encoder-Decoder network.

The python files contains not only the code but also comments wherever necessary to explain the code and the working. For any further questions, you can send a request.

Scroll to bottom to see how you can train your model using Google Colab's GPU (which is a powerful Nvidia Tesla K80 GPU) coupled with 12GB Ram. Also, Google provides this service for free with a limit imposed on your runtime session, you can have a session for maximum of 12 hours after which it terminates.

Diagram:
![Encoder & Decoder Network](https://miro.medium.com/proxy/1*sO-SP58T4brE9EHazHSeGA.png)

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

### Dataset - Cornell Movie-Dialogs Corpus
```http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html```

## Result
I was only able to train the model for 15 epochs, which is very less for training a chatbot with a huge and complex dataset like cornell's movie dialouge corpus.
