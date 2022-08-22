=================================
Product Embeddings with Prod2BERT
=================================

.. image:: https://github.com/vinid/prodb/raw/master/training_example.jpg
   :align: center
   :width: 600px


Abstract
========

Word embeddings (e.g., word2vec) have been applied successfully to eCommerce products through prod2vec. Inspired by the recent performance improvements on several NLP tasks brought by contextualized embeddings, we propose to transfer BERT-like architectures to eCommerce: our model - Prod2BERT - is trained to generate representations of products through masked session modeling. Through extensive experiments over multiple shops, different tasks, and a range of design choices, we systematically compare the accuracy of Prod2BERT and prod2vec embeddings: while Prod2BERT is found to be superior in several scenarios, we highlight the importance of resources and hyperparameters in the best performing models. Finally, we provide guidelines to practitioners for training embeddings under a variety of computational and data constraints.

Reference
=========

Most of the code has been taken from here: https://keras.io/examples/nlp/masked_language_modeling/
We added minor stuff to better adapt it to our use case (like saving to disk and doing next item prediction).


Citation for the paper:

::

    @inproceedings{tagliabue-etal-2021-bert,
    title = "{BERT} Goes Shopping: Comparing Distributional Models for Product Representations",
    author = "Bianchi, Jacopo  and
      Tagliabue, Jacopo  and
      Yu, Bingqing",
    booktitle = "Proceedings of The 4th Workshop on e-Commerce and NLP",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.ecnlp-1.1",
    doi = "10.18653/v1/2021.ecnlp-1.1",
    pages = "1--12",
    abstract = "Word embeddings (e.g., word2vec) have been applied successfully to eCommerce products through prod2vec. Inspired by the recent performance improvements on several NLP tasks brought by contextualized embeddings, we propose to transfer BERT-like architectures to eCommerce: our model - Prod2BERT - is trained to generate representations of products through masked session modeling. Through extensive experiments over multiple shops, different tasks, and a range of design choices, we systematically compare the accuracy of Prod2BERT and prod2vec embeddings: while Prod2BERT is found to be superior in several scenarios, we highlight the importance of resources and hyperparameters in the best performing models. Finally, we provide guidelines to practitioners for training embeddings under a variety of computational and data constraints.",
    }

Usage
=====

Code should be easy to use.

You should first git clone the repository and install the package locally. From the project folder:

.. code-block:: bash

    pip install -e .

Then, you need to first declare the config class with the parameters of interest

.. code-block:: python

    @dataclass
    class Config:
        MAX_LEN = 20
        BATCH_SIZE = 32
        LR = 0.001
        VOCAB_SIZE = 20000
        EMBED_DIM = 128
        NUM_HEAD = 8
        EPOCHS = 100
        MASKING_PROBABILITY = 0.25
        DATA_RATIO = 10 # dummy variable, we used this to understand if data size had an effect
        FF_DIM = 128
        NUM_LAYERS = 1


    config = Config()

Then, you can simply use the prodb class and give it in input a sequence of sessions:

.. code-block:: python

    from prodb.prodb import ProdB

    sessions = ["item1 item2 item3", "item2 item8 item9 item1 item5"]

    pb = ProdB(sessions, config)
    pb()

    next_element_prediction = ["item1 item8 item9"]
    # the model will use the sequence item1 item8 to predict the next item (item9 is considered the next item to predict)
    # the model will return the groundtruth (item9) and the items predicted by the model

    results = pb.run_next_item_predictions(next_element_prediction)


After training, you should find the model saved on disk
