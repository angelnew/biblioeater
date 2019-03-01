# BiblioEater

This is a didactic project to illustrate a technique to assign literary texts to its
original authors. Edgar Allan Poe and Mark Twain were selected for this exercise.

## Installation

Clone this repository on your local machine for testing and playing purposes. This is an analytical project not intended for production.

### Prerequisites

This project was developed for Python3.6. The file requirements.txt incorporates all Python packages
that are dependencies for the full functionality of the project with you can install with

```
python3.6 -m pip install -r requirements.txt
```

To insulate your existing installation from these requirements it is advised to set up and activate a virtual
environment first (see for example https://realpython.com/python-virtual-environments-a-primer/).

As part of the requirements, stanfordnlp will be installed. We need now to download the 
English model provided by the library. Use the interactive Python3.6 interpreter to do it

```
import stanfordnlp
stanfordnlp.download('en')
```

### Data dowloading and preparation

Use a web browser or curl utility to download the following books:

* Arthur Gordon Pym: http://www.gutenberg.org/cache/epub/51060/pg51060.txt
* Tom Sawyer: https://www.gutenberg.org/files/74/74-0.txt
* Eureka: https://www.gutenberg.org/files/32037/32037-0.txt
* Huckleberry Finn: https://www.gutenberg.org/files/76/76-0.txt

Use a text editor to remove any text at the beginning and end of the four works that does not 
belong to the actual book. For Tom Sawyer and Huckleberry Finn replace curly double 
quotes with regular ones. For those two books remove underscores. KEEP THESE MODIFIED 
COPIES TO YOURSELF, as Project Gutenberg licence forbis distribution of modified copies. Customise
the constant CORPORA_FOLDER in constants.py to the folder where you hold the texts.

## Configuration

You may want to customise the conf/logging.ini file and the constants.py file which 
hold a number of filenames and paths. The existing configuration should work in most cases.

### Loading and parsing the data

Run the module load_corpora.py. 

```
python3.6 load_corpora.py
```

It is the slowest of them all especially if you do not have access to GPU(s).
It uses StanfordNLP to parse the four books and store the parsed result to disk.

### Produce some descriptive statistics

Run module descriptive.py

### Generate features for modelling

Run module feature_preparation.py, which will produce generate the training data set for both models
in the project

### Train the neural network

Run train_seq.py. This is the neural network featured in our article. Training parameters
can be tweaked at the beginning of class BiblioEater in module biblio_eater.py. You can ignore the warnings
at the beginning.

Alternatively you can run train_multi.py for another network topology not presented in
the article.

### Validate the results

Run validation_seq.py (or validation_multo.py) to assess how the network performs on the 
two books not included in the training. 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

