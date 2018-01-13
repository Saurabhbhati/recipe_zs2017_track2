Phoneme Based Embedded Segmental K-Means for ZeroSpeech2017 Track 2
===================================================================

ES-Kmeans starts from an initial set of boundaries and iteratively eliminates boundaries to discover frequently occurring longer word patterns. We use phonemes for initializing the ES-Kmeans. The phoneme initialization usually results in a lower deviation between the discovered word boundaries and true word boundaries as smaller units like phoneme allow finer adjustments while discovering words. The usage of smaller acoustic units also increases the number of combinations that the algorithm has to check. We use a deep stacked autoencoder to learn compact embeddings to reduce the computational cost.  

Warning
-------
This is a preliminary version of our system. This is not a final recipe, and is still being worked on.

Overview
--------
A description of the challenge can be found here:
<http://sapience.dec.ens.fr/bootphon/2017/index.html>.


Disclaimer
----------
The code provided here is not pretty. I provide no guarantees with the code,
but please let me know if you have any problems, find bugs or have general
comments.


Preliminaries
-------------
Clone the zerospeech repositories:

    mkdir ../src/
    git clone https://github.com/bootphon/zerospeech2017.git \
        ../src/zerospeech2017/
    # To-do: add installation and data download instructions
    git clone https://github.com/bootphon/zerospeech2017_surprise.git \
        ../src/zerospeech2017_surprise/

Clone the `eskmeans` repository:

    git clone https://github.com/kamperh/eskmeans.git \
        ../src/eskmeans/

Get the surprise data:
    
    cd ../src/zerospeech2017_surprise/
    source download_surprise_data.sh \
        /share/data/lang/users/kamperh/zerospeech2017/data/surprise/
    cd -

Update all the paths in `paths.py` to match your directory structure.


Feature extraction
------------------
Extract MFCC features by running the steps in
[features/readme.md](features/readme.md).


Unsupervised phoneme boundary detection
----------------------------------------
We use the unsupervised phoneme boundary detection algorithm described in:

- Saurabhchand Bhati, Shekhar Nayak, and K. Sri Rama Murty, “Unsupervised Segmentation of Speech Signals Using Kernel-Gram Matrices" in Proc. NCVPRIPG, Communications in Computer and Information Science, Springer

A phoneme based system for feature learning and spoken term discovery can be found here: 

- Saurabhchand Bhati, Shekhar Nayak, and K. Sri Rama Murty, “Unsupervised Speech Signal to Symbol Transformation for Zero Resource Speech Application” in Proc. Interspeech 2017 [pdf](http://www.isca-speech.org/archive/Interspeech_2017/pdfs/1476.PDF) 

Acoustic word embeddings: downsampling
--------------------------------------
We use one of the simplest methods to obtain acoustic word embeddings:
downsampling. Different types of input features can be used. Run the steps in
[downsample/readme.md](downsample/readme.md).

We use keras to learn low dimensional embeddings from the downsampled segments. 

Unsupervised segmentation and clustering
----------------------------------------
Segmentation and clustering is performed using the
[ESKMeans](https://bitbucket.org/kamperh/eskmeans/) package. Run the steps
in [segmentation/readme.md](segmentation/readme.md).


Dependencies
------------
- [Python](https://www.python.org/)
- [NumPy](http://www.numpy.org/) and [SciPy](http://www.scipy.org/).
- [HTK](http://htk.eng.cam.ac.uk/): Used for MFCC feature extraction.
- [Matlab](https://www.mathworks.com/): Used for phoneme boundary detection.
- [keras](https://keras.io/): Used for training stacked auto-encoder
