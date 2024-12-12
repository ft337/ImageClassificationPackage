.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/ImageClassificationPackage.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/ImageClassificationPackage
    .. image:: https://readthedocs.org/projects/ImageClassificationPackage/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://ImageClassificationPackage.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/ImageClassificationPackage/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/ImageClassificationPackage
    .. image:: https://img.shields.io/pypi/v/ImageClassificationPackage.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/ImageClassificationPackage/
    .. image:: https://img.shields.io/conda/vn/conda-forge/ImageClassificationPackage.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/ImageClassificationPackage
    .. image:: https://pepy.tech/badge/ImageClassificationPackage/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/ImageClassificationPackage
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/ImageClassificationPackage

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

==========================
ImageClassificationPackage
==========================


    This repository contains a trained model to classify electrodes from the Kaggle dataset: ``https://www.kaggle.com/datasets/thgere/spent-lithium-ion-battery-recyclingslibr-dataset``
    Pictures can be calssified as having an "anode", "cathode", or "nothing". A notebook walk through of the training and testing is within `Notebooks` folder. 
    NOTE: Notebook example requires the Kaggle dataset to be installed to run it exactly as it is there. 


Two scripts are provided to run prediction with this model on either a single image or a set of images. 
    To install the package and run these scripts, follow these instructions: 

    1. Clone the repository, and set up a virtual environment

    2. Download requirements ``pip install -r docs/requirements.txt``
    
    3. Install the package locally ``pip install -e .`` 

    Two scripts are provided. If you would like to run inference on a single image, you can use:
    ``single-predict --input_path <path_to_image.jpg>`` 

    I have provided some sample data from the Chang Battery Dynamics and Engineering lab at Drexel. Thus, you can try:

    ``single-predict --input_path data/lab_electrode_images/nothing/nothing1.jpg`` 

    The other script is a gallery function, providing predictions for several images in a dataset. If your dataset is labelled, make a directory with subdirectories `anodes`, `cathodes`, and `nothing`.  To run this with our data, you may run:

    ``gallery --input_path data/lab_electrode_images --known_labels``

    If you have unlabelled data, exclude the `known_labels` flag.
    

.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.6. For details and usage
information on PyScaffold see https://pyscaffold.org/.
