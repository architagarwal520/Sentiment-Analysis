Title:          README.txt
Date:           2023-15-08
Author:         M.Conway

++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
+++ Data for Assignment 3 of COMP90049_2023_SM2 
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

Files used for Assignment 3:


## TRAINING/VALIDATION/TEST SPLITS

Columns:
(1) index
(2) dr-id-adjusted
(3) dr_id_gender
(4) review-text-cleaned
(5) rating

See PDF file associated with Assignment 3 for a detailed description
of the fields

Note that the rating column (i.e. the sentiment labels) is not present 
in the test set

./TRAIN.csv
Approximately 80% of the data

./VALIDATION.csv
Approximately 10% of the data

./TEST_NO_LABELS.csv
Approximately 10% of the data


## TFIDF TRAINING/VALIDATION/TEST SPLITS

Note that row numbers are consistent across all
training/validation/test data files

./rTFIDF_TRAIN.csv
TFIDF representations (500 highest rank TFIDF features) for training
set.

./TFIDF_VALIDATION.csv
TFIDF representations (500 highest rank TFIDF features) for validation
set.

./TFIDF_TEST.csv
TFIDF representations (500 highest rank TFIDF features) for test
set.


## WORD EMBEDDING TRAINING/VALIDATION/TEST SPLITS

Note that row numbers are consistent across all
training/validation/test data files

./384EMBEDDINGS_TRAIN.csv
384-dimension word embedding representation for training set

./384EMBEDDINGS_VALIDATION.csv
384-dimension word embedding representation for validation set

./384EMBEDDINGS_TEST.csv
384-dimension word embedding representation for test set


# TFIDF FEATURE SELECTION

./tfidf_words.txt
Contains the 500 most discriminating features as identified by TFIDF
