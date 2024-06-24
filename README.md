# SkimLit

A Deep Learning project from https://arxiv.org/abs/1710.06071 and the data retrieved at https://github.com/Franck-Dernoncourt/pubmed-rct. We take an input of 20k data and process them in order to create a DL program. Mainly used Tensorflow. The project consists of multiple functions such as:
1. load_and_preprocess_pubmed_data
2.  read the lines of a document
3.  preprocess_text_with_line_numbers
4.  tensorBoard callback
5.  one_hot_encoding & label_encoding
6.  char_token_datasets with batch and prefetch
7.  calculate_results {accuracy, precision, recall, f1}
8.  plot the DL model

 There are 3+1(baseline) models where each of them is created,compiled and fitted and the summary of each one. Also, there are result comparisons at the end of the project to see which performed better. The models are:

 0. Baseline
 1. Hybrid with Universal Sentence Encoder
 2. A tribrid embedding model with transfer learning
 3. Model with callbacks(ModelCheckpoint, EarlyStopping, ReduceLROnPlateau)
