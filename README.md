# Summary
This project implements a BERT–BiLSTM architecture for news sentiment classification by combining contextual embeddings from BERT with sequential modeling using BiLSTM. 
The model is trained on the [LIAR Fake News Dataset](https://www.kaggle.com/datasets/csmalarkodi/liar-fake-news-dataset), which consists of 14 feature columns and 6 sentiment classification labels. The project aims to evaluate the effectiveness of the model on large news dataset which has over 10000 datas
 

## Content
- [Requirements](#requirements)
- [Methods](#methods)
- [Steps](#steps)
- [Result](#result)

## Requirements
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

## Methods

### Data Preparation
- Remove missing values
- No text cleaning 
    * Since BERT preserves contextual meaning internally, standard text cleaning steps such as stopword removal or lemmatization are not suitable. 
- Construct metadata
    * Metadata is consist of fields like subject, speaker, context, etch. The metadata exclude credit scores since it's ambiguous in assigning numerical values to categorical attributes
    * Labels are divide to 6 classes, following the original dataset and 2 classes which is only `true` and `mostly-true` is classified as **True** and rest of other classes is **False** 
- Split the datas portion
    * The dataset is already split into training, validation, and testing parts. To reduce memory usage, this project use 80% of train, 50% of valid, 50% of test 

### Models
- BERT 
    * The BERT model uses `bert-base-uncased`, so manual lowercasing of the statements is not required. BERT produces contextual embedding vectors with a dimensionality of 768 for each token.
    
    * Input tokens are generated using the `BertTokenizer`, and each sequence is padded or truncated 
to a fixed length of 512 tokens  
- BI-LSTM
    * Despite BERT meaningful vector representations, it still lack of information about token order. BI-LSTM is processing tokens from both directions that provide past and future contextual information.
    * This model uses 2 BI-LSTM layers that produce logits using softmax for both 6 or 2 multi-classes classification 

### Training and Evaluation
- Training is using `CrossEntropyLoss` as criterion and `AdamW` as optimizer
- The result is evaluated using accuracy, recall, precision, and F1-score
- Several conditions are tested to find the best case for both 6 or 2 multi-classes classification with:
    * number of epoch from 3 to 6
    * number of BERT layers to be frezed which is used is 2, 4, or 6 layers
    * BERT learning rate which is used is 2e-5 or 3e-5

## Steps

## Result
