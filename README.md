# Disaster Tweet Classification Project  
   
This project aims to classify tweets related to disasters using a machine learning model based on the DistilBERT architecture. The primary objective is to determine whether a given tweet is about a real disaster (target=1) or not (target=0).  
   
## Table of Contents  
- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Training the Model](#training-the-model)  
- [Evaluation](#evaluation)  
- [Prediction](#prediction)  
- [Gradio UI](#gradio-ui)  
- [Results](#results)  
- [Contributing](#contributing)  
- [License](#license)  
   
## Project Overview  
This project uses the DistilBERT model from Hugging Face's Transformers library for sequence classification. The steps involved include data preprocessing, tokenization, model training, evaluation, and prediction. Additionally, a Gradio UI is provided for testing the model interactively.  
   
## Dataset  
The dataset consists of tweets labeled as either related to disasters or not. The dataset is split into training, validation, and test sets. The training set is used to train the model, the validation set is used for hyperparameter tuning, and the test set is used for evaluating the model's performance.  
   
## Installation  
To install the required dependencies, run the following commands:  
   
```bash  
git clone https://github.com/utsav-desai/CS772.git  
pip install accelerate -U datasets  
pip install transformers[torch]  
pip install spacy_cleaner  
pip install gradio  
```  
   
## Training the Model  
To train the DistilBERT model, follow these steps:  
   
1. Preprocess the data using the `spacy_cleaner` library.  
2. Tokenize the text data using the `AutoTokenizer` from Hugging Face.  
3. Train the DistilBERT model using the `Trainer` and `TrainingArguments` classes.  
   
## Evaluation  
The model's performance is evaluated using accuracy and F1-score metrics. The evaluation is performed on the validation set.  
   
## Prediction  
To make predictions on new data, the trained model is used to classify tweets. The model outputs a label indicating whether the tweet is related to a disaster or not.  
   
## Gradio UI  
A Gradio UI is provided for testing the model interactively. The UI allows users to input a text sentence and get the model's prediction.  
   
To launch the Gradio UI, run the following code:  
   
```python  
import gradio as gr  
   
def classify_text(text):  
    # Preprocess the text  
    text = text.lower()  
    text = re.sub("[#=><\/.]", "", text)  
    text = re.sub("@\w+", "", text)  
      
    # Tokenize the text  
    tokenized_text = tokenizer(text)  
      
    # Convert the tokenized text to a tensor  
    input_ids = torch.tensor(tokenized_text["input_ids"]).unsqueeze(0)  
    attention_mask = torch.tensor(tokenized_text["attention_mask"]).unsqueeze(0)  
      
    # Load the model  
    model = AutoModelForSequenceClassification.from_pretrained("my_model_weights")  
      
    # Make predictions  
    outputs = model(input_ids, attention_mask=attention_mask)  
    predictions = torch.argmax(outputs.logits, dim=-1)  
      
    # Return the predictions  
    return predictions.item()  
   
demo = gr.Interface(  
    fn=classify_text,  
    inputs=gr.Textbox(label="Enter a text sentence here"),  
    outputs="label",  
    examples=[  
        "This is a disaster!",  
        "Earthquake is expected in China!",  
        "I'm feeling happy.",  
    ],  
)  
demo.launch()  
```  
   
## Results  
The model achieved an accuracy of 98.68% and an F1-score of 98.48% on the validation set. The performance metrics indicate that the model is effective in classifying tweets related to disasters
