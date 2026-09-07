# 🧬 **Biomedical NLP Sequence Classifier**

**A biomedical Natural Language Processing project for identifying abbreviations and their corresponding long forms using sequence classification, neural networks and a FastAPI inference service.**

**The project covers the complete workflow from NLP data exploration and model experimentation through model selection, API development, automated testing, load testing, monitoring and CI/CD.**

---

## 🎯 **Project Overview**

**The objective is to identify and classify abbreviations and their corresponding long forms within biomedical text using the BIO labelling scheme.**

**The project uses the PLOD-CW dataset, which contains annotated biomedical text derived from PLOS scientific literature.**

**The classification task uses BIO-style labels to identify token positions associated with abbreviations and long forms.**

**The project explores multiple text representations, neural network architectures, optimisation strategies and hyperparameter configurations before selecting a model for deployment.**

---

## 🧠 **NLP Problem**

**Biomedical literature contains large amounts of specialised terminology and abbreviations.**

**Automatically identifying abbreviations and their corresponding long forms can support information extraction, literature analysis and biomedical text processing.**

**This project approaches the problem as a sequence classification task using token-level BIO labels.**

**The primary labels include:**

- **`B-AC` — Beginning of an abbreviation**
- **`B-LF` — Beginning of a long form**
- **`I-LF` — Inside a long form**
- **`B-O` — Other / outside the target entity**

---

## 📚 **Dataset**

**The project uses the PLOD-CW dataset hosted on Hugging Face.**

**Dataset source:**

**https://huggingface.co/datasets/surrey-nlp/PLOD-CW**

**The dataset contains approximately 50,000 annotated tokens organised in parquet format.**

**The main fields used during the analysis include:**

- **`tokens`**
- **`pos_tags`**
- **`ner_tags`**

**The dataset provides token-level linguistic information and BIO-style annotations for abbreviation and long-form identification.**

---

## 🧹 **Data Preprocessing**

**The notebook begins by loading the training and test datasets from parquet files.**

**The preprocessing workflow includes:**

- **Dataset structure verification**
- **Shape and dimension analysis**
- **Column inspection**
- **Data-type inspection**
- **Missing-value analysis**
- **Missing-value visualisation**
- **Unique-value analysis**
- **Token analysis**
- **POS-tag analysis**
- **NER-tag analysis**

**The analysis confirms the structure and completeness of the data before model experimentation.**

---

## 🔎 **Exploratory Data Analysis**

**Exploratory analysis was performed to understand the linguistic characteristics of the dataset.**

### **🔤 Token Analysis**

**The unique tokens were examined to understand lexical diversity within the biomedical corpus.**

### **🏷️ POS-Tag Analysis**

**Part-of-speech tags were analysed to understand the grammatical composition of the dataset.**

**The analysis identified frequent categories such as nouns and punctuation.**

### **🏷️ NER-Tag Analysis**

**NER labels were examined to understand the distribution of entity-related categories.**

### **☁️ Word Clouds**

**Word clouds were generated to visualise frequently occurring tokens, POS tags and NER labels.**

### **📏 Sequence Length Analysis**

**Token sequence lengths were analysed to understand the distribution of text sequence sizes.**

**The observed distribution was right-skewed, with most sequences concentrated around shorter lengths and fewer longer sequences.**

### **📐 Long-Form Length Analysis**

**The lengths of token, POS and NER sequences were compared to understand differences in linguistic and entity-span structure.**

### **🔗 Feature Relationship Analysis**

**Pair plots were used to examine relationships between token, POS-tag and NER-tag sequence lengths.**

---

## 🏷️ **Label Processing**

**The NER labels were transformed using `MultiLabelBinarizer` to create a machine-learning-compatible representation.**

**The binarizer was fitted using the training labels and then applied consistently to the test labels.**

**Label distributions were also visualised to understand class frequency and imbalance.**

**The `O` label represents the majority of tokens outside the target entities, while abbreviation and long-form labels occur less frequently.**

---

## 🤖 **Model Experimentation**

**Multiple experiments were conducted to investigate the effect of text representation, neural network architecture, loss functions, optimisers and hyperparameter configurations.**

---

## 🔤 **Experiment 1 — Text Vectorisation**

**Recurrent Neural Networks were evaluated using different text representation techniques.**

### **TF-IDF + RNN**

**TF-IDF was used to represent the text using the top 5,000 features.**

**The resulting representation was reshaped for processing by a SimpleRNN architecture.**

**The RNN used two recurrent layers with 128 and 64 units.**

### **Word2Vec + RNN**

**Pre-trained Word2Vec embeddings were used to represent tokens as 300-dimensional vectors.**

**Sequences were padded to a length of 100 tokens before being passed to the RNN.**

### **GloVe + RNN**

**Pre-trained GloVe embeddings were used to create 100-dimensional word representations.**

**The sequences were padded to a common length before being processed by the RNN architecture.**

---

## 🧠 **Experiment 2 — NLP Architectures**

**TF-IDF representations were used to compare different neural network architectures.**

### **CNN**

**A convolutional architecture was evaluated using TF-IDF representations.**

**The model included Conv1D layers, max pooling, global max pooling, dense layers and dropout regularisation.**

### **FFNN**

**A Feedforward Neural Network was evaluated using TF-IDF representations.**

**The architecture included dense layers with 512, 256 and 128 units together with dropout regularisation.**

**The final layer used sigmoid activation for the multi-label classification setup.**

### **BiLSTM**

**A Bidirectional Long Short-Term Memory architecture was evaluated to capture contextual information from both directions of the text representation.**

**The architecture used stacked bidirectional LSTM layers followed by a sigmoid output layer.**

---

## ⚙️ **Experiment 3 — Loss Functions and Optimisers**

**The BiLSTM architecture was further evaluated using different loss functions and optimisation strategies.**

### **Binary Focal Loss + Nadam**

**Binary Focal Loss was evaluated with the Nadam optimiser to place greater emphasis on difficult examples and address class imbalance.**

### **Binary Cross Entropy + Adam**

**Binary Cross Entropy was evaluated with the Adam optimiser as an alternative training configuration.**

**The experiments compared how different optimisation strategies affected classification performance and minority-class behaviour.**

---

## 🔧 **Experiment 4 — Hyperparameter Optimisation**

**Hyperparameter optimisation was performed to improve the BiLSTM model.**

### **Grid Search**

**Grid Search systematically evaluated combinations of:**

- **Optimiser**
- **LSTM units**
- **Activation function**
- **Dropout rate**
- **Batch size**
- **Number of epochs**

**The best reported configuration achieved a weighted F1 score of `0.9234`.**

**The selected configuration used:**

- **Nadam optimiser**
- **128 LSTM units**
- **Sigmoid activation**
- **0.5 dropout**
- **Batch size of 32**
- **20 epochs**

### **Random Search**

**Random Search explored a broader hyperparameter space by sampling different configurations.**

**The best reported configuration achieved a weighted F1 score of `0.9232`.**

**The configuration included:**

- **Nadam optimiser**
- **187 LSTM units**
- **Sigmoid activation**
- **Approximately 0.288 dropout**
- **Batch size of 32**
- **42 epochs**
- **Learning rate of approximately 0.0013**

**The results demonstrate that hyperparameter optimisation improved the reported F1 performance compared with earlier configurations.**

---

## 📊 **Model Evaluation**

**The experiments were evaluated using accuracy and F1-based metrics, with particular attention to performance across different BIO labels.**

### **Vectorisation Comparison**

| **Model** | **Accuracy** | **Weighted F1** |
|---|---:|---:|
| **RNN + TF-IDF** | **66.01%** | **0.9209** |
| **RNN + Word2Vec** | **59.48%** | **0.8889** |
| **RNN + GloVe** | **63.40%** | **0.9089** |

**TF-IDF produced the strongest weighted F1 score among the three evaluated vectorisation approaches.**

---

## 🧠 **Architecture Comparison**

| **Architecture** | **Accuracy** | **Weighted F1** |
|---|---:|---:|
| **CNN + TF-IDF** | **65.36%** | **0.9032** |
| **FFNN + TF-IDF** | **64.71%** | **0.9173** |
| **BiLSTM + TF-IDF** | **60.13%** | **0.9169** |

**The FFNN provided a strong balance between accuracy and weighted F1 within the TF-IDF architecture comparison.**

---

## 🏆 **Selected Deployment Model**

**The deployed model is the Feedforward Neural Network using TF-IDF representations.**

**The individual evaluation reported an accuracy of approximately 86% and a micro-average F1 score of `0.8646` for the selected FFNN configuration.**

**The reported weighted F1 score for the selected model was `0.8628`.**

**The model was selected as the deployment candidate and saved as:**

**`ffnn_tfidf_model.h5`**

**The associated preprocessing and label components were saved as:**

- **`tfidf_vectorizer.pkl`**
- **`label_encoder.pkl`**

---

## 🔬 **Error Analysis**

**The classification results show stronger performance on common labels than on less frequent abbreviation and long-form labels.**

**The analysis identified lower precision and recall for:**

- **`B-AC`**
- **`B-LF`**

**The project also identified examples where tokens such as `GEMS` and `ventilated` were incorrectly predicted as `B-O`.**

**These errors highlight the difficulty of distinguishing less frequent entity categories within biomedical text.**

---

## 🚀 **FastAPI Inference Service**

**The trained FFNN model is exposed through a FastAPI web service.**

**The application loads:**

- **TensorFlow FFNN model**
- **TF-IDF vectorizer**
- **Label encoder**

**The service exposes a POST endpoint:**

**`/predict`**

**The endpoint accepts JSON containing a text field.**

### **Example Request**

```json
{
  "text": "EPI = Echo planar imaging ."
}
```

### **Example API Structure**

```text
POST /predict
        │
        ▼
Receive text
        │
        ▼
Tokenise input
        │
        ▼
TF-IDF transformation
        │
        ▼
FFNN prediction
        │
        ▼
Decode predicted class
        │
        ▼
Return JSON predictions
```

**The FastAPI application also validates the input using Pydantic and returns an HTTP error when prediction processing fails.**

---

## 🧪 **API Testing**

**A dedicated `api_testing.ipynb` notebook was created to test the FastAPI endpoint using HTTP POST requests.**

**The testing workflow verifies:**

- **HTTP request handling**
- **Response status**
- **JSON response structure**
- **Prediction output**
- **Error responses**

**Example biomedical text inputs include:**

**`EPI = Echo planar imaging`**

**`NER EPI stem cell transplantation (SCT)`**

**The testing notebook confirms that the API returns labelled predictions for representative inputs.**

---

## ✅ **Automated Testing**

**The repository includes `test_service.py` for automated API testing using Pytest.**

**The test starts the FastAPI service using Uvicorn, sends a POST request to `/predict`, verifies a successful HTTP response and checks that prediction results are returned.**

**The test uses:**

```json
{
  "text": "EPI = Echo planar imaging ."
}
```

**The test verifies that:**

- **The response status code is `200`**
- **The response contains `predictions`**
- **At least one prediction is returned**

---

## 📈 **Performance Testing**

**Locust was used to evaluate the `/predict` endpoint under concurrent load.**

**The reported test configuration included:**

- **100 simulated users**
- **10 users spawned per second**
- **10-minute test duration**

**The evaluation focused on:**

- **Median response time**
- **95th percentile response time**
- **Maximum response time**
- **Request failures**

**The reported median response time was approximately 12 seconds, while the 95th percentile response time was approximately 22 seconds.**

**These results provide a baseline for identifying performance and scalability improvements in future versions.**

---

## 📝 **Monitoring and Logging**

**The FastAPI service includes basic application logging.**

**Prediction requests and generated predictions are recorded in the application log.**

**This provides a simple mechanism for observing API activity and investigating runtime behaviour.**

**The generated `app.log` file is intentionally excluded from the repository because it is a runtime artifact rather than source code.**

---

## 🔄 **CI/CD**

**The project includes a GitLab CI/CD configuration with three stages:**

- **Build**
- **Test**
- **Deploy**

**The pipeline creates a Python virtual environment and installs the project dependencies.**

**The test stage starts the FastAPI application and executes the Pytest suite.**

**The deployment stage starts the FastAPI application using Uvicorn according to the configured pipeline.**

**The repository also includes `manual_deploy.sh` for manually setting up the environment, installing dependencies, starting the API service and running the tests.**

---

## 🏗️ **Project Architecture**

```text
                         Biomedical Text
                               │
                               ▼
                     ┌───────────────────┐
                     │   TF-IDF Vectorizer│
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │   FFNN Classifier │
                     │   TensorFlow/Keras│
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │   Label Encoder   │
                     └─────────┬─────────┘
                               │
                               ▼
                     ┌───────────────────┐
                     │    FastAPI API    │
                     │    POST /predict  │
                     └─────────┬─────────┘
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
              Pytest        Locust      API Notebook
                 │             │             │
                 └─────────────┼─────────────┘
                               ▼
                         CI/CD Pipeline
```

---

## 🛠️ **Technology Stack**

### **Programming**

- **Python**

### **NLP & Machine Learning**

- **TensorFlow**
- **Scikit-learn**
- **TF-IDF**
- **RNN**
- **CNN**
- **FFNN**
- **BiLSTM**
- **Word2Vec**
- **GloVe**

### **API & Serving**

- **FastAPI**
- **Uvicorn**
- **Pydantic**

### **Testing**

- **Pytest**
- **Requests**
- **Locust**

### **Deployment & Automation**

- **GitLab CI/CD**
- **Bash**
- **Python Virtual Environments**

---

## 📁 **Repository Structure**

```text
biomedical-nlp-sequence-classifier/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── 6838689_Kishore_NLP.ipynb
├── api_testing.ipynb
│
├── app.py
├── test_service.py
├── locustfile.py
├── manual_deploy.sh
├── .gitlab-ci.yml
│
├── ffnn_tfidf_model.h5
├── tfidf_vectorizer.pkl
└── label_encoder.pkl
```

---

## ▶️ **Running the Project**

### **1. Install Python**

**Use a compatible Python environment for the dependencies listed in `requirements.txt`.**

### **2. Install Dependencies**

```bash
python -m venv venv
```

```bash
source venv/bin/activate
```

**On Windows:**

```text
venv\Scripts\activate
```

**Install the required packages:**

```bash
pip install -r requirements.txt
```

### **3. Start the API**

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

**The API will be available locally at:**

**`http://127.0.0.1:8000`**

### **4. Test the API**

**Open the FastAPI interactive documentation at:**

**`http://127.0.0.1:8000/docs`**

**Send a POST request to `/predict` using a biomedical text input.**

---

## 🧪 **Running Automated Tests**

**With the FastAPI service running, execute:**

```bash
pytest
```

**The automated test verifies that the `/predict` endpoint responds successfully and returns predictions.**

---

## 📈 **Running Load Tests**

**The repository includes a Locust configuration for sending repeated prediction requests to the `/predict` endpoint.**

**The test behaviour sends requests containing biomedical abbreviation and long-form examples.**

---

## 📓 **Notebook**

**`6838689_Kishore_NLP.ipynb` contains the main NLP experimentation workflow.**

**The notebook covers:**

- **Data preprocessing**
- **Exploratory data analysis**
- **Data visualisation**
- **TF-IDF**
- **Word2Vec**
- **GloVe**
- **RNN**
- **CNN**
- **FFNN**
- **BiLSTM**
- **Loss-function experimentation**
- **Optimiser experimentation**
- **Grid Search**
- **Random Search**
- **Model evaluation**
- **Error analysis**

---

## 📌 **Key Learning Outcomes**

**This project provided practical experience across the complete machine learning lifecycle.**

**Key areas include:**

- **Biomedical NLP**
- **Sequence classification**
- **BIO labelling**
- **Text vectorisation**
- **Neural network experimentation**
- **Model comparison**
- **Hyperparameter optimisation**
- **Error analysis**
- **Model serialisation**
- **REST API development**
- **Automated API testing**
- **Load testing**
- **Application logging**
- **CI/CD automation**

---

## ⚠️ **Dataset and Model Notes**

**The original PLOD-CW dataset is not included in this repository.**

**The trained model and preprocessing artefacts are included so that the inference service can be run without retraining the model.**

**The coursework reports and original academic submission documents are intentionally excluded from the repository.**

---

## 🎓 **Academic Context**

**Developed as part of the MSc Data Science / Natural Language Processing coursework at the University of Surrey.**

**The project combines NLP experimentation with practical model serving and software engineering, demonstrating the transition from an experimental machine learning workflow to a testable API-based application.**
