import google.generativeai as genai
import PIL.Image
import os
import tempfile
import streamlit as st
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

#apna api key add karo
genai_api_key =  'AIzaSyCtkHjX1sXhvEQ5_6ySoWqCHIO2xMdEsak'  

#api key ko kise variable mai dalu age use kar nai 
genai.configure(api_key=genai_api_key)

# cancer ka datast hai
@st.cache_data
def load_data():
    data = pd.read_csv(r"C:\Users\shubh\Downloads\cancerdataset.csv")  #path 
    return data

data = load_data()

# input data
X = data[['GENDER', 'AGE', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE',
          'FATIGUE_como', 'ALLERGYSS', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING',
          'SHORTNESS OF BREATH', 'CHEST PAIN', 'smoke', 'cancer_history',
          'itch', 'grew', 'hurt', 'bleed', 'creatinine', 'LYVE1', 'REG1B', 'TFF1']]

# output data
y = data[['LUNG_CANCER', 'skin cancer', 'pancrise']]

# Handle missing values
X['smoke'].fillna(X['smoke'].mode()[0], inplace=True)
X['cancer_history'].fillna(X['cancer_history'].mode()[0], inplace=True)

# Label Encoding for binary categorical columns
binary_columns = ['GENDER', 'itch', 'grew', 'hurt', 'bleed', 'smoke', 'cancer_history']
label_encoder = LabelEncoder()
for column in binary_columns:
    X[column] = label_encoder.fit_transform(X[column])

# Label encode the target columns
for column in y.columns:
    y[column] = label_encoder.fit_transform(y[column])

# Standardize the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the neural network model with 3 binary outputs
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),  # Input layer
    tf.keras.layers.Dense(64, activation='relu'),  # Hidden layer
    tf.keras.layers.Dropout(0.3),  # Dropout for regularization
    tf.keras.layers.Dense(32, activation='relu'),  # Another hidden layer
    tf.keras.layers.Dropout(0.3),  # Dropout for regularization
    tf.keras.layers.Dense(3, activation='sigmoid')  # 3 output neurons for 3 binary classifications
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
@st.cache_resource
def train_model():
    model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2, verbose=1)
    return model

# Train or load trained model
if "trained_model" not in st.session_state:
    trained_model = train_model()
    st.session_state.trained_model = trained_model
else:
    trained_model = st.session_state.trained_model

# Streamlit app for MALIGNANT.AI
st.title("MALIGNANT.AI")
st.markdown("AI-driven system designed to detect early-stage cancer, enhancing early diagnosis and improving treatment outcomes.")

# Sidebar for Navigation
st.sidebar.title("Navigation")
page_selection = st.sidebar.radio("Choose a section:", ("Text Query", "Image Query", "Document analysis", "Cancer Prediction"))

# Section 1: Text Query with Google GenAI (Gemini)
if page_selection == "Text Query":
    st.header("Text Query with MALIGNANT.AI")

    # Input for the user to ask questions
    query_text = st.text_input("Enter the question you want to ask:")

    # Function to interact with Gemini API
    def gemini_model(prompt):
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([prompt])
        return response

    # Button to submit the question
    if st.button("Ask MALIGNANT.AI"):
        if query_text:
            with st.spinner("Processing your query..."):
                response = gemini_model(query_text)
                st.success("Response from MALIGNANT.AI:")
                st.write(response.text)
        else:
            st.warning("Please enter a question to ask MALIGNANT.AI.")

# Section 2: Image Query using Google GenAI (Gemini)
elif page_selection == "Image Query":
    st.header("Image Query with MALIGNANT.AI")
    image_query = st.text_input("What do you want to know about the image:")
    uploaded_image = st.file_uploader("Upload your image", type=["png", "jpg", "jpeg"])

    if uploaded_image is not None:
        img = PIL.Image.open(uploaded_image)
        st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Ask MALIGNANT.AI about the Image"):
        if image_query and uploaded_image:
            with st.spinner("Analyzing the image..."):
                model = genai.GenerativeModel(model_name="gemini-1.5-flash")
                response = model.generate_content([image_query])
                st.success("Response:")
                st.write(response.text)
        else:
            st.warning("Please upload an image and provide a question.")

# Section 3: Document Embedding with FAISS
elif page_selection == "Document analysis":
    st.header("Document Analysis")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.loader = PyPDFLoader(tmp_file.name)
        st.success("PDF loaded successfully.")

    def vector_embedding():
        if "vectors" not in st.session_state:
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            embeddings = embedding_model.embed_documents([doc.page_content for doc in st.session_state.final_documents])

            # FAISS Index setup
            index = faiss.IndexFlatL2(len(embeddings[0]))
            index.add(np.array(embeddings))
            st.session_state.vectors = (index, embeddings)
            st.success("Document embeddings generated and indexed successfully.")

    if st.button("Generate Document Embeddings"):
        vector_embedding()

    query_text_doc = st.text_input("Enter your question to search within the document:")

    if query_text_doc:
        if "vectors" in st.session_state:
            index, embeddings = st.session_state.vectors
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            query_embedding = embedding_model.embed_query(query_text_doc)

            # Search for the closest vectors/documents
            D, I = index.search(np.array([query_embedding]), k=5)
            response_documents = [st.session_state.final_documents[i] for i in I[0]]

            st.success("Documents found based on your query:")
            for i, doc in enumerate(response_documents):
                st.write(f"Document {i+1}:")
                st.write(doc.page_content)
                st.write("--------------------------------")
        else:
            st.warning("Please generate document embeddings first.")

# Section 4: Cancer Prediction based on user input
# Section 4: Cancer Prediction based on user input
elif page_selection == "Cancer Prediction":
    st.header("Cancer Prediction")

    def get_user_input():
        # Gender selection with descriptive labels
        GENDER = st.selectbox("GENDER", ("Female", "Male"), help="Select your gender")
        GENDER = 1 if GENDER == "Male" else 0

        # Age input
        AGE = st.number_input("AGE", min_value=0, help="Enter your age")

        # Binary inputs with descriptive Yes/No labels
        YELLOW_FINGERS = st.selectbox("YELLOW FINGERS", ("No", "Yes"), help="Do you have yellow fingers?")
        YELLOW_FINGERS = 1 if YELLOW_FINGERS == "Yes" else 0

        ANXIETY = st.selectbox("ANXIETY", ("No", "Yes"), help="Do you experience anxiety?")
        ANXIETY = 1 if ANXIETY == "Yes" else 0

        PEER_PRESSURE = st.selectbox("PEER PRESSURE", ("No", "Yes"), help="Do you experience peer pressure?")
        PEER_PRESSURE = 1 if PEER_PRESSURE == "Yes" else 0

        CHRONIC_DISEASE = st.selectbox("CHRONIC DISEASE", ("No", "Yes"), help="Do you have any chronic diseases?")
        CHRONIC_DISEASE = 1 if CHRONIC_DISEASE == "Yes" else 0

        FATIGUE_como = st.selectbox("FATIGUE", ("No", "Yes"), help="Do you experience fatigue?")
        FATIGUE_como = 1 if FATIGUE_como == "Yes" else 0

        ALLERGYSS = st.selectbox("ALLERGIES", ("No", "Yes"), help="Do you have allergies?")
        ALLERGYSS = 1 if ALLERGYSS == "Yes" else 0

        WHEEZING = st.selectbox("WHEEZING", ("No", "Yes"), help="Do you experience wheezing?")
        WHEEZING = 1 if WHEEZING == "Yes" else 0

        ALCOHOL_CONSUMING = st.selectbox("ALCOHOL CONSUMING", ("No", "Yes"), help="Do you consume alcohol?")
        ALCOHOL_CONSUMING = 1 if ALCOHOL_CONSUMING == "Yes" else 0

        COUGHING = st.selectbox("COUGHING", ("No", "Yes"), help="Do you experience coughing?")
        COUGHING = 1 if COUGHING == "Yes" else 0

        SHORTNESS_OF_BREATH = st.selectbox("SHORTNESS OF BREATH", ("No", "Yes"), help="Do you experience shortness of breath?")
        SHORTNESS_OF_BREATH = 1 if SHORTNESS_OF_BREATH == "Yes" else 0

        CHEST_PAIN = st.selectbox("CHEST PAIN", ("No", "Yes"), help="Do you experience chest pain?")
        CHEST_PAIN = 1 if CHEST_PAIN == "Yes" else 0

        smoke = st.selectbox("Do you smoke?", ("No", "Yes"), help="Do you smoke?")
        smoke = 1 if smoke == "Yes" else 0

        cancer_history = st.selectbox("Do you have a family history of cancer?", ("No", "Yes"), help="Do you have a family history of cancer?")
        cancer_history = 1 if cancer_history == "Yes" else 0

        # More features...
        itch = st.selectbox("Do you have skin itchiness?", ("No", "Yes"))
        itch = 1 if itch == "Yes" else 0

        grew = st.selectbox("Do you notice any abnormal skin growth?", ("No", "Yes"))
        grew = 1 if grew == "Yes" else 0

        hurt = st.selectbox("Do you feel pain on abnormal skin growth?", ("No", "Yes"))
        hurt = 1 if hurt == "Yes" else 0

        bleed = st.selectbox("Does the abnormal skin growth bleed?", ("No", "Yes"))
        bleed = 1 if bleed == "Yes" else 0

        creatinine = st.number_input("Enter your creatinine level:", min_value=0.0)
        LYVE1 = st.number_input("Enter your LYVE1 level:", min_value=0.0)
        REG1B = st.number_input("Enter your REG1B level:", min_value=0.0)
        TFF1 = st.number_input("Enter your TFF1 level:", min_value=0.0)

        # Combine all inputs into a DataFrame
        input_data = pd.DataFrame({
            'GENDER': [GENDER],
            'AGE': [AGE],
            'YELLOW_FINGERS': [YELLOW_FINGERS],
            'ANXIETY': [ANXIETY],
            'PEER_PRESSURE': [PEER_PRESSURE],
            'CHRONIC DISEASE': [CHRONIC_DISEASE],
            'FATIGUE_como': [FATIGUE_como],
            'ALLERGYSS': [ALLERGYSS],
            'WHEEZING': [WHEEZING],
            'ALCOHOL CONSUMING': [ALCOHOL_CONSUMING],
            'COUGHING': [COUGHING],
            'SHORTNESS OF BREATH': [SHORTNESS_OF_BREATH],
            'CHEST PAIN': [CHEST_PAIN],
            'smoke': [smoke],
            'cancer_history': [cancer_history],
            'itch': [itch],
            'grew': [grew],
            'hurt': [hurt],
            'bleed': [bleed],
            'creatinine': [creatinine],
            'LYVE1': [LYVE1],
            'REG1B': [REG1B],
            'TFF1': [TFF1]
        })
        return input_data

    # User input
    user_input = get_user_input()

    if st.button("Predict Cancer"):
        with st.spinner("Predicting..."):
            scaled_input = scaler.transform(user_input)
            prediction = trained_model.predict(scaled_input)
            lung_cancer_prob = prediction[0][0]
            skin_cancer_prob = prediction[0][1]
            pancreatic_cancer_prob = prediction[0][2]

            # Determine if cancer is detected
            cancer_probs = [lung_cancer_prob, skin_cancer_prob, pancreatic_cancer_prob]
            high_risk_count = sum(prob > 0.3 for prob in cancer_probs)  # Example threshold of 0.5
            st.error("Warning: This is an AI this may be wrong.")

            if high_risk_count >= 1:
                st.error("Cancer Detected!")
                st.write(f"Lung Cancer Probability: {lung_cancer_prob:.2f}")
                st.write(f"Skin Cancer Probability: {skin_cancer_prob:.2f}")
                st.write(f"Pancreatic Cancer Probability: {pancreatic_cancer_prob:.2f}")
            else:
                st.success("No Cancer Detected!")
                st.write(f"Lung Cancer Probability: {lung_cancer_prob:.2f}")
                st.write(f"Skin Cancer Probability: {skin_cancer_prob:.2f}")
                st.write(f"Pancreatic Cancer Probability: {pancreatic_cancer_prob:.2f}")
