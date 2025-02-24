import os
from fastapi import FastAPI, Query, HTTPException
import pandas as pd
import uvicorn
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API Key not found. Set OPENAI_API_KEY in .env file.")

# Load Titanic Dataset
data_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
try:
    df = pd.read_csv(data_url)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")

# Initialize FastAPI app
app = FastAPI()

# Initialize LangChain LLM
llm = OpenAI(model_name="gpt-4", openai_api_key=OPENAI_API_KEY)

# Load Documents for RetrievalQA
text_file_path = "titanic_info.txt"  # Replace with actual Titanic dataset description
if os.path.exists(text_file_path):
    loader = TextLoader(text_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
else:
    retriever = None
    qa_chain = None

# Function to convert plots to base64
def generate_plot_image(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

@app.get("/ask")
def ask_question(query: str = Query(..., title="User Question")):
    """Handles user queries and returns text response or visualizations."""
    response = ""
    image_base64 = ""
    query_lower = query.lower()
    
    try:
        if "percentage of passengers were male" in query_lower:
            male_percentage = (df['Sex'].value_counts(normalize=True).get('male', 0)) * 100
            response = f"{male_percentage:.2f}% of passengers were male."
        
        elif "histogram of passenger ages" in query_lower:
            fig, ax = plt.subplots()
            sns.histplot(df['Age'].dropna(), bins=20, kde=True, ax=ax)
            ax.set_title("Distribution of Passenger Ages")
            image_base64 = generate_plot_image(fig)
        
        elif "average ticket fare" in query_lower:
            response = f"The average ticket fare was ${df['Fare'].mean():.2f}."
        
        elif "passengers embarked from each port" in query_lower:
            fig, ax = plt.subplots()
            sns.countplot(x=df['Embarked'].dropna(), ax=ax)
            ax.set_title("Passengers per Embarkation Port")
            image_base64 = generate_plot_image(fig)
        
        elif "survival rate based on class" in query_lower:
            fig, ax = plt.subplots()
            survival_rate = df.groupby('Pclass')['Survived'].mean() * 100
            sns.barplot(x=survival_rate.index, y=survival_rate.values, ax=ax)
            ax.set_title("Survival Rate by Class (%)")
            ax.set_xlabel("Passenger Class")
            ax.set_ylabel("Survival Rate (%)")
            image_base64 = generate_plot_image(fig)
        
        elif "survival rate based on gender" in query_lower:
            fig, ax = plt.subplots()
            survival_rate = df.groupby('Sex')['Survived'].mean() * 100
            sns.barplot(x=survival_rate.index, y=survival_rate.values, ax=ax)
            ax.set_title("Survival Rate by Gender (%)")
            ax.set_xlabel("Gender")
            ax.set_ylabel("Survival Rate (%)")
            image_base64 = generate_plot_image(fig)
        
        else:
            if qa_chain:
                response = qa_chain.run(query)
            else:
                response = "I couldn't find relevant information for your query. Try asking about Titanic statistics."
    
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"Missing column in dataset: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    return {"response": response, "image": image_base64 if image_base64 else None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)