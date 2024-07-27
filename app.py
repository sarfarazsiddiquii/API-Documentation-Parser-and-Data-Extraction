import csv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI 
import os

os.environ["GOOGLE_API_KEY"] = "AIzaSyATsu1gYkV4VbwLt3OKjN_vPS9pe5cuhfE"


def scrape_api_docs(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    return data

def split_text(data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(data)
    return splits

def save_to_csv(splits, filename="parsed_data.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Document ID", "Text"])
        for i, split in enumerate(splits):
            writer.writerow([i, split.page_content])

def create_vector_store(splits):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

def setup_qa_system(vectorstore):
    try:
        llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.2, use_context=True)  
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(),
            return_source_documents=True
        )
        return qa_chain
    except Exception as e:
        print(f"Error setting up QA system: {e}")
        return None

def generate_code_from_api_docs(api_url, query):
    data = scrape_api_docs(api_url)
    splits = split_text(data)
    save_to_csv(splits) 
    vectorstore = create_vector_store(splits)
    qa_system = setup_qa_system(vectorstore)
    
    result = qa_system({"query": f"Generate code for the following task using the API: {query}"})
    return result["result"]

if __name__ == "__main__":
    api_url = "https://nextjs.org/docs/app/building-your-application/data-fetching/fetching-caching-and-revalidating" 
    user_query = "how to fetch data"
    generated_code = generate_code_from_api_docs(api_url, user_query)
    print(generated_code)