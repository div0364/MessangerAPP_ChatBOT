import streamlit as st
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Simple Q&A Chatbot With Ollama"

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the user queries."),
        ("user", "Question:{question}")
    ]
)

# Define the function to generate responses
def generate_response(question, llm, temperature, max_tokens):
    llm = OllamaLLM(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

# Streamlit app
st.title("Enhanced Q&A Chatbot With Ollama")

# Select the OpenAI model
llm = st.sidebar.selectbox("Select Open Source model", ["gemma2:2b"])

# Adjust response parameters
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150)

# Main interface for user input
st.write("Go ahead and ask any question")
user_input = st.text_input("You:")

# Generate and display the response
if user_input:
    try:
        response = generate_response(user_input, llm, temperature, max_tokens)
        st.write(response)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.write("Please provide the user input")
