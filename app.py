from dotenv import load_dotenv
import os
import streamlit as st
from streamlit_extras.stoggle import stoggle
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
#from langchain.callbacks import get_openai_callback


def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Kevin's PDF AI 💬")
    stoggle(
    "Click me!",
    """I am a Large Language Model (LLM) trained to analyze PDF files 📚 """)

    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    # extract text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()


        # split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # create embeddings
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # show user input
        user_question = st.text_input("Ask me a question about your PDF:", 
                                      placeholder='"Please provide a concise summary of the main ideas and key findings in this doc"') # "Please provide a concise summary of the main ideas, key findings, and important points from the PDF, along with any relevant context, figures, or examples."
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            # with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)
              #print(cb)

            st.write(response)


if __name__ == '__main__':
    main()