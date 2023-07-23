import streamlit as st
import time
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def unique_extract_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def unique_split_text_into_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def unique_create_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def unique_create_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def unique_handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def unique_main():
    load_dotenv()
    st.set_page_config(page_title="ChatBot для общения с PDF документами",
                       page_icon=":computer:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("ChatBot для общения с PDF документами :computer:")
    user_question = st.text_input("Задайте вопрос о ваших документах:")
    if user_question:
        unique_handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Документы")
        pdf_docs = st.file_uploader(
            "Загрузите свои PDF-файлы сюда и нажмите на кнопку «Преобразовать»", accept_multiple_files=True)
        if st.button("Преобразовать"):
            with st.spinner("Преобразование"):
                # get pdf text
                raw_text = unique_extract_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = unique_split_text_into_chunks(raw_text)

                # create vector store
                embeddings = OpenAIEmbeddings()
                # Добавим задержку перед вызовом API
                time.sleep(1)  # Задержка в 1 секунду
                vectorstore = unique_create_vector_store(text_chunks)

                # create conversation chain
                st.session_state.conversation = unique_create_conversation_chain(
                    vectorstore)

if __name__ == '__main__':
    unique_main()

