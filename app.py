import streamlit as st
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI

# Set page config
st.set_page_config(
    page_title="Technical Writing Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS with fixed header
st.markdown("""
    <style>
        /* Header positioning */
        .fixed-header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 999;
            background-color: var(--text-color);
            padding: 20px;
            border-bottom: 1px solid #e5e5e5;
        }

        /* Main content padding to prevent overlap */
        .main-content {
            margin-top: 200px; /* Adjust based on your header height */
            padding: 20px;
        }

        /* Hide default Streamlit elements */
        MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Center title */
        .title {
            text-align: center;
            font-size: 30px;
            font-weight: bold;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Your existing setup
\key = os.environ['key']
db_path = r"E:\UPSKILL\NLP\Langchain\rag\pdf_data_chatbot\db\path_to_saved_db_gpt_4_chunk_500"

# Initialize database
@st.cache_resource
def get_db():
    embedding_model = OpenAIEmbeddings(openai_api_key=key)
    if os.path.exists(db_path):
        db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    else:
        st.error("Database not found. Please ensure the database is saved at the specified path.")
        return None
    return db

# Create fixed header HTML
st.markdown("""
    <div class="fixed-header">
        <div class="title">Technical Writing Assistant</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar selectors
st.sidebar.title("Settings")
writing_format = st.sidebar.selectbox(
    "Writing Format",
    ["None","concise", "concise-tabular", "detailed", "procedural"],
    key="writing_format"
)
output_format = st.sidebar.selectbox(
    "Output Format",
    ["None",".md", ".xlsx", ".docx"],
    key="output_format"
)

# Add spacing for fixed header
st.markdown('<div class="main-content"></div>', unsafe_allow_html=True)

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize other components
db = get_db()
llm = ChatOpenAI(model="gpt-4o", openai_api_key=key, temperature=0)

# Your existing prompt template
prompt = ChatPromptTemplate.from_template("""
You are a highly skilled assistant for technical writing, tasked with creating clear and accurate sections for a user guide based on specific requests.
Below is the existing user guide content: {context}

Your task:
1. **Review the User Guide**: First, thoroughly understand the overall features, structure, tone, and formatting of the guide.
2. **Analyze the Request**: Identify key elements in the Jira Ticket request and determine how they relate to the features and sections in the guide.

   - If the request aligns with an existing section, use that section's title as the response heading.
   - If it's new information, create a new heading that fits the guide's style and structure.

3. **Write the Content**: Draft the section in a way that maintains consistency with the guide's format, tone, and style. Ensure clarity, completeness, and readiness for direct inclusion in the guide.
4. Follow *Important Guidelines* and *Writing Format*
5. Then produce the output according to *Output Format*


*Important Guidelines*:
- **Do not include any confidential or internal information**, such as Jira ticket numbers, system logs, error codes, or issue numbers, in the response.
- **If the input request asks for correcting existing user guide content, simply provide the revised text without highlighting or mentioning the errors in the user guide.**
- If the Jira Ticket Request lacks sufficient information to complete the task, respond with a list of specific questions or clarifications needed to proceed.

*Writing Format*:
- **Concise**: If `{writing_format}` is "concise," provide a brief and focused response, highlighting only key points.
- **Concise-Tabular**: If `{writing_format}` is "concise-tabular," present a brief response in a tabular format, emphasizing key points.
- **Detailed**: If `{writing_format}` is "detailed," provide an in-depth explanation with examples, background information, and context. Focus on the 'why' and 'what' of the topic.
- **Procedural**: If `{writing_format}` is "procedural," provide step-by-step instructions or a sequence of actions in a **Concise** format. Use clear action verbs, and format the steps as needed for each `{output_format}` option.

*Output Format*:
- **Markdown (.md)**: If `{output_format}` is ".md," format the response in .md Markdown format for easy integration into a markdown file. Use list formatting for steps if the procedural style is selected.
- **Excel (.xlsx)**: If `{output_format}` is ".xlsx," present the information in a structured format suitable for Excel sheet for easy integration into sheet, with clear headings and data organization.
- **Word (.docx)**: If `{output_format}` is ".docx," format the content appropriately for a Word document for easy integration into documentation, ensuring proper styling and layout.

Jira Ticket Request: {input}
Writing Format: {writing_format}
Output Format: {output_format}

""")

document_chain = create_stuff_documents_chain(llm, prompt)

@st.cache_resource
def get_retriever():
    retriever = db.as_retriever()
    return retriever

retriever = get_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# User input
user_input = st.chat_input("How can I assist you with technical writing?")

if user_input:
    st.chat_message("user").markdown(user_input)
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    with st.spinner("Generating response..."):
        response = retrieval_chain.invoke({
            "input": user_input,
            "writing_format": writing_format,
            "output_format": output_format
        })
        answer = response['answer']

        if answer != user_input:
            st.chat_message("assistant").markdown(answer)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })
