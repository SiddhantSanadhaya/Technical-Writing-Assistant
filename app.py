import sys
# import pysqlite3 as sqlite3
# sys.modules["sqlite3"] = sqlite3
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
    initial_sidebar_state="expanded"
)


# ============= Styles =============

# Enhanced Custom CSS with modern styling
st.markdown("""
    <style>
        /* Global Styles */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }

        /* Header styling */
        .fixed-header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 999;
            background: linear-gradient(135deg, #1e3799, #0c2461);
            color: white;
            padding: 1.5rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }

        /* Title styling */
        .title {
            text-align: center;
            font-size: 2rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: white;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
        }
        
        .subtitle {
            text-align: center;
            font-size: 1rem;
            color: rgba(255,255,255,0.8);
            margin-bottom: 1rem;
        }

        /* Main content area */
        .main-content {
            margin-top: 180px;
            padding: 2rem;
            max-width: 1200px;
            margin-left: auto;
            margin-right: auto;
        }

        /* Chat message styling */
        .stChatMessage {
            background-color: #f8f9fa;
            border-radius: 15px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            transition: transform 0.2s ease;
        }

        .stChatMessage:hover {
            transform: translateY(-2px);
        }

        /* Sidebar styling */
        .css-1d391kg {
            background-color: #f8f9fa;
        }

        .streamlit-expanderHeader {
            background-color: white;
            border-radius: 10px;
        }

        /* Button styling */
        .stButton>button {
            background: linear-gradient(135deg, #1e3799, #0c2461);
            color: white;
            border: none;
            border-radius: 25px;
            padding: 0.5rem 1.5rem;
            font-weight: 500;
            transition: all 0.3s ease;
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }

        /* Select box styling */
        .stSelectbox {
            border-radius: 10px;
        }

        .stSelectbox > div > div {
            background-color: white;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
        }

        /* Chat input styling */
        .stTextInput > div > div > input {
            border-radius: 25px;
            border: 2px solid #e0e0e0;
            padding: 1rem;
            font-size: 1rem;
            transition: all 0.3s ease;
        }

        .stTextInput > div > div > input:focus {
            border-color: #1e3799;
            box-shadow: 0 0 0 2px rgba(30,55,153,0.1);
        }

        /* Hide default Streamlit elements */
        MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Error message styling */
        .stAlert {
            background-color: #ffe4e4;
            border-color: #ff0000;
            color: #ff0000;
            border-radius: 10px;
            padding: 1rem;
            margin: 1rem 0;
        }

        /* Responsive design */
        @media screen and (max-width: 768px) {
            .main-content {
                padding: 1rem;
            }
            
            .title {
                font-size: 1.5rem;
            }
            
            .subtitle {
                font-size: 0.875rem;
            }
        }
    </style>
""", unsafe_allow_html=True)

# ============= Layout Components =============

# Create enhanced header with subtitle
st.markdown("""
    <div class="fixed-header">
        <div class="title">Technical Writing Assistant</div>
        <div class="subtitle">Create Professional Documentation Effortlessly</div>
    </div>
""", unsafe_allow_html=True)


# Enhanced sidebar with better organization
with st.sidebar:
    st.title("⚙️ Settings")
    
    with st.expander("Writing Preferences", expanded=True):
        writing_format = st.selectbox(
            "📝 Writing Format",
            ["None", "concise", "concise-tabular", "detailed", "procedural"],
            index=0,
            key="writing_format",
            help="Select the writing style for your documentation"
        )
        
        output_format = st.selectbox(
            "📄 Output Format",
            [".md", ".xlsx", ".docx"],
            index=2,
            key="output_format",
            help="Choose the file format for your output"
        )
    
    with st.expander("About", expanded=True):
        st.markdown("""
            ### 🤖 How to Use
            1. Select your preferred writing format
            2. Choose your desired output format
            3. Type your request in the chat
            4. Get professionally written technical documentation
            
            ### 📚 Features
            - Multiple writing styles
            - Various output formats
            - Real-time documentation generation
        """)
    # if os.path.exists(logo_path):
    #     st.image(logo_path, width=200, caption="Created with ❤️ by StatusNeo")
    # else:
    #     st.error("Logo image not found. Please check the path.")



# Add spacing for fixed header
st.markdown('<div class="main-content"></div>', unsafe_allow_html=True)

#
# Your existing setup
key = os.environ['key']
db_path = "path_to_saved_db_gpt_4_chunk_500_new"

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


# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize other components
db = get_db()
@st.cache_resource
def get_llm():
    llm = ChatOpenAI(model="gpt-4o", openai_api_key=key, temperature=0)
    return llm
llm = get_llm()

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
4. Follow *Important Guidelines* and *Writing Format* strictly.
5. Then produce the output according to *Output Format* strictly.


*Important Guidelines*:
- **Do not include any confidential or internal information**, such as Jira ticket numbers, system logs, error codes, or issue numbers, in the response.
- **If the input request asks for correcting existing user guide content, simply provide the revised text without highlighting or mentioning the errors in the user guide.**
- If the Jira Ticket Request lacks sufficient information to complete the task, respond with a list of specific questions or clarifications needed to proceed.

*Writing Format*:
- **Concise**: If `{writing_format}` is "concise", provide a brief and focused response, highlighting only key points.
- **Concise-Tabular**: If `{writing_format}` is "concise-tabular", present a brief response in a tabular format, emphasizing key points.
- **Detailed**: If `{writing_format}` is "detailed", provide an in-depth explanation with examples, background information, and context. Focus on the 'why' and 'what' of the topic.
- **Procedural**: If `{writing_format}` is "procedural", in numbered index only (not bullets or anything) provide step-by-step instructions or a sequence of list of actions in a **Concise** writing_format. Use clear action verbs.

*Output Format*:                     
- **Markdown (.md)**: If `{output_format}` is ".md", format the response in .md Markdown format for easy integration into a markdown file. Use list formatting for steps if the procedural style is selected.
- **Excel (.xlsx)**: If `{output_format}` is ".xlsx", present the information in a structured format suitable for Excel sheet for easy integration into sheet, with clear headings and data organization.
- **Word (.docx)**: If `{output_format}` is ".docx", format the content appropriately for a Word document for easy integration into documentation, ensuring proper styling and layout.

Jira Ticket Request: {input}
Writing Format: {writing_format}
Output Format: {output_format}

""")



document_chain = create_stuff_documents_chain(llm, prompt)

# @st.cache_resource
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

