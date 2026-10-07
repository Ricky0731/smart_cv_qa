# import os
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_chroma import Chroma
# from langchain_groq import ChatGroq
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# import sentence_transformers

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)

# def build_rag(pdf_path, query):
#     print("1. Loading document...")
#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()

#     print("2. Splitting text...")
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splits = text_splitter.split_documents(docs)

#     print("3. Creating embeddings and vector store...")
#     embeddings = HuggingFaceEmbeddings(model_name="openai/gpt-oss-120b")
#     vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    
#     print("4. Setting up retriever and LLM...")
#     retriever = vectorstore.as_retriever()
#     llm = ChatGroq(model="llama3-8b-8192", temperature=0)

#     print("5. Creating chain...")
#     system_prompt = (
#         "You are an assistant for question-answering tasks. "
#         "Use the following pieces of retrieved context to answer the question. "
#         "If you don't know the answer, say that you don't know. "
#         "Use three sentences maximum and keep the answer concise."
#         "\n\n"
#         "{context}"
#     )

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ])
    
#     # Modern LCEL approach
#     rag_chain = (
#         {"context": retriever | format_docs, "input": RunnablePassthrough()}
#         | prompt
#         | llm
#         | StrOutputParser()
#     )

#     print("6. Invoking chain...")
#     response = rag_chain.invoke(query)
#     return response

# if __name__ == "__main__":
#     # Path to the PDF
#     pdf_file = "Sai Tulasi CV.pdf"
    
#     # Using the key you pasted in the file
#     groq_key = "GROQ_API_KEY"
#     os.environ["GROQ_API_KEY"] = groq_key
        
#     test_query = "What is the summary of this CV?"
    
#     try:
#         answer = build_rag(pdf_file, test_query)
#         print("\nQuery:", test_query)
#         print("Answer:", answer)
#     except Exception as e:
#         print(f"An error occurred: {e}")


import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ✅ Load environment variables
load_dotenv()

def format_docs(docs):
    """Format retrieved documents for context"""
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag(pdf_path, query):
    """
    Build and invoke RAG pipeline for resume Q&A
    """
    print("=" * 70)
    print("Resume RAG Pipeline - LCEL Approach")
    print("=" * 70)
    
    # 1. Load document
    print("\n1. Loading document...")
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        print(f"   ✓ Loaded {len(docs)} pages from '{pdf_path}'")
    except FileNotFoundError:
        print(f"   ✗ Error: File '{pdf_path}' not found")
        return None

    # 2. Split text
    print("\n2. Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    print(f"   ✓ Created {len(splits)} chunks")

    # 3. Create embeddings and vector store
    print("\n3. Creating embeddings and vector store...")
    # ✅ FIXED: Use correct embedding model (27MB only)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("   ✓ Vector store created and persisted")
    
    # 4. Setup retriever and LLM
    print("\n4. Setting up retriever and LLM...")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # ✅ FIXED: Get API key from environment
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("   ✗ Error: GROQ_API_KEY not found in environment")
        print("   Please set it in .env file: GROQ_API_KEY=your_key")
        return None
    
    # ✅ FIXED: Use current working model
    llm = ChatGroq(
        model="openai/gpt-oss-120b",  # ✅ Current working model
        temperature=0.7,
        groq_api_key=groq_api_key
    )
    print("   ✓ Retriever and LLM initialized")

    # 5. Create prompt template
    print("\n5. Creating prompt template...")
    system_prompt = (
        "You are an expert resume analyst assistant. "
        "Your task is to answer questions about the provided resume/CV. "
        "Use ONLY the information from the provided resume context. "
        "If the answer is not in the resume, clearly state that. "
        "Format your answer as bullet points when listing multiple items. "
        "Keep answers concise and relevant. "
        "\n\n"
        "Resume Context:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    print("   ✓ Prompt template created")
    
    # 6. Create LCEL chain
    print("\n6. Building LCEL chain...")
    # ✅ Modern LCEL approach
    rag_chain = (
        {
            "context": retriever | format_docs,  # Retrieve and format docs
            "input": RunnablePassthrough()        # Pass through the query
        }
        | prompt                                  # Apply prompt template
        | llm                                     # Send to LLM
        | StrOutputParser()                       # Parse output as string
    )
    print("   ✓ LCEL chain created")

    # 7. Invoke chain
    print("\n7. Invoking chain...")
    try:
        response = rag_chain.invoke(query)
        print("   ✓ Response generated successfully")
        return response
    except Exception as e:
        print(f"   ✗ Error invoking chain: {e}")
        return None


def interactive_chat(pdf_path):
    """
    Interactive resume chat using LCEL approach
    """
    print("\n" + "=" * 70)
    print("Interactive Resume Chat (LCEL)")
    print("=" * 70)
    print("\nType 'exit', 'quit', or 'bye' to exit\n")
    
    # Load and setup once
    print("Setting up RAG pipeline...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma.from_documents(
        documents=splits, 
        embedding=embeddings
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not set")
        return
    
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.7,
        groq_api_key=groq_api_key
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a resume assistant. Answer questions about the resume. "
            "Use bullet points for lists. Be concise.\n\n{context}"
        )),
        ("human", "{input}"),
    ])
    
    rag_chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("✓ Ready for questions!\n")
    
    # Chat loop
    while True:
        user_query = input("You: ").strip()
        
        if user_query.lower() in ['exit', 'quit', 'bye']:
            print("\nGoodbye!")
            break
        
        if not user_query:
            continue
        
        print("\nAssistant: ", end="", flush=True)
        try:
            response = rag_chain.invoke(user_query)
            print(response)
            print()
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    # ✅ FIXED: Use environment variable from .env file
    # Create .env file with: GROQ_API_KEY=your_key_here
    
    pdf_file = "Sai Tulasi CV.pdf"
    
    # Validate setup
    if not os.path.exists(pdf_file):
        print(f"Error: '{pdf_file}' not found")
        print("Please ensure the PDF file is in the current directory")
        exit(1)
    
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY not set")
        print("\nSetup instructions:")
        print("1. Create a .env file in the current directory:")
        print("   GROQ_API_KEY=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")
        print("2. Run this script again")
        exit(1)
    
    # Test queries
    test_queries = [
        "What are the main skills in this CV?",
        "What is the work experience?",
        "What is the educational background?"
    ]
    
    try:
        # Mode 1: Single query
        print("\n" + "=" * 70)
        print("MODE 1: Single Query Demonstration")
        print("=" * 70)
        
        for query in test_queries[:1]:  # Show first query only
            answer = build_rag(pdf_file, query)
            
            if answer:
                print("\n" + "=" * 70)
                print("QUESTION:")
                print("=" * 70)
                print(f"\n{query}\n")
                print("=" * 70)
                print("ANSWER:")
                print("=" * 70)
                print(f"\n{answer}\n")
        
        # Mode 2: Interactive chat
        # Uncomment below to use interactive mode instead
        print("\n" + "=" * 70)
        print("MODE 2: Interactive Chat")
        print("=" * 70)
        interactive_chat(pdf_file)
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("\nInstall required packages:")
        print("pip install langchain langchain-community langchain-text-splitters")
        print("pip install langchain-huggingface langchain-groq chromadb pypdf")
        print("pip install sentence-transformers python-dotenv")