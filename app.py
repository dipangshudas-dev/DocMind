
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import os
import time
import csv
import io
from datetime import datetime

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Initialize session state
if "answer" not in st.session_state:
    st.session_state.answer = ""

if "sources" not in st.session_state:
    st.session_state.sources = []

if "retrieval_results" not in st.session_state:
    st.session_state.retrieval_results = {}

if "evaluation_history" not in st.session_state:
    st.session_state.evaluation_history = []


# PDF and TXT extraction
def get_document_text(uploaded_files):
    documents = []

    for file in uploaded_files:

        file_extension = file.name.lower().split(".")[-1]

        # PDF extraction
        if file_extension == "pdf":
            file.seek(0)
            pdf_reader = PdfReader(file)

            for page_number, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text() or ""

                if page_text.strip():
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "source": file.name,
                                "page": page_number
                            }
                        )
                    )

        # TXT extraction
        elif file_extension == "txt":
            raw_text = file.getvalue()

            try:
                text = raw_text.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = raw_text.decode("latin-1")

            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file.name,
                            "page": "N/A"
                        }
                    )
                )

    return documents


# Text chunking
def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


# Create FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    vector_store = FAISS.from_documents(
        text_chunks,
        embedding=embeddings
    )

    vector_store.save_local("faiss_index")


# Load FAISS vector store
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )


# Gemini conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question using only the provided context.
    If the answer is not available in the context, say:
    "Answer is not available in the context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.3,
        max_retries=5
    )

    prompt = PromptTemplate.from_template(prompt_template)

    chain = prompt | model

    return chain


# Handle Gemini response formats
def extract_answer(response):
    if isinstance(response.content, str):
        return response.content

    return "\n".join(
        block.get("text", "")
        for block in response.content
        if isinstance(block, dict) and block.get("type") == "text"
    )


# Get unique sources
def get_sources(docs):
    sources = []
    seen = set()

    for doc in docs:
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "Unknown")

        source_info = (source, page)

        if source_info not in seen:
            sources.append({
                "source": source,
                "page": page
            })
            seen.add(source_info)

    return sources


# Retrieval Strategy Comparison
def compare_retrieval_strategies(vector_store, question):

    results = {}

    # Strategy 1: Similarity Search
    start_time = time.perf_counter()

    similarity_docs = vector_store.similarity_search(
        question,
        k=4
    )

    similarity_time = time.perf_counter() - start_time

    results["Similarity Search"] = {
        "docs": similarity_docs,
        "time": similarity_time
    }

    # Strategy 2: Maximum Marginal Relevance
    start_time = time.perf_counter()

    mmr_docs = vector_store.max_marginal_relevance_search(
        question,
        k=4,
        fetch_k=12,
        lambda_mult=0.5
    )

    mmr_time = time.perf_counter() - start_time

    results["MMR Search"] = {
        "docs": mmr_docs,
        "time": mmr_time
    }

    return results


# Generate answer
def generate_answer(question, docs):

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    chain = get_conversational_chain()

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return extract_answer(response)


# Display sources
def display_sources(docs):

    st.write("#### Sources")

    sources = get_sources(docs)

    if not sources:
        st.caption("No source information available.")
        return

    for source in sources:
        st.caption(
            f"Source: {source['source']} | Page: {source['page']}"
        )


# Ask documents using selected retrieval strategy
def user_input(user_question, strategy="Similarity Search"):

    new_db = load_vector_store()

    if strategy == "MMR Search":
        docs = new_db.max_marginal_relevance_search(
            user_question,
            k=4,
            fetch_k=12,
            lambda_mult=0.5
        )
    else:
        docs = new_db.similarity_search(
            user_question,
            k=4
        )

    answer = generate_answer(user_question, docs)

    st.session_state.answer = answer
    st.session_state.sources = docs

    return answer, docs


# Answer Quality Evaluation
def save_evaluation(
    question,
    answer,
    feedback,
    reference_answer,
    relevance_score,
    faithfulness_score
):

    evaluation = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "feedback": feedback,
        "reference_answer": reference_answer,
        "relevance_score": relevance_score,
        "faithfulness_score": faithfulness_score
    }

    st.session_state.evaluation_history.append(evaluation)


# Main application
def main():

    st.set_page_config(
        page_title="DocMind - AI Document Assistant",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 DocMind")
    st.caption("LLM-Based Intelligent Question Answering System")
    st.markdown("---")

    # Sidebar
    with st.sidebar:

        st.header("📄 Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )

        if st.button("Process Documents", use_container_width=True):

            if uploaded_files:

                with st.spinner("Processing your documents..."):

                    try:
                        documents = get_document_text(uploaded_files)

                        if documents:

                            text_chunks = get_text_chunks(documents)

                            if text_chunks:

                                get_vector_store(text_chunks)

                                st.success(
                                    f"Successfully processed {len(documents)} document sections "
                                    f"and created {len(text_chunks)} chunks."
                                )

                            else:
                                st.warning("No text chunks were extracted.")

                        else:
                            st.warning(
                                "No readable text found in the uploaded files."
                            )

                    except Exception as e:
                        st.error(f"Processing Error: {str(e)}")

            else:
                st.warning(
                    "Please upload at least one PDF or TXT file."
                )

        st.markdown("---")
        st.caption("Supported formats: PDF, TXT")

    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "💬 Ask Documents",
        "🔍 Retrieval Comparison",
        "📊 Answer Evaluation"
    ])

    # TAB 1: Document Question Answering
    with tab1:

        st.subheader("Ask Your Documents")

        user_question = st.text_input(
            "Enter your question",
            placeholder="Ask anything about your uploaded documents...",
            key="main_question"
        )

        selected_strategy = st.selectbox(
            "Select Retrieval Strategy",
            [
                "Similarity Search",
                "MMR Search"
            ]
        )

        if st.button("Get Answer", type="primary"):

            if user_question.strip():

                if os.path.exists("faiss_index/index.faiss"):

                    with st.spinner("Finding the answer..."):

                        try:

                            answer, docs = user_input(
                                user_question,
                                selected_strategy
                            )

                            st.success("Answer generated successfully!")

                        except Exception as e:
                            st.error(f"Error: {str(e)}")

                else:
                    st.warning(
                        "Please upload and process documents first."
                    )

            else:
                st.warning("Please enter a question.")

        # Display saved answer after Streamlit rerun
        if st.session_state.answer:

            st.markdown("---")
            st.write("### Answer")

            st.write(st.session_state.answer)

            display_sources(st.session_state.sources)

    # TAB 2: Retrieval Strategy Comparison
    with tab2:

        st.subheader("Retrieval Strategy Comparison")

        st.write(
            "Compare Similarity Search and Maximum Marginal Relevance (MMR) "
            "to understand how different retrieval strategies select document chunks."
        )

        comparison_question = st.text_input(
            "Enter a question for comparison",
            placeholder="Enter your question...",
            key="comparison_question"
        )

        if st.button("Compare Retrieval Strategies"):

            if comparison_question.strip():

                if os.path.exists("faiss_index/index.faiss"):

                    try:

                        with st.spinner("Comparing retrieval strategies..."):

                            db = load_vector_store()

                            results = compare_retrieval_strategies(
                                db,
                                comparison_question
                            )

                            st.session_state.retrieval_results = results

                        st.success("Comparison completed!")

                    except Exception as e:
                        st.error(f"Comparison Error: {str(e)}")

                else:
                    st.warning(
                        "Please upload and process documents first."
                    )

            else:
                st.warning("Please enter a question.")

        if st.session_state.retrieval_results:

            results = st.session_state.retrieval_results

            st.markdown("---")
            st.write("### Performance Comparison")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Similarity Search Time",
                    f"{results['Similarity Search']['time']:.4f} sec"
                )

                st.metric(
                    "Retrieved Chunks",
                    len(results["Similarity Search"]["docs"])
                )

            with col2:

                st.metric(
                    "MMR Search Time",
                    f"{results['MMR Search']['time']:.4f} sec"
                )

                st.metric(
                    "Retrieved Chunks",
                    len(results["MMR Search"]["docs"])
                )

            st.info(
                "Lower retrieval time means faster retrieval. "
                "These timings measure the retrieval operation, not LLM answer generation."
            )

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:

                st.write("### Similarity Search")

                for i, doc in enumerate(
                    results["Similarity Search"]["docs"],
                    start=1
                ):

                    with st.expander(f"Chunk {i}"):

                        st.write(doc.page_content)

                        st.caption(
                            f"Source: {doc.metadata.get('source', 'Unknown')} | "
                            f"Page: {doc.metadata.get('page', 'Unknown')}"
                        )

            with col2:

                st.write("### MMR Search")

                for i, doc in enumerate(
                    results["MMR Search"]["docs"],
                    start=1
                ):

                    with st.expander(f"Chunk {i}"):

                        st.write(doc.page_content)

                        st.caption(
                            f"Source: {doc.metadata.get('source', 'Unknown')} | "
                            f"Page: {doc.metadata.get('page', 'Unknown')}"
                        )

    # TAB 3: Answer Quality Evaluation
    with tab3:

        st.subheader("Answer Quality Evaluation")

        st.write(
            "Evaluate generated answers using human feedback, "
            "answer relevance and faithfulness scores."
        )

        if st.session_state.answer:

            st.write("### Latest Generated Answer")

            st.info(st.session_state.answer)

            evaluation_question = st.text_area(
                "Question",
                value=st.session_state.get("main_question", ""),
                disabled=True
            )

            feedback = st.radio(
                "Was this answer helpful?",
                ["👍 Helpful", "👎 Not Helpful"],
                horizontal=True
            )

            reference_answer = st.text_area(
                "Reference Answer (Optional)",
                placeholder="Enter the expected or reference answer..."
            )

            col1, col2 = st.columns(2)

            with col1:

                relevance_score = st.slider(
                    "Answer Relevance (1 = Poor, 5 = Excellent)",
                    min_value=1,
                    max_value=5,
                    value=3
                )

            with col2:

                faithfulness_score = st.slider(
                    "Context Faithfulness (1 = Poor, 5 = Excellent)",
                    min_value=1,
                    max_value=5,
                    value=3
                )

            if st.button("Submit Evaluation"):

                save_evaluation(
                    evaluation_question,
                    st.session_state.answer,
                    feedback,
                    reference_answer,
                    relevance_score,
                    faithfulness_score
                )

                st.success("Evaluation saved successfully!")

        else:

            st.info(
                "Generate an answer in the Ask Documents tab first "
                "to evaluate its quality."
            )

        # Evaluation history
        st.markdown("---")
        st.write("### Evaluation History")

        history = st.session_state.evaluation_history

        if history:

            total_evaluations = len(history)

            average_relevance = sum(
                item["relevance_score"] for item in history
            ) / total_evaluations

            average_faithfulness = sum(
                item["faithfulness_score"] for item in history
            ) / total_evaluations

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Evaluations",
                total_evaluations
            )

            col2.metric(
                "Average Relevance",
                f"{average_relevance:.2f}/5"
            )

            col3.metric(
                "Average Faithfulness",
                f"{average_faithfulness:.2f}/5"
            )

            # CSV Export
            csv_buffer = io.StringIO()

            fieldnames = [
                "timestamp",
                "question",
                "answer",
                "feedback",
                "reference_answer",
                "relevance_score",
                "faithfulness_score"
            ]

            writer = csv.DictWriter(
                csv_buffer,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerows(history)

            st.download_button(
                label="📥 Download Evaluation Report (CSV)",
                data=csv_buffer.getvalue(),
                file_name="docmind_evaluation_report.csv",
                mime="text/csv"
            )

            for index, item in enumerate(
                reversed(history),
                start=1
            ):

                with st.expander(
                    f"Evaluation {index} - {item['timestamp']}"
                ):

                    st.write("**Question:**")
                    st.write(item["question"])

                    st.write("**Generated Answer:**")
                    st.write(item["answer"])

                    st.write("**Feedback:**", item["feedback"])

                    st.write(
                        "**Relevance Score:**",
                        f"{item['relevance_score']}/5"
                    )

                    st.write(
                        "**Faithfulness Score:**",
                        f"{item['faithfulness_score']}/5"
                    )

                    if item["reference_answer"]:
                        st.write("**Reference Answer:**")
                        st.write(item["reference_answer"])

        else:

            st.caption("No evaluations submitted yet.")

    # Footer
    st.markdown("---")

    st.caption(
        "DocMind | AI-powered Document Question Answering System"
    )


if __name__ == "__main__":
    main()
