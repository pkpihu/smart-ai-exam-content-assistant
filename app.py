import os
import re

import streamlit as st
import requests
import feedparser
from bs4 import BeautifulSoup

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS


# -----------------------------------------------------------------------------
# Streamlit page configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Exam Content Local AI Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Smart AI Assistant for Exam Content Aggregation (Ollama Local)")
st.caption(
    "Collect current-affairs content, create concise exam notes, "
    "and practise with AI-generated MCQs using local models."
)


# -----------------------------------------------------------------------------
# Ollama Configuration sidebar
# -----------------------------------------------------------------------------
st.sidebar.markdown("### Ollama Configuration")
ollama_base_url = st.sidebar.text_input(
    "Ollama Base URL",
    value="http://localhost:11434",
)
ollama_model = st.sidebar.text_input(
    "Chat Model Name",
    value="qwen2.5:3b",
)
ollama_embed_model = st.sidebar.text_input(
    "Embedding Model Name",
    value="nomic-embed-text",
)
st.sidebar.caption(
    "Ensure Ollama is running locally and models are pulled "
    "(e.g., `ollama pull qwen2.5:3b`, `ollama pull nomic-embed-text`)."
)

DEFAULT_RSS = "https://feeds.bbci.co.uk/news/world/rss.xml"
rss_url = st.text_input("News RSS feed URL", value=DEFAULT_RSS)
article_urls = st.text_area(
    "Optional: paste article URLs (one per line)",
    height=90,
    placeholder="https://example.com/news-article",
)
extra_text = st.text_area(
    "Optional: paste syllabus, class notes, or article text",
    height=120,
)


# -----------------------------------------------------------------------------
# Text collection and cleaning
# -----------------------------------------------------------------------------
def clean_text(text):
    """Normalize whitespace in scraped or user-provided text."""
    return re.sub(r"\s+", " ", text or "").strip()


def fetch_article(url):
    """Download an article page and extract its main readable text."""
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ExamStudyAssistant/1.0)"
    }
    response = requests.get(url, headers=headers, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    main = soup.find("article") or soup.find("main") or soup.body or soup
    title = clean_text(
        soup.title.get_text(" ", strip=True) if soup.title else url
    )

    paragraphs = [
        element.get_text(" ", strip=True)
        for element in main.find_all(["p", "h1", "h2", "h3"])
    ]
    body = clean_text(" ".join(paragraphs))

    if len(body) < 250:
        body = clean_text(main.get_text(" ", strip=True))

    return title, body[:18000]


def collect_sources():
    """Collect documents from RSS, article URLs, and pasted study material."""
    docs = []
    errors = []

    if rss_url.strip():
        try:
            feed = feedparser.parse(rss_url.strip())

            if getattr(feed, "bozo", False) and not feed.entries:
                errors.append("Could not read the RSS feed. Check the URL.")

            for entry in feed.entries[:8]:
                link = entry.get("link", "")
                title = clean_text(entry.get("title", "News item"))
                summary_html = entry.get("summary", "")
                summary = clean_text(
                    BeautifulSoup(summary_html, "html.parser").get_text(" ")
                )
                content = summary

                if link:
                    try:
                        scraped_title, scraped_body = fetch_article(link)
                        title = scraped_title or title
                        content = scraped_body or summary
                    except Exception:
                        pass

                if content:
                    docs.append(
                        Document(
                            page_content=(
                                f"Title: {title}\nSource: {link}\n{content}"
                            ),
                            metadata={
                                "title": title,
                                "source": link or rss_url,
                            },
                        )
                    )

        except Exception as exc:
            errors.append(f"RSS error: {exc}")

    for url in [u.strip() for u in article_urls.splitlines() if u.strip()]:
        try:
            title, body = fetch_article(url)
            if body:
                docs.append(
                    Document(
                        page_content=f"Title: {title}\nSource: {url}\n{body}",
                        metadata={"title": title, "source": url},
                    )
                )
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    if extra_text.strip():
        docs.append(
            Document(
                page_content=extra_text.strip(),
                metadata={
                    "title": "User-provided study material",
                    "source": "Pasted text",
                },
            )
        )

    return docs, errors


# -----------------------------------------------------------------------------
# Ollama response handling
# -----------------------------------------------------------------------------
def extract_response_text(content):
    """Extract readable text from string or block-based LangChain responses."""
    if isinstance(content, str):
        text = content.strip()
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str) and block.strip():
                text_parts.append(block.strip())
            elif isinstance(block, dict):
                block_text = block.get("text")
                if isinstance(block_text, str) and block_text.strip():
                    text_parts.append(block_text.strip())
        text = "\n\n".join(text_parts).strip()
    elif isinstance(content, dict):
        block_text = content.get("text")
        text = block_text.strip() if isinstance(block_text, str) else ""
    else:
        text = ""

    text = re.sub(
        r"^\s*```(?:markdown|md)?\s*\n(.*?)\n```\s*$",
        r"\1",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return text.strip()


def extract_stream_chunk_text(content):
    """Extract a streaming chunk without stripping meaningful whitespace."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_text = block.get("text")
                if isinstance(block_text, str):
                    parts.append(block_text)
        return "".join(parts)
    if isinstance(content, dict):
        block_text = content.get("text")
        return block_text if isinstance(block_text, str) else ""
    return ""


@st.cache_resource(show_spinner=False)
def get_ollama_model(base_url, model_name, temperature=0.2):
    """Initialize local ChatOllama instance."""
    return ChatOllama(
        base_url=base_url,
        model=model_name,
        temperature=temperature,
    )


@st.cache_resource(show_spinner=False)
def get_embedding_model(base_url, embed_model_name):
    """Initialize local Ollama embeddings instance."""
    return OllamaEmbeddings(
        base_url=base_url,
        model=embed_model_name,
    )


def retrieve_documents_with_retry(vectorstore, query, k=3):
    """Retrieve documents from FAISS using local embeddings."""
    embeddings = st.session_state.get("embeddings")
    if embeddings is None:
        embeddings = getattr(vectorstore, "embedding_function", None)
    if embeddings is None:
        raise RuntimeError(
            "The embedding model is not available. Rebuild the knowledge base."
        )

    cache = st.session_state.setdefault("query_embedding_cache", {})
    cache_key = query.strip().casefold()

    if cache_key not in cache:
        cache[cache_key] = embeddings.embed_query(query)

        if len(cache) > 100:
            oldest_key = next(iter(cache))
            if oldest_key != cache_key:
                cache.pop(oldest_key, None)

    return vectorstore.similarity_search_by_vector(cache[cache_key], k=k)


def stream_markdown_response(llm, prompt, placeholder, formatter=None):
    """Stream text from local Ollama instance."""
    pieces = []
    chunk_count = 0

    for chunk in llm.stream(prompt):
        piece = extract_stream_chunk_text(getattr(chunk, "content", ""))
        if not piece:
            continue

        pieces.append(piece)
        chunk_count += 1

        if chunk_count == 1 or chunk_count % 3 == 0:
            placeholder.markdown("".join(pieces))

    final_text = "".join(pieces).strip()
    if formatter is not None:
        final_text = formatter(final_text)

    placeholder.markdown(final_text or "Ollama returned no readable text.")
    return final_text


def format_mcq_option_lines(markdown_text):
    """Ensure MCQ options format correctly."""
    if not markdown_text:
        return markdown_text

    heading_match = re.search(
        r"(?im)^#{1,6}\s*(?:B\.\s*)?MULTIPLE\s*CHOICE QUESTIONS\b[^\n]*",
        markdown_text,
    )
    if not heading_match:
        return markdown_text

    notes_section = markdown_text[:heading_match.end()]
    mcq_section = markdown_text[heading_match.end():]
    formatted_lines = []

    for line in mcq_section.splitlines():
        stripped = line.strip()

        if not stripped:
            if formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
            continue

        if re.match(
            r"^(?:\*\*)?(?:Correct Answer|Explanation)\s*:",
            stripped,
            flags=re.IGNORECASE,
        ):
            if formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
            formatted_lines.append(stripped)
            continue

        option_markers = list(
            re.finditer(r"(?<![A-Za-z0-9_])([A-D])[.)]\s+", line)
        )

        if len(option_markers) >= 2:
            question_or_prefix = line[:option_markers[0].start()].strip()
            if question_or_prefix:
                formatted_lines.append(question_or_prefix)
                formatted_lines.append("")

            for index, marker in enumerate(option_markers):
                option_end = (
                    option_markers[index + 1].start()
                    if index + 1 < len(option_markers)
                    else len(line)
                )
                option_text = line[marker.start():option_end].strip()
                if option_text:
                    formatted_lines.append(option_text)
                    formatted_lines.append("")

        elif re.match(r"^[A-D][.)]\s+", stripped):
            if formatted_lines and formatted_lines[-1] != "":
                formatted_lines.append("")
            formatted_lines.append(stripped)
            formatted_lines.append("")

        else:
            formatted_lines.append(stripped)

    return notes_section + "\n\n" + "\n".join(formatted_lines).strip()


def docs_to_sources(docs):
    """Convert retrieved documents to a simple, session-state-safe list."""
    return [
        (d.metadata.get("title", "Source"), d.metadata.get("source", ""))
        for d in docs
    ]


def render_sources(label, sources):
    """Show a sources expander from a list of (title, source) tuples."""
    with st.expander(label):
        for title, src in sources:
            st.write(title)
            st.caption(src)


# -----------------------------------------------------------------------------
# Question-answering prompt
# -----------------------------------------------------------------------------
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful current-affairs and competitive-exam study assistant.

Answer the user's question using the retrieved source material below.

Important instructions:
- Do NOT require the source material to contain the exact words used in the question.
- When the user asks what is relevant to a government/competitive exam, identify relevant facts by their subject matter.
- Do NOT add outside factual claims, dates, figures, names, or events that are not supported by the retrieved source material.
- Be concise, specific, and readable. Use headings and bullet points where helpful.

Retrieved source material:
{context}

User question:
{question}

Answer:""",
)


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "source_count" not in st.session_state:
    st.session_state.source_count = 0
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "query_embedding_cache" not in st.session_state:
    st.session_state.query_embedding_cache = {}
if "qa_result" not in st.session_state:
    st.session_state.qa_result = None
if "notes_result" not in st.session_state:
    st.session_state.notes_result = None


# -----------------------------------------------------------------------------
# Build the searchable knowledge base
# -----------------------------------------------------------------------------
left, right = st.columns([1, 1])

with left:
    build = st.button(
        "Collect content & build knowledge base",
        type="primary",
        use_container_width=True,
    )

with right:
    clear = st.button("Clear session", use_container_width=True)

if clear:
    st.session_state.vectorstore = None
    st.session_state.source_count = 0
    st.session_state.embeddings = None
    st.session_state.query_embedding_cache = {}
    st.session_state.qa_result = None
    st.session_state.notes_result = None
    st.rerun()

if build:
    try:
        with st.spinner(
            "Collecting articles and preparing the local searchable knowledge base..."
        ):
            docs, errors = collect_sources()

            if errors:
                with st.expander("Source warnings"):
                    for error in errors:
                        st.warning(error)

            if not docs:
                st.error(
                    "No content was collected. Add a working RSS URL, "
                    "article URL, or pasted text."
                )
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=900,
                    chunk_overlap=150,
                )
                chunks = splitter.split_documents(docs)

                embeddings = get_embedding_model(
                    ollama_base_url, ollama_embed_model
                )

                st.session_state.vectorstore = FAISS.from_documents(
                    chunks,
                    embeddings,
                )
                st.session_state.embeddings = embeddings
                st.session_state.query_embedding_cache = {}
                st.session_state.source_count = len(docs)

                # A new knowledge base makes old answers stale
                st.session_state.qa_result = None
                st.session_state.notes_result = None

                st.success(
                    f"Knowledge base ready: {len(docs)} source(s), "
                    f"{len(chunks)} searchable chunk(s)."
                )

    except Exception as exc:
        st.error(f"Could not build the knowledge base: {exc}")


# -----------------------------------------------------------------------------
# Step 2: Ask questions from the collected content
# -----------------------------------------------------------------------------
if st.session_state.vectorstore is not None:
    st.divider()
    st.subheader("2. Ask questions from collected content")

    question = st.text_input(
        "Example: Summarize the key facts and explain their possible relevance to competitive exams.",
        key="qa_question",
    )

    # True only on the run where a fresh answer was just streamed to the screen
    qa_just_generated = False

    if st.button("Get answer", key="qa_button"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            stage_message = st.empty()
            try:
                stage_message.info("Searching local vector database...")
                source_docs = retrieve_documents_with_retry(
                    st.session_state.vectorstore, question, k=3
                )

                if not source_docs:
                    stage_message.empty()
                    st.warning("No relevant material found.")
                else:
                    context = "\n\n".join(
                        doc.page_content[:1800] for doc in source_docs
                    )
                    prompt = QA_PROMPT.format(
                        context=context, question=question
                    )
                    llm = get_ollama_model(
                        ollama_base_url, ollama_model, temperature=0.2
                    )

                    stage_message.info(
                        "Generating response with local Ollama model..."
                    )
                    st.markdown(f"**Question:** {question}")
                    answer_placeholder = st.empty()
                    answer_text = stream_markdown_response(
                        llm, prompt, answer_placeholder
                    )
                    stage_message.empty()

                    sources = docs_to_sources(source_docs)
                    render_sources("Sources used", sources)

                    # Persist so it survives later reruns (e.g. Step 3 button)
                    st.session_state.qa_result = {
                        "question": question,
                        "answer": answer_text,
                        "sources": sources,
                    }
                    qa_just_generated = True

            except Exception as exc:
                stage_message.empty()
                st.error(f"Error during execution: {exc}")

    # On every other rerun, redraw the saved answer from session state
    qa = st.session_state.qa_result
    if qa and not qa_just_generated:
        st.markdown(f"**Question:** {qa['question']}")
        st.markdown(qa["answer"])
        render_sources("Sources used", qa["sources"])


# -----------------------------------------------------------------------------
# Step 3: Generate revision notes and MCQs
# -----------------------------------------------------------------------------
if st.session_state.vectorstore is not None:
    st.divider()
    st.subheader("3. Generate revision notes and practice MCQs")

    topic = st.text_input(
        "Topic for revision material",
        placeholder="e.g. International relations, science and technology",
        key="notes_topic",
    )
    count = st.slider(
        "Number of MCQs", min_value=3, max_value=10, value=5, key="mcq_count"
    )

    # True only on the run where fresh notes were just streamed to the screen
    notes_just_generated = False

    if st.button("Generate notes + MCQs", type="primary", key="notes_button"):
        if not topic.strip():
            st.warning("Enter a topic first.")
        else:
            try:
                with st.spinner(
                    "Retrieving content and generating study notes locally..."
                ):
                    relevant_docs = retrieve_documents_with_retry(
                        st.session_state.vectorstore, topic, k=4
                    )

                    if not relevant_docs:
                        st.warning("No relevant material found for this topic.")
                    else:
                        context = "\n\n".join(
                            doc.page_content for doc in relevant_docs
                        )

                        llm = get_ollama_model(
                            ollama_base_url, ollama_model, temperature=0.2
                        )

                        prompt = f"""
You are an expert competitive-exam preparation tutor.

Use ONLY the information provided in the CONTEXT below. Do not invent facts, answers, or dates.

TOPIC: {topic}

Create the following two sections.

## A. REVISION NOTES
- Use clear headings, subheadings, and concise bullet points.
- Highlight important terms using Markdown bold.

## B. MULTIPLE-CHOICE QUESTIONS
Generate EXACTLY {count} different MCQs based only on the supplied context.
Each question must have four options and one correct answer.

Use this exact format for every question:
### Question 1. [Write question here]
A. [Option A]
B. [Option B]
C. [Option C]
D. [Option D]

**Correct Answer:** B. [Correct option text]
**Explanation:** [One concise sentence.]

CONTEXT:
{context}
"""

                        st.markdown(f"**Topic:** {topic}")
                        output_placeholder = st.empty()
                        generated_text = stream_markdown_response(
                            llm,
                            prompt,
                            output_placeholder,
                            formatter=format_mcq_option_lines,
                        )

                        sources = docs_to_sources(relevant_docs)
                        render_sources("Source material used", sources)

                        # Persist so it survives later reruns (e.g. Step 2 button)
                        st.session_state.notes_result = {
                            "topic": topic,
                            "text": generated_text,
                            "sources": sources,
                        }
                        notes_just_generated = True

            except Exception as exc:
                st.error(f"Could not generate notes and MCQs: {exc}")

    # On every other rerun, redraw the saved notes from session state
    nr = st.session_state.notes_result
    if nr and not notes_just_generated:
        st.markdown(f"**Topic:** {nr['topic']}")
        st.markdown(nr["text"])
        render_sources("Source material used", nr["sources"])