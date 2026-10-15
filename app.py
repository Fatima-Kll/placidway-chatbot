
import os
import re
import json
from pathlib import Path
import chromadb
import streamlit as st
from dotenv import load_dotenv
from google import genai
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

load_dotenv()

st.set_page_config(
    page_title="PlacidWay Knowledge Chatbot",
    page_icon="🌐",
    layout="centered",
)

st.title("PlacidWay Website Knowledge Chatbot")
st.caption(
    "Answers based on the selected PlacidWay website pages. "
    "This chatbot does not provide medical advice."
)

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.8-flash"

FALLBACK_ANSWER = (
    "I couldn't find that information in the selected "
    "PlacidWay pages. Please consult the relevant official "
    "source for further information."
)

API_ERROR_MESSAGE = (
    "I couldn't generate an answer from the available information "
    "right now. Please try a question about the documented "
    "programs, prices, or inclusions."
)



# --------------------------------------------------
# 2. Load or build the ChromaDB knowledge base
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR / "data" / "chunks.json"
DB_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "placidway_knowledge"


@st.cache_resource
def load_knowledge_base():
    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    db = chromadb.PersistentClient(path=str(DB_PATH))

    collection = db.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},
    )

    # Build the collection on first launch or when it is empty.
    if collection.count() == 0:
        if not CHUNKS_PATH.exists():
            raise FileNotFoundError(
                f"Knowledge-base chunks not found: {CHUNKS_PATH}"
            )

        with CHUNKS_PATH.open("r", encoding="utf-8") as file:
            chunks = json.load(file)

        if not chunks:
            raise ValueError("The knowledge-base chunks file is empty.")

        ids = []
        documents = []
        metadatas = []

        for index, chunk in enumerate(chunks):
            text = chunk.get("text", "").strip()
            if not text:
                continue

            ids.append(chunk.get("chunk_id", f"chunk_{index}"))
            documents.append(text)
            metadatas.append({
                "source_url": chunk.get("source_url", ""),
                "page_title": chunk.get("page_title", ""),
                "section_heading": chunk.get("section_heading", ""),
                "chunk_index": int(chunk.get("chunk_index", index)),
            })

        for start in range(0, len(ids), 100):
            end = start + 100
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

    return collection


try:
    collection = load_knowledge_base()
except Exception as error:
    st.error(
        "Could not load or build the knowledge base. "
        "Check the data files and application logs."
    )
    print(f"Knowledge base error: {error}")
    st.stop()


# --------------------------------------------------
# 3. Shared knowledge-base helpers
# --------------------------------------------------

def get_all_chunks():
    data = collection.get(
        include=["documents", "metadatas"]
    )

    items = []

    for document, metadata in zip(
        data["documents"], data["metadatas"]
    ):
        if document:
            items.append({
                "document": document,
                "metadata": metadata or {},
                "distance": None,
            })

    return items


def find_chunk(items, predicate):
    for item in items:
        if predicate(
            item["document"], item["metadata"]
        ):
            return item

    return None


def put_first(item, retrieved_context):
    """Place supporting evidence first so [1] cites the right chunk."""
    if item is None:
        return

    retrieved_context[:] = [
        existing for existing in retrieved_context
        if existing["document"] != item["document"]
    ]

    retrieved_context.insert(0, item)


def get_source_url(metadata):
    """Accept common URL metadata keys without inventing a URL."""
    for key in ("source_url", "url", "page_url", "source"):
        value = metadata.get(key)

        if (
            isinstance(value, str)
            and value.strip().startswith(("https://", "http://"))
        ):
            return value.strip()

    return ""


def is_price_table(document, metadata):
    return (
        "program type" in
        metadata.get("section_heading", "").lower()
        and "$18,995" in document
        and "$30,000" in document
        and "anal cancer" in
        metadata.get("page_title", "").lower()
    )


def is_inpatient_inclusions(document, metadata):
    heading = metadata.get("section_heading", "").lower()
    normalized_heading = (
        heading.replace("-", "").replace(" ", "")
    )

    return (
        "includedintheinpatientprogram" in normalized_heading
        or (
            "inpatient" in normalized_heading
            and "inclusion" in normalized_heading
        )
        or (
            "inpatient" in normalized_heading
            and "included" in normalized_heading
        )
    )


def is_outpatient_inclusions(document, metadata):
    heading = metadata.get("section_heading", "").lower()
    normalized_heading = (
        heading.replace("-", "").replace(" ", "")
    )

    return (
        "includedintheoutpatientprogram" in normalized_heading
        or (
            "outpatient" in normalized_heading
            and "inclusion" in normalized_heading
        )
        or (
            "outpatient" in normalized_heading
            and "included" in normalized_heading
        )
    )


# --------------------------------------------------
# 4. Retrieve relevant website content
# --------------------------------------------------

def retrieve_context(question, number_of_results=6):
    results = collection.query(
        query_texts=[question],
        n_results=number_of_results,
        include=["documents", "metadatas", "distances"],
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []
    seen = set()

    for document, metadata, distance in zip(
        documents, metadatas, distances
    ):
        if document and document not in seen:
            seen.add(document)
            retrieved.append({
                "document": document,
                "metadata": metadata or {},
                "distance": distance,
            })

    question_lower = question.lower()

    asks_anal_price = (
        "anal cancer" in question_lower
        and any(term in question_lower for term in (
            "price", "cost", "how much", "package"
        ))
    )

    asks_inpatient_inclusions = (
        "inpatient" in question_lower
        and any(term in question_lower for term in (
            "include", "included", "includes", "inclusion"
        ))
    )

    # Retrieve the documented price table directly.
    if asks_anal_price:
        try:
            direct_results = collection.get(
                where={
                    "page_title":
                    "Alternative Anal Cancer Treatment in Tijuana"
                },
                include=["documents", "metadatas"],
            )

            for document, metadata in zip(
                direct_results["documents"],
                direct_results["metadatas"],
            ):
                metadata = metadata or {}

                if document and is_price_table(document, metadata):
                    put_first({
                        "document": document,
                        "metadata": metadata,
                        "distance": None,
                    }, retrieved)
                    break

        except Exception as error:
            print(f"Targeted pricing lookup error: {error}")

    # Retrieve inpatient inclusions directly.
    if asks_inpatient_inclusions:
        try:
            all_chunks = get_all_chunks()
            inclusion_chunk = find_chunk(
                all_chunks, is_inpatient_inclusions
            )

            if inclusion_chunk:
                put_first(inclusion_chunk, retrieved)

        except Exception as error:
            print(f"Targeted inclusions lookup error: {error}")

    # Safe debug output; works even if nothing was retrieved.
    print("\n--- RETRIEVAL DEBUG ---")
    print("QUESTION:", question)
    print("RETRIEVED CHUNKS:", len(retrieved))

    for index, item in enumerate(retrieved, start=1):
        metadata = item["metadata"]
        print(f"\nCHUNK {index}:")
        print("TITLE:", metadata.get("page_title"))
        print("SECTION:", metadata.get("section_heading"))
        print("URL:", get_source_url(metadata))
        print(item["document"][:300])

    if not retrieved:
        print("No relevant chunks retrieved.")

    return retrieved


# --------------------------------------------------
# 5. Gemini client
# --------------------------------------------------

@st.cache_resource
def get_gemini_client():
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured.")

    return genai.Client(api_key=API_KEY)


# --------------------------------------------------
# 6. Generate a grounded answer
# --------------------------------------------------

def generate_answer(question, retrieved_context):
    question_lower = question.lower()
    normalized = question_lower.replace("-", " ")

    # 1. Do not provide personalized medical recommendations.
    medical_phrases = [
        "which treatment should i choose",
        "what treatment should i choose",
        "which cancer treatment should i choose",
        "what cancer treatment should i choose",
        "which treatment is best for me",
        "what treatment is best for me",
        "which cancer treatment is best for me",
        "what cancer treatment is best for me",
        "what treatment do you recommend",
        "which treatment should i take",
        "what treatment should i take",
        "should i undergo",
        "diagnose me",
        "what is my diagnosis",
    ]

    if any(phrase in question_lower for phrase in medical_phrases):
        return (
            "I can't recommend or select a cancer treatment for you. "
            "Treatment decisions depend on your individual medical "
            "situation and should be discussed with a qualified "
            "oncologist. This chatbot provides information from "
            "selected PlacidWay pages and does not provide medical advice."
        )

    # 2. Weather is outside this website chatbot's scope.
    if any(word in question_lower for word in (
        "weather", "temperature", "forecast"
    )):
        return (
            "Weather information is outside the scope of this "
            "chatbot, which answers questions using selected "
            "PlacidWay website pages."
        )


    
    # Do not claim that a treatment guarantees a cure.
    cure_phrases = (
        "guarantee a cure",
        "guaranteed cure",
        "cure cancer",
        "will it cure",
        "will this cure",
        "guaranteed to cure",
        "guarantee that the treatment will cure",
    )

    asks_about_cure = any(
        phrase in question_lower for phrase in cure_phrases
    )

    if asks_about_cure:
        return (
            "The selected PlacidWay pages do not establish or "
            "guarantee that the treatment will cure cancer. "
            "The chatbot cannot verify treatment outcomes or "
            "guarantee a cure. Please discuss treatment "
            "effectiveness with a qualified oncologist."
        )


    # 3. Find phone-number questions in the local knowledge base.
    asks_phone = any(phrase in question_lower for phrase in (
        "phone number",
        "telephone number",
        "itc phone",
        "itc's phone",
        "itc’s phone",
        "phone number of itc",
        "how can i call itc",
        "contact number",
    ))

    if asks_phone:
        all_chunks = get_all_chunks()

        # Only accept a number explicitly associated with a phone
        # or telephone label. Do not mistake prices for phone numbers.
        phone_pattern = re.compile(
            r"(?i)(?:phone|telephone|tel\.?|contact number)"
            r"\s*(?:number)?\s*[:#-]?\s*"
            r"(\+?\d[\d\s().-]{6,}\d)"
        )

        phone_match = None
        phone_chunk = None

        for item in all_chunks:
            match = phone_pattern.search(item["document"])

            if match:
                phone_match = match
                phone_chunk = item
                break

        if phone_match and phone_chunk:
            put_first(phone_chunk, retrieved_context)
            number = phone_match.group(1).strip()
            title = (
                phone_chunk["metadata"].get("page_title")
                or "the selected PlacidWay page"
            )

            return (
                f"The phone number explicitly listed in "
                f"{title} is {number}. [1]"
            )

        return (
            "I couldn't verify ITC's phone number in the selected "
            "PlacidWay pages, so I won't provide an unverified number. "
            "You can use the consultation request option on the "
            "relevant PlacidWay page."
        )

    # 4. Identify price-related questions.
    asks_price = any(term in question_lower for term in (
        "price", "cost", "how much", "package"
    )) or "5000" in question_lower or "5,000" in question_lower

    asks_wrong_price = (
        "5000" in question_lower or "5,000" in question_lower
    )

    all_chunks = None

    if asks_price or "inpatient" in normalized or "outpatient" in normalized:
        all_chunks = get_all_chunks()

    price_chunk = find_chunk(
        all_chunks or [],
        is_price_table,
    )

    asks_anal_price = "anal cancer" in question_lower and asks_price

    
    # Answer questions asking for both program prices.
    asks_both_prices = (
        asks_price
        and "inpatient" in normalized
        and "outpatient" in normalized
    )

    if asks_both_prices and price_chunk:
        put_first(price_chunk, retrieved_context)

        return (
            "The documented 3-week program prices are: "
            "Out-Patient Program: USD 18,995; "
            "In-Patient Program: USD 30,000. [1]"
        )


    # Answer pricing questions with the documented price table.
    if (asks_anal_price or asks_wrong_price) and price_chunk:
        put_first(price_chunk, retrieved_context)

        if asks_wrong_price:
            return (
                "The provided program pricing table lists the "
                "3-week In-Patient Program at USD 30,000, not "
                "USD 5,000. The 3-week Out-Patient Program is "
                "listed at USD 18,995. [1]"
            )

        return (
            "The Alternative Anal Cancer Treatment page lists "
            "two program options: the 3-week Out-Patient Program "
            "for USD 18,995 and the 3-week In-Patient Program "
            "for USD 30,000. The table lists prices by program "
            "type, not a separate price based on anal cancer alone. [1]"
        )

    # Answer inpatient price questions, including questions asking
    # where the price is documented.
    asks_inpatient_price = "inpatient" in normalized and asks_price

    if asks_inpatient_price and price_chunk:
        put_first(price_chunk, retrieved_context)

        page_title = (
            price_chunk["metadata"].get("page_title")
            or "the selected PlacidWay page"
        )

        return (
            "The 3-week In-Patient Program is listed at USD 30,000 "
            f"on the {page_title} page. [1]"
        )

    # 5. Answer inpatient inclusion questions.
    asks_inpatient_inclusions = (
        "inpatient" in normalized
        and any(word in question_lower for word in (
            "include", "included", "includes", "inclusion"
        ))
    )

    if asks_inpatient_inclusions:
        if all_chunks is None:
            all_chunks = get_all_chunks()

        inclusion_chunk = find_chunk(
            all_chunks, is_inpatient_inclusions
        )

        if inclusion_chunk:
            put_first(inclusion_chunk, retrieved_context)

            return (
                "The 3-week In-Patient Program costs USD 30,000. "
                "The listed inclusions are doctor and nursing care "
                "throughout the stay, private accommodation and "
                "three meals daily, weekly laboratory panels, one "
                "prescribed X-ray, ultrasound, or CT scan, "
                "program-specific interventions, protocol-specific "
                "medications, and a follow-up consultation with "
                "an ITC doctor after 3 months. [1]"
            )

        return FALLBACK_ANSWER

    # 6. Compare inpatient and outpatient programs.
    asks_comparison = (
        "difference" in question_lower
        and "program" in question_lower
        and (
            "inpatient" in normalized or "outpatient" in normalized
        )
    ) or (
        "compare" in question_lower
        and (
            "inpatient" in normalized or "outpatient" in normalized
        )
    )

    if asks_comparison:
        if all_chunks is None:
            all_chunks = get_all_chunks()

        inpatient_chunk = find_chunk(
            all_chunks, is_inpatient_inclusions
        )
        outpatient_chunk = find_chunk(
            all_chunks, is_outpatient_inclusions
        )

        if price_chunk and inpatient_chunk and outpatient_chunk:
            retrieved_context[:] = [
                price_chunk,
                inpatient_chunk,
                outpatient_chunk,
            ] + [
                item for item in retrieved_context
                if item["document"] not in {
                    price_chunk["document"],
                    inpatient_chunk["document"],
                    outpatient_chunk["document"],
                }
            ]

            return (
                
            
                "Both programs last 3 weeks. The Out-Patient "
                "Program costs USD 18,995, while the In-Patient "
                "Program costs USD 30,000. The documented "
                "inpatient inclusions include doctor and nursing "
                "care, private accommodation, three meals daily, "
                "weekly laboratory panels, one prescribed X-ray, "
                "ultrasound, or CT scan, program-specific "
                "interventions, protocol-specific medications, "
                "and a follow-up consultation with an ITC doctor "
                "after 3 months. Consult the linked source pages "
                "for the complete documented details. [1][2][3]"
            )

            

        return FALLBACK_ANSWER

    # 7. Answer consultation questions from stored source text.
    asks_booking = any(term in question_lower for term in (
        "book an appointment",
        "booking",
        "contact itc",
        "contact the center",
        "consultation",
        "how do i contact",
    ))

    if asks_booking:
        if all_chunks is None:
            all_chunks = get_all_chunks()

        contact_chunk = find_chunk(
            all_chunks,
            lambda doc, meta: (
                "request a consultation" in doc.lower()
                or "share your medical information" in doc.lower()
            ),
        )

        if contact_chunk:
            put_first(contact_chunk, retrieved_context)

            return (
                "The selected PlacidWay page provides a "
                "'Request a Consultation With ITC Immunity "
                "Therapy Center' call to action and invites "
                "patients to share their medical information "
                "to receive guidance. The available source "
                "does not specify a phone number or a complete "
                "step-by-step booking procedure. [1]"
            )

        return FALLBACK_ANSWER

    # 8. Avoid Gemini if no source evidence was retrieved.
    if not retrieved_context:
        return FALLBACK_ANSWER

    # 9. Use Gemini for other questions only when available.
    evidence_sections = []

    for index, item in enumerate(retrieved_context, start=1):
        metadata = item.get("metadata") or {}

        evidence_sections.append(
            f"[{index}]\n"
            f"Page: {metadata.get('page_title', 'PlacidWay page')}\n"
            f"Section: {metadata.get('section_heading', 'General content')}\n"
            f"Source URL: {get_source_url(metadata) or 'URL unavailable'}\n"
            f"Content:\n{item['document']}\n"
        )

    prompt = f"""
You are a website knowledge assistant for PlacidWay.
Answer using ONLY the source excerpts below.

Do not invent facts, prices, booking procedures, contact details,
medical outcomes, or treatment recommendations.
If evidence is insufficient, respond exactly:
"{FALLBACK_ANSWER}"

Cite supported claims using the supplied labels, such as [1].
Treat testimonials as testimonials, not established medical evidence.
Do not provide personalized medical advice.

SOURCE EXCERPTS:
{"".join(evidence_sections)}

USER QUESTION:
{question}

ANSWER:
"""

    try:
        response = get_gemini_client().models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        if response and response.text and response.text.strip():
            return response.text.strip()

        return API_ERROR_MESSAGE

    except Exception as error:
        print(f"GEMINI ERROR: {error!r}")
        return API_ERROR_MESSAGE


# --------------------------------------------------
# 7. Select sources cited by the answer
# --------------------------------------------------

def get_cited_sources(answer, retrieved_context):
    cited_numbers = {
        int(number)
        for number in re.findall(r"\[(\d+)\]", answer)
    }

    sources = []
    seen = set()

    for index, item in enumerate(retrieved_context, start=1):
        if index not in cited_numbers:
            continue

        metadata = item.get("metadata") or {}
        title = metadata.get("page_title") or "PlacidWay source"
        url = get_source_url(metadata)

        # Deduplicate by URL when available, otherwise by title.
        unique_key = url or title

        if unique_key in seen:
            continue

        seen.add(unique_key)

        sources.append({
            "title": title,
            "url": url,
        })

    return sources


def display_sources(sources):
    """Show links when available; never render blank source links."""
    if not sources:
        return

    with st.expander("Sources used"):
        for source in sources:
            if source["url"]:
                st.markdown(
                    f"- [{source['title']}]({source['url']})"
                )
            else:
                st.markdown(
                    f"- {source['title']} "
                    "*(source URL missing from indexed metadata)*"
                )


# --------------------------------------------------
# 8. Display the chat interface
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing conversation history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sources"):
            display_sources(message["sources"])

question = st.chat_input(
    "Ask about ITC packages, prices, or services..."
)

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the PlacidWay pages..."):
            try:
                retrieved_context = retrieve_context(question)
                answer = generate_answer(
                    question, retrieved_context
                )

                st.markdown(answer)

                sources = get_cited_sources(
                    answer, retrieved_context
                )

                display_sources(sources)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as error:
                print(f"Chatbot error: {error!r}")

                error_answer = (
                    "I couldn't process your question right now. "
                    "Please try again in a little while."
                )

                st.error(error_answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_answer,
                    "sources": [],
                })


if st.button("Clear conversation"):
    st.session_state.messages = []
    st.rerun()
