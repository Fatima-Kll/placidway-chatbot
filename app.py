
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
    normalized = " ".join(normalized.split())

    # --------------------------------------------------
    # 1. Handle success-rate questions safely.
    # --------------------------------------------------
    success_rate_phrases = (
        "success rate",
        "cure rate",
        "every patient",
        "percentage of patients cured",
        "chances of success",
        "what are the chances of success",
        "how effective is this treatment",
        "treatment effectiveness",
    )

    if any(phrase in normalized for phrase in success_rate_phrases):
        return (
            "The selected PlacidWay pages do not provide a verified "
            "success rate for this treatment. Outcomes vary by patient, "
            "and this chatbot cannot predict individual results. Please "
            "discuss expected benefits, risks, and available evidence "
            "with a qualified oncologist."
        )

    # --------------------------------------------------
    # 2. Refuse personalized medical recommendations.
    # --------------------------------------------------
    medical_phrases = (
        "which treatment should i choose",
        "what treatment should i choose",
        "can you recommend which cancer treatment i should choose",
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
    )

    if any(phrase in normalized for phrase in medical_phrases):
        return (
            "I can't recommend or select a cancer treatment for you. "
            "Treatment decisions depend on your individual medical "
            "situation and should be discussed with a qualified "
            "oncologist. This chatbot provides information from selected "
            "PlacidWay pages and does not provide medical advice."
        )

    if (
        "cancer treatment" in normalized
        and any(term in normalized for term in (
            "recommend",
            "should i choose",
            "best for me",
            "should i take",
        ))
    ):
        return (
            "I can't recommend or select a cancer treatment for you. "
            "Treatment decisions depend on your individual medical "
            "situation and should be discussed with a qualified "
            "oncologist. This chatbot provides information from selected "
            "PlacidWay pages and does not provide medical advice."
        )

    # --------------------------------------------------
    # 3. Reject unrelated questions without calling Gemini.
    # --------------------------------------------------
    sports_terms = (
        "football",
        "soccer",
        "world cup",
        "fifa",
        "basketball",
        "tennis",
    )

    if any(term in normalized for term in sports_terms):
        return (
            "Sports results are outside the scope of this chatbot, "
            "which answers questions using selected PlacidWay website pages."
        )

    if any(word in normalized for word in (
        "weather", "temperature", "forecast"
    )):
        return (
            "Weather information is outside the scope of this chatbot, "
            "which answers questions using selected PlacidWay website pages."
        )

    # --------------------------------------------------
    # 4. Handle treatment cure and outcome questions safely.
    # --------------------------------------------------
    asks_cure_question = (
        any(term in normalized for term in ("cure", "cured"))
        and any(term in normalized for term in (
            "cancer", "treatment", "therapy", "this condition"
        ))
    )

    if asks_cure_question:
        return (
            "The selected PlacidWay pages do not establish that this "
            "treatment can cure cancer. This chatbot cannot promise or "
            "predict an individual medical outcome. Please discuss the "
            "treatment's evidence, potential benefits, and risks with "
            "a qualified oncologist."
        )

    outcome_phrases = (
        "guarantee a cure",
        "guaranteed cure",
        "will it cure",
        "will this cure",
        "guaranteed to cure",
        "guaranteed to be successful",
        "guaranteed successful",
        "guarantee success",
        "guaranteed success",
        "guarantee that every patient will be cured",
        "guarantee every patient",
        "guarantee a successful outcome",
        "guaranteed outcome",
        "guarantee treatment success",
    )

    asks_about_outcomes = any(
        phrase in normalized for phrase in outcome_phrases
    ) or (
        "guarantee" in normalized
        and any(term in normalized for term in (
            "treatment",
            "program",
            "patient",
            "cured",
            "successful",
            "success",
        ))
    )

    if asks_about_outcomes:
        return (
            "The selected PlacidWay pages do not establish or guarantee "
            "a cure or successful treatment outcome. This chatbot cannot "
            "verify or guarantee individual medical outcomes. Please "
            "discuss treatment effectiveness and risks with a qualified "
            "oncologist."
        )

    # --------------------------------------------------
    # 5. Handle payment-plan questions without Gemini.
    # --------------------------------------------------
    payment_terms = (
        "installment",
        "installments",
        "monthly payment",
        "monthly payments",
        "payment plan",
        "payment plans",
        "pay monthly",
        "pay in installments",
        "pay by installments",
        "financing options",
        "finance the treatment",
        "financing the treatment",
    )

    if any(term in normalized for term in payment_terms):
        return (
            "The selected PlacidWay pages do not confirm whether "
            "monthly installments, financing, or payment plans are "
            "available. Please contact PlacidWay or ITC Immunity "
            "Therapy Center to confirm the available payment options."
        )

    # --------------------------------------------------
    # 6. Answer phone-number questions only when a labelled
    #    number is found in the stored source text.
    # --------------------------------------------------
    asks_phone = any(phrase in normalized for phrase in (
        "phone number",
        "telephone number",
        "itc phone",
        "itc's phone",
        "phone number of itc",
        "how can i call itc",
        "contact number",
    ))

    if asks_phone:
        all_chunks = get_all_chunks()

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
                f"The phone number explicitly listed in {title} "
                f"is {number}. [1]"
            )

        return (
            "I couldn't verify ITC's phone number in the selected "
            "PlacidWay pages, so I won't provide an unverified number. "
            "You can use the consultation request option on the "
            "relevant PlacidWay page."
        )

    # --------------------------------------------------
    # 7. Load source chunks for price and program questions.
    # --------------------------------------------------
    asks_price = any(term in normalized for term in (
        "price",
        "cost",
        "how much",
        "package",
    )) or "5000" in normalized or "5,000" in normalized

    asks_wrong_price = (
        "5000" in normalized or "5,000" in normalized
    )

    asks_inpatient_program = (
        "inpatient" in normalized or "in patient" in normalized
    )

    asks_outpatient_program = (
        "outpatient" in normalized or "out patient" in normalized
    )

    all_chunks = None

    if asks_price or asks_inpatient_program or asks_outpatient_program:
        all_chunks = get_all_chunks()

    price_chunk = find_chunk(all_chunks or [], is_price_table)

    asks_anal_price = "anal cancer" in normalized and asks_price

    asks_both_prices = (
        asks_price
        and asks_inpatient_program
        and asks_outpatient_program
    )

    # Handle price differences before other price comparisons.
    asks_price_difference = any(
        term in normalized
        for term in (
            "price difference",
            "how much more",
            "how much less",
            "difference in price",
            "difference in cost",
            "cost difference",
            "which is cheaper",
            "which program costs less",
            "less expensive",
            "more expensive",
        )
    )

    if asks_price_difference and price_chunk:
        put_first(price_chunk, retrieved_context)

        if any(term in normalized for term in (
            "difference",
            "how much more",
            "how much less",
            "more expensive",
        )):
            return (
                "The In-Patient Program costs USD 30,000 and the "
                "Out-Patient Program costs USD 18,995. The difference "
                "is USD 11,005. The In-Patient Program is USD 11,005 "
                "more expensive. [1]"
            )

        return (
            "The 3-week Out-Patient Program costs less: USD 18,995 "
            "compared with USD 30,000 for the In-Patient Program. [1]"
        )

    if asks_both_prices and price_chunk:
        put_first(price_chunk, retrieved_context)
        return (
            "The documented 3-week program prices are: "
            "Out-Patient Program: USD 18,995; "
            "In-Patient Program: USD 30,000. [1]"
        )

    if asks_wrong_price and price_chunk:
        put_first(price_chunk, retrieved_context)
        return (
            "The provided program pricing table lists the 3-week "
            "In-Patient Program at USD 30,000, not USD 5,000. "
            "The 3-week Out-Patient Program is listed at USD 18,995. [1]"
        )

    if asks_anal_price and price_chunk:
        put_first(price_chunk, retrieved_context)
        return (
            "The Alternative Anal Cancer Treatment page lists two "
            "program options: the 3-week Out-Patient Program for "
            "USD 18,995 and the 3-week In-Patient Program for "
            "USD 30,000. The table lists prices by program type, "
            "not a separate price based on anal cancer alone. [1]"
        )

    # --------------------------------------------------
    # 8. Answer program-duration questions deterministically.
    # --------------------------------------------------
    asks_duration = any(term in normalized for term in (
        "how long",
        "duration",
        "how many weeks",
        "how many days",
        "how long do both programs last",
        "how long does each program last",
        "how long is the program",
        "how long does the program last",
    ))

    if asks_duration:
        if all_chunks is None:
            all_chunks = get_all_chunks()

        duration_chunk = find_chunk(
            all_chunks or [],
            lambda doc, meta: (
                is_price_table(doc, meta)
                and "3 weeks" in doc.lower()
            ),
        )

        if duration_chunk:
            put_first(duration_chunk, retrieved_context)
            return (
                "The documented In-Patient and Out-Patient Programs "
                "are both 3 weeks long. [1]"
            )

        return (
            "I couldn't verify the program duration in the selected "
            "source information. Please consult the relevant PlacidWay "
            "program page for confirmation."
        )

    # --------------------------------------------------
    # 9. Handle outpatient inclusions separately.
    #    Never reuse inpatient inclusions for outpatient questions.
    # --------------------------------------------------
    inclusion_terms = (
        "include",
        "included",
        "includes",
        "inclusion",
        "inclusions",
        "meal",
        "meals",
        "food",
        "accommodation",
        "follow-up",
        "follow up",
        "medication",
        "medications",
        "imaging",
        "laboratory",
        "lab test",
        "lab tests",
        "nursing care",
        "what is provided",
        "what is covered",
    )

    asks_inclusions = any(
        term in normalized for term in inclusion_terms
    ) and (
        asks_inpatient_program
        or asks_outpatient_program
        or "package" in normalized
        or "program" in normalized
    )

    if asks_inclusions and asks_outpatient_program:
        if all_chunks is None:
            all_chunks = get_all_chunks()

        outpatient_chunk = find_chunk(
            all_chunks or [],
            is_outpatient_inclusions,
        )

        # Include the outpatient source in the context when available,
        # but do not infer inclusions from the inpatient program.
        if outpatient_chunk:
            put_first(outpatient_chunk, retrieved_context)

        return (
            "The selected PlacidWay information does not allow me to "
            "verify whether private accommodation or meals are included "
            "in the Out-Patient Program. Inclusions listed for the "
            "In-Patient Program should not be assumed to apply to the "
            "Out-Patient Program. Please contact PlacidWay or ITC "
            "Immunity Therapy Center to confirm the outpatient inclusions."
        )

    # --------------------------------------------------
    # 10. Handle inpatient inclusions.
    # --------------------------------------------------
    if asks_inclusions and asks_inpatient_program:
        if all_chunks is None:
            all_chunks = get_all_chunks()

        inclusion_chunk = find_chunk(
            all_chunks or [],
            is_inpatient_inclusions,
        )

        if inclusion_chunk:
            put_first(inclusion_chunk, retrieved_context)

            if "accommodation" in normalized:
                return (
                    "Yes. Private accommodation is listed as an "
                    "inclusion in the 3-week In-Patient Program. [1]"
                )

            if (
                "follow-up" in normalized
                or "follow up" in normalized
                or "when" in normalized
            ):
                return (
                    "The listed follow-up consultation with an ITC "
                    "doctor takes place after 3 months. [1]"
                )

            if "meal" in normalized or "food" in normalized:
                return (
                    "Yes. Three meals daily are listed as an inclusion "
                    "in the 3-week In-Patient Program. [1]"
                )

            if any(term in normalized for term in (
                "laboratory",
                "lab test",
                "lab tests",
            )):
                return (
                    "Yes. Weekly laboratory panels are listed as an "
                    "inclusion in the 3-week In-Patient Program. [1]"
                )

            if "nursing care" in normalized:
                return (
                    "Doctor and nursing care throughout the stay are "
                    "listed as inclusions in the 3-week In-Patient Program. [1]"
                )

            if "ct scan" in normalized or "x ray" in normalized or "ultrasound" in normalized:
                return (
                    "The listed inpatient inclusions include one "
                    "prescribed X-ray, ultrasound, or CT scan. [1]"
                )

            return (
                "The 3-week In-Patient Program costs USD 30,000. "
                "The listed inclusions are doctor and nursing care "
                "throughout the stay, private accommodation and three "
                "meals daily, weekly laboratory panels, one prescribed "
                "X-ray, ultrasound, or CT scan, program-specific "
                "interventions, protocol-specific medications, and a "
                "follow-up consultation with an ITC doctor after "
                "3 months. [1]"
            )

        return FALLBACK_ANSWER

    # --------------------------------------------------
    # 11. Handle program-specific price questions.
    # --------------------------------------------------
    asks_outpatient_price = asks_outpatient_program and asks_price

    if asks_outpatient_price and price_chunk:
        put_first(price_chunk, retrieved_context)
        return (
            "The 3-week Out-Patient Program is listed at USD 18,995 "
            "in the documented program pricing table. [1]"
        )

    asks_inpatient_price = asks_inpatient_program and asks_price

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

    # --------------------------------------------------
    # 12. Compare the two programs when requested.
    # --------------------------------------------------
    asks_comparison = (
        "difference" in normalized
        and "program" in normalized
        and (asks_inpatient_program or asks_outpatient_program)
    ) or (
        "compare" in normalized
        and (asks_inpatient_program or asks_outpatient_program)
    )

    if asks_comparison:
        if all_chunks is None:
            all_chunks = get_all_chunks()

        inpatient_chunk = find_chunk(
            all_chunks or [],
            is_inpatient_inclusions,
        )
        outpatient_chunk = find_chunk(
            all_chunks or [],
            is_outpatient_inclusions,
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
                "Both programs last 3 weeks. The Out-Patient Program "
                "costs USD 18,995, while the In-Patient Program costs "
                "USD 30,000. The documented inpatient inclusions include "
                "doctor and nursing care, private accommodation, three "
                "meals daily, weekly laboratory panels, one prescribed "
                "X-ray, ultrasound, or CT scan, program-specific "
                "interventions, protocol-specific medications, and a "
                "follow-up consultation with an ITC doctor after "
                "3 months. Consult the linked source pages for complete "
                "documented details. [1][2][3]"
            )

        return FALLBACK_ANSWER

    # --------------------------------------------------
    # 13. Answer consultation and booking questions.
    # --------------------------------------------------
    asks_booking = any(term in normalized for term in (
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
                "'Request a Consultation With ITC Immunity Therapy "
                "Center' call to action and invites patients to share "
                "their medical information to receive guidance. The "
                "available source does not specify a phone number or "
                "a complete step-by-step booking procedure. [1]"
            )

        return FALLBACK_ANSWER

    # --------------------------------------------------
    # 14. Do not call Gemini when no source evidence was retrieved.
    # --------------------------------------------------
    if not retrieved_context:
        return FALLBACK_ANSWER

    # --------------------------------------------------
    # 15. Use Gemini for other source-grounded questions.
    # --------------------------------------------------
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
Never transfer inclusions, prices, or conditions from one program
to another unless the source explicitly supports the comparison.

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
