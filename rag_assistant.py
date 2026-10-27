
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic


# Load API keys from the .env file
load_dotenv()

# Locate the existing Chroma database
VECTOR_STORE_DIR = Path(__file__).parent / "chroma_db"

# Configure the embedding model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Connect to the existing knowledge base
vector_store = Chroma(
    collection_name="seo_sop_knowledge",
    persist_directory=str(VECTOR_STORE_DIR),
    embedding_function=embeddings,
)

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0,
)


def retrieve_context(question: str, k: int = 3, candidate_k: int = 10):
    """Retrieve candidate SOP chunks and rerank them with Claude."""

    # Step 1: Retrieve a larger set of candidates from Chroma.
    candidates = vector_store.similarity_search_with_score(
        question,
        k=candidate_k,
    )

    if not candidates:
        print("No relevant SOP chunks found.")
        return []

    # Step 2: Prepare candidates for Claude.
    candidate_text = []

    for index, (document, distance) in enumerate(candidates):
        source = document.metadata.get("source_file", "Unknown source")
        section = document.metadata.get("section", "Unknown section")

        candidate_text.append(
            f"Candidate ID: {index}\n"
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Content:\n{document.page_content}"
        )

    candidates_context = "\n\n---\n\n".join(candidate_text)

    # Step 3: Ask Claude to score each candidate for relevance.
    rerank_prompt = f"""
You are a retrieval relevance evaluator for an SEO SOP knowledge base.

Score each candidate from 0 to 10 based on how directly and
usefully its content answers the user's question.

Scoring guide:
- 0: Completely irrelevant.
- 1-3: Weak relevance.
- 4-6: Partially relevant.
- 7-8: Relevant and useful.
- 9-10: Directly answers the question.

Rules:
- Judge the candidate's actual content, not just its filename.
- Do not reward a candidate merely because it comes from the expected SOP.
- Give every candidate exactly one integer score.
- Return valid JSON only, with no Markdown fences or extra commentary.
- Use this format:
{{"scores": [{{"candidate_id": 0, "score": 8}}]}}

USER QUESTION:
{question}

CANDIDATES:
{candidates_context}
"""

    response = llm.invoke(rerank_prompt)

    # Step 4: Parse and validate Claude's JSON response.
    import json

    raw_response = response.content.strip()

    if raw_response.startswith("```"):
        raw_response = raw_response.split("\n", 1)[1]
        raw_response = raw_response.rsplit("```", 1)[0].strip()

    score_data = json.loads(raw_response)
    scores = score_data["scores"]

    candidate_scores = {}

    for item in scores:
        candidate_id = item["candidate_id"]
        score = item["score"]

        if (
            not isinstance(candidate_id, int)
            or isinstance(candidate_id, bool)
            or not 0 <= candidate_id < len(candidates)
        ):
            raise ValueError(f"Invalid candidate ID: {candidate_id}")

        if candidate_id in candidate_scores:
            raise ValueError(f"Duplicate candidate ID: {candidate_id}")

        if (
            not isinstance(score, int)
            or isinstance(score, bool)
            or not 0 <= score <= 10
        ):
            raise ValueError(f"Invalid relevance score: {score}")

        candidate_scores[candidate_id] = score

    expected_ids = set(range(len(candidates)))

    if set(candidate_scores) != expected_ids:
        raise ValueError(
            "Claude did not return exactly one score for every candidate."
        )

    # Step 5: Sort candidates by relevance score.
    reranked = []

    for index, (document, distance) in enumerate(candidates):
        reranked.append(
            (
                document,
                distance,
                candidate_scores[index],
            )
        )

    reranked.sort(key=lambda item: item[2], reverse=True)

    # Step 6: Return only the requested number of top results.
    results = reranked[:k]

    print(f"\nQuestion: {question}")
    print(f"Candidates evaluated: {len(candidates)}")
    print(f"Chunks returned after reranking: {len(results)}")

    for rank, (document, distance, relevance_score) in enumerate(
        results,
        start=1,
    ):
        print(f"\n--- Reranked Chunk {rank} ---")
        print(f"Source: {document.metadata.get('source_file')}")
        print(f"Section: {document.metadata.get('section')}")
        print(f"Claude relevance score: {relevance_score}/10")
        print(f"Original vector distance: {distance:.4f}")
        print(f"Content:\n{document.page_content}")

    # Keep the original two-item tuple format so existing callers
    # can still unpack (document, distance).
    return [(document, distance) for document, distance, _ in results]


def generate_answer(question: str):
    """Generate an answer grounded in retrieved SOP content."""

    results = retrieve_context(
    question,
    k=3,
    candidate_k=10,
    )

    if not results:
        return "I could not find relevant information in the SOP documents."

    context_parts = []

    for document, distance in results:
        source = document.metadata.get("source_file", "Unknown source")
        section = document.metadata.get("section", "Unknown section")

        context_parts.append(
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Content:\n{document.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are an SEO knowledge assistant.

Answer the user's question using only the supplied SOP context.

Instructions:
- Answer only with information explicitly supported by the supplied SOP context.
- Do not use outside knowledge, assumptions, or general knowledge to fill gaps.
- If the context does not answer the question, say:
  "The supplied SOP documents do not contain enough information to answer this question."
- You may explain what the SOP does cover, but clearly distinguish it from missing information.
- Cite the source filename and section for factual claims.
- Do not invent facts, procedures, metrics, ranking factors, or numerical values.
- Do not present an inference as an established fact.
- Keep the answer clear and concise.

SOP CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = llm.invoke(prompt)

    return response.content

def generate_answer_with_sources(question: str):
    """Return the answer and the retrieved SOP sources."""

    results = retrieve_context(
        question,
        k=3,
        candidate_k=10,
    )

    if not results:
        return (
            "I could not find relevant information in the SOP documents.",
            [],
        )

    context_parts = []
    sources = []

    for document, distance in results:
        source = document.metadata.get("source_file", "Unknown source")
        section = document.metadata.get("section", "Unknown section")

        context_parts.append(
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Content:\n{document.page_content}"
        )

        sources.append({
            "source": source,
            "section": section,
            "content": document.page_content,
        })

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""
You are an SEO knowledge assistant.

Answer the user's question using only the supplied SOP context.

Instructions:
- Answer only with information explicitly supported by the supplied SOP context.
- Do not use outside knowledge, assumptions, or general knowledge to fill gaps.
- If the context does not answer the question, say:
  "The supplied SOP documents do not contain enough information to answer this question."
- You may explain what the SOP does cover, but clearly distinguish it from missing information.
- Cite the source filename and section for factual claims.
- Do not invent facts, procedures, metrics, ranking factors, or numerical values.
- Do not present an inference as an established fact.
- Keep the answer clear and concise.

SOP CONTEXT:
{context}

USER QUESTION:
{question}
"""

    response = llm.invoke(prompt)

    return response.content, sources

if __name__ == "__main__":
    question = "What is Google's exact ranking algorithm weight for canonical tags?"

    answer = generate_answer(question)

    print("\n--- RAG Assistant Answer ---\n")
    print(answer)