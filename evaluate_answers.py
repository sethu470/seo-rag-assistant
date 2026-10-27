TEST_CASES = [
    {
        "type": "direct",
        "question": (
            "How should I investigate pages that are not indexed by Google?"
        ),
        "expected_behavior": (
            "Give investigation steps supported by the indexation SOP "
            "and cite the source sections."
        ),
    },
    {
        "type": "combined_evidence",
        "question": (
            "How should I investigate a URL that is not indexed, "
            "has a canonical pointing elsewhere, and is missing from "
            "the XML sitemap?"
        ),
        "expected_behavior": (
            "Combine relevant evidence from the SOPs without claiming "
            "that any single signal proves the root cause."
        ),
    },
    {
        "type": "partial",
        "question": (
            "How should I investigate canonicalization problems, "
            "and what exact ranking weight does Google assign "
            "to canonical tags?"
        ),
        "expected_behavior": (
            "Explain the documented canonical auditing procedure, "
            "but explicitly state that the SOPs do not provide "
            "the exact ranking weight."
        ),
    },
    {
        "type": "unsupported",
        "question": (
            "What is Google's exact ranking algorithm weight "
            "for canonical tags?"
        ),
        "expected_behavior": (
            "State that the supplied SOP documents do not contain "
            "enough information to answer."
        ),
    },
]


def score_answer(test_case, answer, retrieved_results):
    """Use Claude to evaluate an answer against retrieved SOP evidence."""

    question = test_case["question"]
    expected_behavior = test_case["expected_behavior"]

    context_parts = []

    for document, distance in retrieved_results:
        source = document.metadata.get("source_file", "Unknown source")
        section = document.metadata.get("section", "Unknown section")

        context_parts.append(
            f"Source: {source}\n"
            f"Section: {section}\n"
            f"Content:\n{document.page_content}"
        )

    context = "\n\n---\n\n".join(context_parts)

    evaluation_prompt = f"""
You are an independent evaluator for a RAG-based SEO knowledge assistant.

Evaluate the answer using ONLY the supplied SOP context, the question,
and the expected behavior. Do not use outside knowledge.

QUESTION:
{question}

EXPECTED BEHAVIOR:
{expected_behavior}

RETRIEVED SOP CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Score each criterion from 1 to 5.

1. groundedness:
Are factual claims supported by the supplied context?
5 = Fully supported; 1 = Major unsupported claims.

2. completeness:
Does the answer address the question and expected behavior?
5 = Fully addresses them; 1 = Mostly fails to address them.

3. citation_quality:
Are cited source filenames and sections supported by the context?
5 = Accurate and specific; 1 = Missing or materially incorrect citations.
If the answer makes no factual claims requiring citations, explain this
in the reasoning and score appropriately.

4. abstention:
Does the answer correctly acknowledge information missing from the SOPs?
For answerable questions, score 5 if it answers rather than unnecessarily
refusing. For unsupported or partially answerable questions, score 5 if
it clearly identifies the missing information without inventing facts.
Use the expected behavior to determine what is appropriate.

Return valid JSON only, with this exact structure:
{{
  "groundedness": 1,
  "completeness": 1,
  "citation_quality": 1,
  "abstention": 1,
  "reasoning": {{
    "groundedness": "Brief explanation",
    "completeness": "Brief explanation",
    "citation_quality": "Brief explanation",
    "abstention": "Brief explanation"
  }},
  "manual_review": true
}}

Use integer scores from 1 to 5. Set manual_review to true if any
criterion scores 1 or 2, or if you cannot confidently judge the answer.
Otherwise set it to false.
"""

    response = llm.invoke(evaluation_prompt)
    raw_response = response.content.strip()

    if raw_response.startswith("```"):
        raw_response = raw_response.split("\n", 1)[1]
        raw_response = raw_response.rsplit("```", 1)[0].strip()

    evaluation = json.loads(raw_response)

    criteria = [
        "groundedness",
        "completeness",
        "citation_quality",
        "abstention",
    ]

    for criterion in criteria:
        score = evaluation[criterion]

        if (
            not isinstance(score, int)
            or isinstance(score, bool)
            or not 1 <= score <= 5
        ):
            raise ValueError(
                f"Invalid {criterion} score: {score}"
            )

    if not isinstance(evaluation.get("manual_review"), bool):
        raise ValueError("manual_review must be true or false")

    return evaluation


from rag_assistant import generate_answer, retrieve_context, llm
import json


def evaluate_answer(test_case):
    """Generate and display an answer for one evaluation case."""

    question = test_case["question"]

    print("\n" + "=" * 75)
    print(f"TEST TYPE: {test_case['type'].upper()}")
    print(f"QUESTION: {question}")

    answer = generate_answer(question)

    print("\nEXPECTED BEHAVIOR:")
    print(test_case["expected_behavior"])

    print("\nACTUAL ANSWER:")
    print(answer)

    return {
        "type": test_case["type"],
        "question": question,
        "answer": answer,
    }



if __name__ == "__main__":
    results = []

    for test_case in TEST_CASES:
        print("\n" + "=" * 70)
        print(f"TEST: {test_case['type'].upper()}")
        print(f"QUESTION: {test_case['question']}")

        # Retrieve the same reranked context used by the assistant.
        retrieved_results = retrieve_context(
            test_case["question"],
            k=3,
            candidate_k=10,
        )

        # Generate the answer using the existing RAG pipeline.
        answer = generate_answer(test_case["question"])

        # Evaluate the answer against the retrieved SOP context.
        evaluation = score_answer(
            test_case,
            answer,
            retrieved_results,
        )

        print("\nANSWER:")
        print(answer)

        print("\nEVALUATION:")
        print(json.dumps(evaluation, indent=2))

        results.append({
            "type": test_case["type"],
            "scores": evaluation,
        })

    print("\n" + "=" * 70)
    print("FINAL SCORECARD")

    criteria = [
        "groundedness",
        "completeness",
        "citation_quality",
        "abstention",
    ]

    for criterion in criteria:
        average = sum(
            result["scores"][criterion]
            for result in results
        ) / len(results)

        print(f"{criterion}: {average:.2f}/5")

    review_count = sum(
        result["scores"]["manual_review"]
        for result in results
    )

    print(f"Answers flagged for manual review: {review_count}/{len(results)}")

