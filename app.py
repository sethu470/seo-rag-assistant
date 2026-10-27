import streamlit as st
from rag_assistant import generate_answer_with_sources

st.set_page_config(
    page_title="SEO SOP Knowledge Assistant",
    page_icon="🔎",
    layout="centered",
)

st.title("SEO SOP Knowledge Assistant")

st.write(
    "Ask questions about your SEO and RAG standard operating procedures. "
    "Answers are grounded in your indexed SOP documents."
)

question = st.text_area(
    "Your question",
    placeholder="How should I investigate pages that are not indexed by Google?",
    height=100,
)

if st.button("Ask Assistant", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:

        with st.spinner("Searching SOPs and generating an answer..."):
            try:
                answer, sources = generate_answer_with_sources(
                    question.strip()
                )

                st.subheader("Answer")
                st.markdown(answer)

                st.subheader("Sources used")

                if sources:
                    for index, source in enumerate(sources, start=1):
                        with st.expander(
                            f"{index}. {source['source']} — {source['section']}"
                        ):
                            st.markdown(f"**Section:** {source['section']}")
                            st.markdown("**Retrieved passage:**")
                            st.write(source["content"])
                else:
                    st.info("No supporting SOP passages were retrieved.")

            except Exception:
                st.error(
                    "Something went wrong while generating the answer. "
                    "Check the VS Code terminal for details."
                )
                raise
