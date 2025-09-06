import streamlit as st
from loaders.llama_index_setup import query_data


def chat_panel(df, pre_prompt: str, state_key: str, title: str = "AI Assistant"):
    st.subheader(title)
    history_key = f"chat_{state_key}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []

    for role, content in st.session_state[history_key]:
        with st.chat_message(role):
            st.markdown(content)

    if user_input := st.chat_input("Ask a question about this view…"):
        st.session_state[history_key].append(("user", user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer = query_data(df, user_input, pre_prompt)
                except (RuntimeError, ValueError) as e:
                    answer = f"Sorry, I couldn't process that: {e}"
                st.markdown(answer)
                st.session_state[history_key].append(("assistant", answer))
