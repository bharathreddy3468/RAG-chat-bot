import streamlit as st
from streamlit_chat import message
from app import generate_response


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
query = st.chat_input("Your message")

if query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)

    # Simulate bot response (replace with your actual logic)
    bot_response = generate_response(query)
    # Add bot message to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    # Display bot message
    with st.chat_message("assistant"):
        st.markdown(bot_response)