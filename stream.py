from __future__ import annotations
from typing import Literal, TypedDict
import asyncio
import os
import streamlit as st
import json
import logfire
from supabase import Client
# Remove OpenAI imports for embeddings
#from openai import AsyncOpenAI
from pydantic_ai_1 import Agent, RunContext


# Import SentenceTransformer for embeddings
from sentence_transformers import SentenceTransformer

# Import all the message part classes
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
    ModelMessagesTypeAdapter
)
from pydantic_ai_1 import pydantic_ai_expert, PydanticAIDeps

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Instead of an OpenAI client, we now instantiate a SentenceTransformer model.
embedding_model = SentenceTransformer('all-mpnet-base-v2')

supabase: Client = Client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

# Configure logfire to suppress warnings (optional)
logfire.configure(send_to_logfire='never')

class ChatMessage(TypedDict):
    """Format of messages sent to the browser/API."""
    role: Literal['user', 'model']
    timestamp: str
    content: str

def display_message_part(part):
    """
    Display a single part of a message in the Streamlit UI.
    Customize how you display system prompts, user prompts, tool calls, tool returns, etc.
    """
    if part.part_kind == 'system-prompt':
        with st.chat_message("system"):
            st.markdown(f"**System**: {part.content}")
    elif part.part_kind == 'user-prompt':
        with st.chat_message("user"):
            st.markdown(part.content)
    elif part.part_kind == 'text':
        with st.chat_message("assistant"):
            st.markdown(part.content)

async def run_agent_with_streaming(user_input: str):
    """
    Run the agent with streaming text for the user_input prompt,
    while maintaining the entire conversation in `st.session_state.messages`.
    """
    deps = PydanticAIDeps(
        supabase=supabase,
        embedding_model=embedding_model
    )

    async with pydantic_ai_expert.run_stream(
        user_input,
        deps=deps,
        message_history=st.session_state.messages[:-1],
    ) as result:
        partial_text = ""
        message_placeholder = st.empty()

        async for chunk in result.stream_text(delta=True):
            partial_text += chunk
            message_placeholder.markdown(partial_text)

        filtered_messages = [
            msg for msg in result.new_messages()
            if not (hasattr(msg, 'parts') and any(part.part_kind == 'user-prompt' for part in msg.parts))
        ]
        st.session_state.messages.extend(filtered_messages)
        st.session_state.messages.append(
            ModelResponse(parts=[TextPart(content=partial_text)])
        )

async def main():
    st.title("Bcommune AI")
    st.write("Retrieve resumes and ask questions...")

    # Initialize chat history in session state if not already present.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display all messages from the conversation.
    for msg in st.session_state.messages:
        if isinstance(msg, (ModelRequest, ModelResponse)):
            for part in msg.parts:
                display_message_part(part)

    # Chat input for the user.
    user_input = st.chat_input("What questions do you have about Resumes?")

    if user_input:
        # Append the new user prompt to the conversation.
        st.session_state.messages.append(
            ModelRequest(parts=[UserPromptPart(content=user_input)])
        )

        with st.chat_message("user"):
            st.markdown(user_input)

        # Run the agent (streaming the assistant’s response).
        with st.chat_message("assistant"):
            await run_agent_with_streaming(user_input)

if __name__ == "__main__":
    asyncio.run(main())
