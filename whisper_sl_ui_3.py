import streamlit as st
from typing import Generator
from groq import Groq
from supabase import create_client, Client

st.set_page_config(page_icon="💬", layout="wide",
                   page_title="Groq Goes Brrrrrrrr...")


def icon(emoji: str):
    """Shows an emoji as a Notion-style page icon."""
    st.write(
        f'<span style="font-size: 78px; line-height: 1">{emoji}</span>',
        unsafe_allow_html=True,
    )


# AI Whispers.. assistant that explains the speech from speakers and shows it;s correctness to the audience and help them understand the speech in simple terms as well allow them to clear their following doubts.

icon("👻")

st.subheader("Truth Whispers..")
st.markdown(
    '<p style="font-size: 18px;">AI whispers.. that explain the speech from speaker and shows its correctness to the audience.</p>',
    unsafe_allow_html=True
)
st.subheader("", divider="rainbow", anchor=False)

# Initialize the Groq client with the API key from secrets
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"],
)

# Initialize Supabase client
url: str = st.secrets["SUPABASE_URL"]
key: str = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

# Initialize chat history and selected model
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Define model details
models = {
    "llama2-70b-4096": {"name": "LLaMA2-70b-chat", "tokens": 4096, "developer": "Meta"},
    "llama3-70b-8192": {"name": "LLaMA3-70b-8192", "tokens": 8192, "developer": "Meta"},
    "llama3-8b-8192": {"name": "LLaMA3-8b-8192", "tokens": 8192, "developer": "Meta"},
    "mixtral-8x7b-32768": {"name": "Mixtral-8x7b-Instruct-v0.1", "tokens": 32768, "developer": "Mistral"},
    "gemma-7b-it": {"name": "Gemma-7b-it", "tokens": 8192, "developer": "Google"},
}

# Layout for model selection and max_tokens slider
col1, col2 = st.columns(2)

with col1:
    model_option = st.selectbox(
        "Choose a model:",
        options=list(models.keys()),
        format_func=lambda x: models[x]["name"],
        index=4  # Default to mixtral
    )

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

max_tokens_range = models[model_option]["tokens"]

with col2:
    # Adjust max_tokens slider dynamically based on the selected model
    max_tokens = st.slider(
        "Max Tokens:",
        min_value=512,  # Minimum value to allow some flexibility
        max_value=max_tokens_range,
        # Default value or max allowed if less
        value=min(32768, max_tokens_range),
        step=512,
        help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
    )

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])


def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def save_message_to_supabase(role, content):
    """Save a chat message to Supabase."""
    data = {"role": role, "content": content}
    response = supabase.table("messages").insert(data).execute()
    # print(response)
    # if response.get('error'):
    #     st.error(response['error'], icon="🚨")


def load_messages_from_supabase():
    """Load chat messages from Supabase."""
    response = supabase.table("messages").select("*").execute()
    # print(response.data)
    return response.data
    # if response.get('error'):
    #     st.error(response['error'], icon="🚨")
    # return response.data


# Sidebar for loading chat history
with st.sidebar:
    if st.button("Load Chat History (SupaBase)"):
        messages = load_messages_from_supabase()
        for message in messages:
            avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
            st.write(f"{avatar} {message['content']}")

if prompt := st.chat_input("Enter your prompt here..."):
    # Append user prompt with additional instructions for the model to act as a technical truth whisperer
    engineered_prompt = (
        "As a technical truth whisperer, provide accurate, detailed, and reliable information on the following topic. "
        "Evaluate the correctness of the original speech, indicating how much was correct or incorrect. "
        "Also, ask the user if they have any further questions about the response: "
        f"{prompt}"
    )

    st.session_state.messages.append({"role": "user", "content": engineered_prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(engineered_prompt)

    # Save user message to Supabase
    save_message_to_supabase("user", engineered_prompt)

    # Fetch response from Groq API
    try:
        chat_completion = client.chat.completions.create(
            model=model_option,
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ],
            max_tokens=max_tokens,
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = st.write_stream(chat_responses_generator)

        # Save assistant message to Supabase
        save_message_to_supabase("assistant", full_response)
    except Exception as e:
        st.error(e, icon="🚨")

    # Append the full response to session_state.messages
    if isinstance(full_response, str):
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})
    else:
        # Handle the case where full_response is not a string
        combined_response = "\n".join(str(item) for item in full_response)
        st.session_state.messages.append(
            {"role": "assistant", "content": combined_response})

        # Save combined assistant message to Supabase
        save_message_to_supabase("assistant", combined_response)
