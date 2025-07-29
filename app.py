import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load the model (GPT-2 or GPT-Neo)
@st.cache_resource
def load_model(model_name='gpt2'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
    return generator

# Generate story based on prompt and genre
def generate_story(generator, prompt, genre, num_outputs=3):
    story_prompt = f"{genre} Story: {prompt}"
    return generator(
        story_prompt,
        max_length=200,
        num_return_sequences=num_outputs,
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.1
    )

# Streamlit App UI
st.set_page_config(page_title="AI Dungeon Story Generator", layout="centered")
st.title("🧙‍♂️ AI Dungeon Story Generator")

# Genre selection
genres = ['Fantasy', 'Mystery', 'Sci-Fi', 'Horror', 'Adventure']
selected_genre = st.selectbox("Choose a Genre", genres)

# Prompt input
user_prompt = st.text_area("Enter your story prompt:", height=150)

# Number of variations
num_variants = st.slider("Number of continuations", 1, 5, 3)

# Load model button
if 'generator' not in st.session_state:
    st.session_state.generator = load_model("gpt2")  # or "EleutherAI/gpt-neo-1.3B"

# Generate stories
if st.button("Generate Story"):
    with st.spinner("Generating..."):
        outputs = generate_story(st.session_state.generator, user_prompt, selected_genre, num_outputs=num_variants)
        st.session_state.generated_stories = outputs

# Show results
if 'generated_stories' in st.session_state:
    st.subheader("✨ Story Continuations:")
    for idx, output in enumerate(st.session_state.generated_stories):
        st.markdown(f"**Option {idx+1}:**")
        st.write(output['generated_text'])
        st.markdown("---")

    # Save button
    if st.button("💾 Save First Story"):
        story_text = st.session_state.generated_stories[0]['generated_text']
        with open("generated_story.txt", "w", encoding="utf-8") as f:
            f.write(story_text)
        st.success("Story saved as `generated_story.txt`!")
