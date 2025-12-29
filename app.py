# =========================================
# app.py — Legal AI Assistant (Final Stable)
# =========================================

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# -----------------------------------------
# Page Config
# -----------------------------------------
st.set_page_config(
    page_title="Legal AI Assistant",
    page_icon="⚖️",
    layout="wide",
)

# -----------------------------------------
# Premium CSS
# -----------------------------------------
st.markdown("""
<style>

/* ---------- Global ---------- */
.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(135deg, #1e3a8a, #2563eb);
    padding: 1.6rem 2rem;
    border-radius: 18px;
    margin-bottom: 2rem;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    color: #ffffff;
}

.hero-subtitle {
    font-size: 0.95rem;
    color: #e0e7ff;
    margin-top: 0.3rem;
}

/* ---------- Cards ---------- */
.card {
    background-color: #ffffff;
    border-radius: 16px;
    padding: 1.4rem;
    border: 1px solid #e5e7eb;
    box-shadow: 0 10px 28px rgba(0,0,0,0.06);
}

/* ---------- Inputs ---------- */
textarea {
    border-radius: 14px !important;
    border: 1px solid #cbd5e1 !important;
    font-size: 1rem !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background-color: #2563eb;
    color: #ffffff;
    font-weight: 600;
    border-radius: 12px;
    padding: 0.7rem 1.3rem;
    border: none;
}

.stButton > button:hover {
    background-color: #1e40af;
}

/* ---------- Responses ---------- */
.response-box {
    background-color: #ffffff;
    color: #0f172a;
    padding: 1.5rem;
    border-radius: 16px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 12px 30px rgba(0,0,0,0.08);
    font-size: 1.05rem;
    line-height: 1.7;
    white-space: pre-wrap;
}

.response-title {
    font-weight: 700;
    margin-bottom: 0.6rem;
    color: #1e3a8a;
}

.small-muted {
    font-size: 0.85rem;
    color: #64748b;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------------
# Hero Header
# -----------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-title">⚖️ Legal AI Assistant</div>
    <div class="hero-subtitle">
        QLoRA fine-tuned Mistral-7B • Base vs LoRA • Prompt Presets
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------
# Load Models (Cached)
# -----------------------------------------
@st.cache_resource
def load_models():
    BASE_MODEL = "mistralai/Mistral-7B-v0.1"
    LORA_PATH = "./cs_lora_output_model/checkpoint-1500"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16
    ).eval()

    lora_model = PeftModel.from_pretrained(
        base_model,
        LORA_PATH
    ).eval()

    return tokenizer, base_model, lora_model

tokenizer, base_model, lora_model = load_models()

# -----------------------------------------
# Prompt Presets
# -----------------------------------------
PRESETS = {
    "Explain Consideration (Basics)": 
        "Explain the concept of consideration in contract law with a simple example.",

    "Void vs Voidable Contract":
        "Differentiate between void and voidable contracts under Indian Contract Law.",

    "Coercion Scenario (Reasoning)":
        "A contract was signed under coercion. Step by step, analyze whether the contract is enforceable.",

    "Draft Legal Notice":
        "Draft a formal legal notice for non-payment of dues under a service agreement.",

    "Force Majeure Clause":
        "Explain the force majeure clause and when it can be invoked, with an example.",

    "Custom Prompt":
        ""
}

# -----------------------------------------
# Layout
# -----------------------------------------
col1, col2 = st.columns([1, 2.4])

# -----------------------------------------
# Controls Card
# -----------------------------------------
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚙️ Controls")

    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max tokens", 50, 600, 300, 50)
    compare_mode = st.checkbox("🔄 Compare Base vs LoRA", value=True)

    st.markdown("<div class='small-muted'>Lower temperature → more factual</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------
# Prompt Card + Presets
# -----------------------------------------
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("✍️ Prompt")

    selected_preset = st.selectbox(
        "Choose a preset",
        list(PRESETS.keys())
    )

    prompt = st.text_area(
        "",
        value=PRESETS[selected_preset],
        height=160,
        placeholder="Ask a legal question, draft a notice, or test reasoning..."
    )

    generate = st.button("🚀 Generate Response", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------
# Correct Generation Logic (NO BROKEN OUTPUT)
# -----------------------------------------
def generate_response(model, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    return response.strip()

# -----------------------------------------
# Output Section
# -----------------------------------------
if generate:
    if not prompt.strip():
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("🧠 Analyzing…"):

            if compare_mode:
                col_base, col_lora = st.columns(2)

                with col_base:
                    base_response = generate_response(base_model, prompt)
                    st.markdown("<div class='response-title'>🧱 Base Model</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='response-box'>{base_response}</div>", unsafe_allow_html=True)

                with col_lora:
                    lora_response = generate_response(lora_model, prompt)
                    st.markdown("<div class='response-title'>✨ LoRA Fine-Tuned</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='response-box'>{lora_response}</div>", unsafe_allow_html=True)

            else:
                response = generate_response(lora_model, prompt)
                st.markdown("<div class='response-title'>🤖 Model Response</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='response-box'>{response}</div>", unsafe_allow_html=True)

# -----------------------------------------
# Footer
# -----------------------------------------
st.divider()
st.caption("Streamlit • Mistral-7B • QLoRA • PEFT • Lightning AI")
