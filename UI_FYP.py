# 1. Environment setup MUST COME FIRST
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

# 2. PyTorch imports and workaround
import torch
torch.__streamlit__ = False  # Block Streamlit's class inspection

# 3. Other imports
import streamlit as st
import torch.nn as nn
from transformers import AutoModel, BertTokenizerFast

# --------------------------
# STREAMLIT CONFIG
# --------------------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide"
)

# --------------------------
# MODEL LOADING (Cached)
# --------------------------
@st.cache_resource
def load_model():
    # Load BERT base model
    bert = AutoModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    # Model architecture
    class BERT_Arch(nn.Module):
        def __init__(self, bert):
            super().__init__()
            self.bert = bert
            self.dropout = nn.Dropout(0.1)
            self.fc1 = nn.Linear(768, 512)
            self.fc2 = nn.Linear(512, 2)
            self.softmax = nn.LogSoftmax(dim=1)
        
        def forward(self, sent_id, mask):
            cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
            x = self.fc1(cls_hs)
            x = torch.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.softmax(x)
            return x
    
    # Initialize and load weights
    model = BERT_Arch(bert)
    model.load_state_dict(torch.load(
        r'c1_fakenews_weights.pt',
        map_location=torch.device('cpu')  # Ensures CPU compatibility
    ))
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# --------------------------
# STREAMLIT UI
# --------------------------
# Custom styling
st.markdown("""
<style>
    .reportview-container { background: #f0f2f6 }
    .stTextArea textarea { font-size: 16px !important; padding: 10px !important; }
    .stButton>button { background: #4CAF50; color: white; font-weight: bold; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🔍 Fake News Detector")
st.markdown("---")

# Example selector
examples = [
    "Select an example...",
    "COVID vaccines contain microchips to track people",  # Fake
    "NASA announces new Mars rover mission",              # Real
    "World leaders sign global climate agreement",        # Real
    "5G networks spread coronavirus"                      # Fake
]

selected_example = st.selectbox("Try a sample text:", examples)

# Text input
input_text = st.text_area(
    "**Enter news text to analyze:**",
    height=150,
    value=selected_example if selected_example != examples[0] else ""
)

# Prediction logic
if st.button("🔎 Analyze Text", use_container_width=True):
    if not input_text.strip():
        st.warning("Please enter some text to analyze!")
    else:
        with st.spinner("Analyzing..."):
            # Tokenization
            tokens = tokenizer.batch_encode_plus(
                [input_text],
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Get predictions
            with torch.no_grad():
                logits = model(tokens['input_ids'], tokens['attention_mask'])
                probs = torch.exp(logits).numpy()[0]
            
            fake_prob = probs[0] * 100
            real_prob = probs[1] * 100
            
            # Display results
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Results")
                if real_prob > fake_prob:
                    st.success(f"✅ Real News ({real_prob:.1f}% confidence)")
                else:
                    st.error(f"❌ Fake News ({fake_prob:.1f}% confidence)")
            
            with col2:
                st.subheader("📈 Confidence Scores")
                st.metric("Real News Probability", f"{real_prob:.1f}%")
                st.metric("Fake News Probability", f"{fake_prob:.1f}%")

# Footer
st.markdown("---")
st.caption("Built with BERT and Streamlit | Fake News Detection System")
