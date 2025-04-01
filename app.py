import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import json
import spacy
import random
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage, Table, TableStyle

# Set up the page configuration
st.set_page_config(
    page_title="DeepShrimp - Disease Detection",
    layout="wide",
    page_icon="🦐",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
/* Centering the image */
.centered {
    display: flex;
    justify-content: center;
}
/* Chat container */
.chat-container {
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 400px;
    height: 500px;
    background-color: white;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    transition: all 0.3s ease;
    z-index: 1000;
}
/* Chat header */
.chat-header {
    background-color: #0083B8;
    color: white;
    padding: 15px;
    font-weight: bold;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
/* Chat messages container */
.chat-messages {
    flex-grow: 1;
    overflow-y: auto;
    padding: 15px;
    display: flex;
    flex-direction: column;
}
/* Individual chat messages */
.chat-message {
    margin-bottom: 10px;
    padding: 10px;
    border-radius: 15px;
    max-width: 80%;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    animation: fadeIn 0.5s;
}
.user {
    align-self: flex-end;
    background-color: #DCF8C6;
}
.assistant {
    align-self: flex-start;
    background-color: #F1F0F0;
}
/* Card styling */
.card {
    border-radius: 10px;
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    padding: 20px;
    margin-bottom: 20px;
    background-color: white;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}
/* Centering the logo */
.logo {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'chat_visible' not in st.session_state:
    st.session_state.chat_visible = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = {
        'original_image': None,
        'processed_image': None,
        'disease_name': None,
        'severity_score': None,
        'impact_fig': None,
        'market_data': None
    }
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = []

# Load resources
@st.cache_resource
def load_nlp_model():
    return spacy.load("en_core_web_md")

@st.cache_resource
def load_knowledge_base():
    try:
        with open("shrimp_knowledge.json", "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "intents": [],
            "default_responses": ["I'm sorry, I cannot assist at the moment."]
        }

knowledge_base = load_knowledge_base()

@st.cache_resource
def load_model():
    try:
        return YOLO("best.pt")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
# Display the logo at the top of the app (centered)
logo_path = "logo.jpg"  # Replace with the actual path to your logo
logo = Image.open(logo_path)

# Use Streamlit columns to center the logo
col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the column ratios as needed
with col2:
    st.image(logo, use_column_width="auto", width=300)  # Adjust width as needed
# Helper functions
def generate_graph(severity, disease_name):
    if "white spot" in disease_name.lower():
        params = ["Oxygen (mg/L)", "pH", "Infections (%)", "Mortality (%)", "Market Loss (%)"]
        values = [
            np.clip(4.0 - (severity / 30), 2.0, 5.0),
            np.clip(7.5 - (severity / 50), 6.5, 8.0),
            np.clip(severity * 1.2, 10, 100),
            np.clip(severity * 1.5, 5, 100),
            np.clip(severity * 1.8, 10, 100)
        ]
    elif "black gill" in disease_name.lower():
        params = ["Oxygen (mg/L)", "pH", "Infections (%)", "Mortality (%)", "Market Loss (%)"]
        values = [
            np.clip(3.5 - (severity / 40), 1.5, 5.0),
            np.clip(7.2 - (severity / 60), 6.5, 8.0),
            np.clip(severity * 0.9, 10, 90),
            np.clip(severity * 1.0, 5, 80),
            np.clip(severity * 1.2, 10, 90)
        ]
    elif "black spot" in disease_name.lower():
        params = ["Oxygen (mg/L)", "pH", "Infections (%)", "Mortality (%)", "Market Loss (%)"]
        values = [
            np.clip(4.5 - (severity / 50), 3.0, 5.0),
            np.clip(7.0 - (severity / 70), 6.5, 8.0),
            np.clip(severity * 0.7, 10, 70),
            np.clip(severity * 0.5, 5, 50),
            np.clip(severity * 1.5, 10, 80)
        ]
    else:  # Default or Healthy
        params = ["Oxygen (mg/L)", "pH", "Infections (%)", "Mortality (%)", "Market Loss (%)"]
        values = [5.0, 7.8, 5.0, 2.0, 0.0]

    sns.set(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.barplot(x=params, y=values, hue=params, palette="Blues_d", legend=False)
    ax2 = ax.twinx()
    ax2.plot(params, values, color='red', marker='o', linestyle='dashed', linewidth=2, markersize=8)
    plt.title(f"Impact Analysis for {disease_name}: {severity}% Severity", fontsize=14, fontweight='bold')
    ax.set_ylabel("Level / Percentage")
    ax2.set_ylabel("Trend Line")
    plt.tight_layout()
    return fig

def calculate_losses(severity, disease_name):
    if "white spot" in disease_name.lower():
        mortality_factor = 1.8
        market_factor = 1.5
    elif "black gill" in disease_name.lower():
        mortality_factor = 1.2
        market_factor = 1.0
    elif "black spot" in disease_name.lower():
        mortality_factor = 0.6
        market_factor = 1.3
    else:  # Default or Healthy
        mortality_factor = 0.1
        market_factor = 0.1

    severity_scale = severity / 100

    data = {
        "Farming Type": ["Semi-Intensive", "Intensive"],
        "Stocking Density": ["100–150 shrimp/m²", "200–300 shrimp/m²"],
        "Mortality Loss": [f"₹{int(80000 * severity_scale * mortality_factor):,}", f"₹{int(200000 * severity_scale * mortality_factor):,}"],
        "Market Value Loss": [f"₹{int(70000 * severity_scale * market_factor):,}", f"₹{int(150000 * severity_scale * market_factor):,}"],
        "Loss Without Detection": [f"₹{int(195000 * severity_scale * (mortality_factor + market_factor)/2):,}", f"₹{int(440000 * severity_scale * (mortality_factor + market_factor)/2):,}"],
        "Savings With Detection": [f"₹{int(145000 * severity_scale * (mortality_factor + market_factor)/2):,}", f"₹{int(340000 * severity_scale * (mortality_factor + market_factor)/2):,}"]
    }
    return pd.DataFrame(data)

def find_best_intent(user_message):
    nlp = load_nlp_model()
    user_doc = nlp(user_message.lower())
    best_intent = None
    best_similarity = 0.0

    for intent in knowledge_base['intents']:
        for pattern in intent['patterns']:
            pattern_doc = nlp(pattern.lower())
            similarity = user_doc.similarity(pattern_doc)

            if similarity > best_similarity:
                best_similarity = similarity
                best_intent = intent

    return best_intent if best_similarity > 0.6 else None

def process_chat_input(user_input):
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Check if the input is related to current analysis
    if st.session_state.analysis_data['disease_name'] and any(word in user_input.lower() for word in ["detected", "found", "analysis", "current", "this", "what", "report"]):
        response = f"Based on the current analysis, we detected {st.session_state.analysis_data['disease_name']} with {st.session_state.analysis_data['severity_score']}% severity. I recommend reviewing the detailed report for specific recommendations."
    else:
        # Find the best matching intent from our knowledge base
        best_intent = find_best_intent(user_input)

        if best_intent:
            response = random.choice(best_intent['responses'])
        else:
            response = random.choice(knowledge_base['default_responses'])

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    return response

def generate_pdf(disease_name, severity_score, recommendations, file_path):
    # Create a PDF document
    doc = SimpleDocTemplate(file_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title = Paragraph("Shrimp Disease Analysis Report", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))

    # Disease Information
    disease_info = f"Disease Name: {disease_name}<br/>Severity Score: {severity_score}%"
    story.append(Paragraph(disease_info, styles['Normal']))
    story.append(Spacer(1, 12))

    # Processed Image
    if st.session_state.analysis_data['processed_image']:
        processed_img_path = "processed_image.png"
        st.session_state.analysis_data['processed_image'].save(processed_img_path)
        story.append(Paragraph("Processed Image:", styles['Normal']))
        story.append(PDFImage(processed_img_path, width=400, height=300))
        story.append(Spacer(1, 12))

    # Impact Analysis Graph
    if st.session_state.analysis_data['impact_fig']:
        impact_fig_path = "impact_fig.png"
        st.session_state.analysis_data['impact_fig'].savefig(impact_fig_path)
        story.append(Paragraph("Impact Analysis Graph:", styles['Normal']))
        story.append(PDFImage(impact_fig_path, width=400, height=300))
        story.append(Spacer(1, 12))

    # Cost-Benefit Analysis Table
    if st.session_state.analysis_data['market_data'] is not None:
        story.append(Paragraph("Cost-Benefit Analysis:", styles['Normal']))
        data = [st.session_state.analysis_data['market_data'].columns.tolist()] + st.session_state.analysis_data['market_data'].values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(table)
        story.append(Spacer(1, 12))

    # Additional Disease Information
    if disease_name in knowledge_base:
        disease_info = knowledge_base[disease_name]
        story.append(Paragraph("Additional Disease Information:", styles['Normal']))
        for key, value in disease_info.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    story.append(Paragraph(f"{sub_key}: {sub_value}", styles['Normal']))
            else:
                story.append(Paragraph(f"{key}: {value}", styles['Normal']))
        story.append(Spacer(1, 12))

    # Recommendations
    story.append(Paragraph("Recommendations:", styles['Normal']))
    for rec in recommendations:
        story.append(Paragraph(f"- {rec}", styles['Normal']))
    story.append(Spacer(1, 12))

    # Build the PDF
    doc.build(story)

# Main app content
def main_content():
    # Introduction card
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    Welcome to DeepShrimp, an advanced AI-powered tool for early detection of shrimp diseases. This application helps shrimp farmers identify diseases quickly, assess their severity, and understand the potential economic impact.

    How to use:
    1. Upload a clear image of your shrimp showing any visible symptoms
    2. Click "Analyze Image" to detect diseases
    3. Review the detailed analysis and recommendations
    4. Use the chat assistant for additional guidance
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Add tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Disease Detection", "Historical Data", "Resources", "Chat with Us"])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Shrimp Image Analysis")
        uploaded_file = st.file_uploader("Upload shrimp image for disease detection", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.session_state.analysis_data['original_image'] = image

            # Create two columns for the images
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="centered">', unsafe_allow_html=True)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            analyze_button = st.button("Analyze Image", key="analyze_button")

            if analyze_button:
                with st.spinner("Detecting diseases... Please wait."):
                    model = load_model()
                    if model:
                        try:
                            image_cv2 = np.array(image)
                            image_cv2 = cv2.cvtColor(image_cv2, cv2.COLOR_RGB2BGR)
                            results = model(image_cv2)

                            detected_objects = results[0].boxes.data.cpu().numpy()
                            annotated_image = results[0].plot()
                            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                            processed_img = Image.fromarray(annotated_image)
                            st.session_state.analysis_data['processed_image'] = processed_img

                            with col2:
                                st.image(processed_img, caption="Processed Image", use_container_width=True)

                            if len(detected_objects) > 0:
                                most_severe = max(detected_objects, key=lambda x: x[4])
                                class_id = int(most_severe[5])

                                # Extract disease name from the text on the bounding box
                                disease_name = results[0].names[class_id]
                                severity_score = round(float(most_severe[4]) * 100, 2)

                                # For healthy shrimp, set a low severity
                                if "healthy" in disease_name.lower():
                                    severity_score = min(severity_score, 10)

                                st.session_state.analysis_data.update({
                                    'disease_name': disease_name,
                                    'severity_score': severity_score,
                                    'impact_fig': generate_graph(severity_score, disease_name),
                                    'market_data': calculate_losses(severity_score, disease_name)
                                })

                                if "healthy" in disease_name.lower():
                                    st.success(f"Good news! Detected {disease_name} with {severity_score}% confidence")
                                else:
                                    st.warning(f"Detected: {disease_name} with {severity_score}% confidence")

                                # Recommendations based on disease
                                st.subheader("Recommended Actions")
                                recommendations = {
                                    "White Spot": [
                                        "Isolate infected shrimp immediately",
                                        "Improve water quality with water exchanges (10-15% daily)",
                                        "Consider emergency harvest if severity > 60%",
                                        "Apply biosecurity measures to prevent spread"
                                    ],
                                    "Black Gill": [
                                        "Improve oxygenation in pond water",
                                        "Consider approved antifungal treatments",
                                        "Reduce feeding rate temporarily",
                                        "Monitor water parameters closely"
                                    ],
                                    "Black Spot": [
                                        "Maintain optimal water pH (7.5-8.5)",
                                        "Apply probiotics to boost immune response",
                                        "Ensure balanced nutrition with vitamin C supplements",
                                        "Reduce stocking density if possible"
                                    ],
                                    "Healthy": [
                                        "Continue regular water quality monitoring",
                                        "Maintain current feeding regimen",
                                        "Apply preventive probiotics as scheduled",
                                        "Monitor for any changes in behavior or appearance"
                                    ],
                                    "Default": [
                                        "Monitor water quality parameters daily",
                                        "Reduce stress factors in the pond",
                                        "Consult a shrimp health specialist",
                                        "Implement biosecurity measures"
                                    ]
                                }

                                found_recommendations = None
                                for key in recommendations:
                                    if key.lower() in disease_name.lower():
                                        found_recommendations = recommendations[key]
                                        break

                                if not found_recommendations:
                                    found_recommendations = recommendations["Default"]

                                for rec in found_recommendations:
                                    st.write(f"• {rec}")

                                # Impact analysis with explanation
                                st.subheader("Impact Analysis")
                                impact_cols = st.columns([2, 1])
                                with impact_cols[0]:
                                    st.pyplot(st.session_state.analysis_data['impact_fig'])
                                with impact_cols[1]:
                                    if "healthy" in disease_name.lower():
                                        st.markdown(f"""
                                        Good Health Indicators:

                                        Your shrimp appear healthy with:

                                        - Optimal water parameter ranges
                                        - Minimal risk of infection spread
                                        - Low mortality risk
                                        - Strong market potential

                                        Continue your current management practices!
                                        """)
                                    else:
                                        st.markdown(f"""
                                        Impact of {disease_name}:

                                        - Increased mortality risk
                                        - Potential market losses
                                        - Need for immediate action to prevent further spread
                                        """)

                                # Cost-Benefit Analysis
                                st.subheader("Cost-Benefit Analysis")
                                cost_benefit_df = st.session_state.analysis_data['market_data']
                                st.dataframe(cost_benefit_df)

                                # Generate PDF report
                                pdf_file_path = "shrimp_analysis_report.pdf"
                                generate_pdf(st.session_state.analysis_data['disease_name'], 
                                             st.session_state.analysis_data['severity_score'], 
                                             found_recommendations, 
                                             pdf_file_path)

                                # Provide a download link
                                with open(pdf_file_path, "rb") as f:
                                    st.download_button("Download PDF Report", f, file_name="shrimp_analysis_report.pdf")

                                # Save historical data
                                st.session_state.historical_data.append({
                                    'image': st.session_state.analysis_data['original_image'],
                                    'disease_name': st.session_state.analysis_data['disease_name'],
                                    'severity_score': st.session_state.analysis_data['severity_score'],
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })
                            else:
                                st.success("No diseases detected!")
                                st.session_state.analysis_data.update({
                                    'disease_name': None,
                                    'severity_score': None,
                                    'impact_fig': None,
                                    'market_data': None
                                })
                        except Exception as e:
                            st.error(f"Error during analysis: {e}")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Historical Data")

        if st.session_state.historical_data:
            for record in st.session_state.historical_data:
                st.image(record['image'], caption=f"Detected: {record['disease_name']} with {record['severity_score']}% severity", use_container_width=True)
                st.write(f"Timestamp: {record['timestamp']}")
                st.write("---")
        else:
            st.write("No historical data available.")

        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Resources")
        st.write("Here are some useful resources for shrimp farming:")
        st.markdown("""
        - [Global Aquaculture Alliance](https://www.aquaculturealliance.org/)
        - [Shrimp News International](http://www.shrimpnews.com/)
        - [FAO Aquaculture](http://www.fao.org/aquaculture/en/)
        - [Aquaculture Research](https://www.aquaculture-research.com/)
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Chat with Us")
        # Chatbot integration
        user_input = st.text_input("Type your message...", key="chat_input")
        if st.button("Send"):
            if user_input:
                response = process_chat_input(user_input)
                st.write(f"Assistant: {response}")
        st.markdown('</div>', unsafe_allow_html=True)

# Main app logic
def main():
    main_content()

if __name__ == "__main__":
    main()