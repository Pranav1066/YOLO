import streamlit as st
from ultralytics import YOLO
from PIL import Image
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Set up Streamlit page
st.title("Lung Condition Detection and Diagnostic Assistant")
st.write("Upload a chest X-ray image to detect lung conditions and receive a diagnostic-style report. You can also ask questions about detected conditions.")

# Load YOLO model
yolo_model = YOLO(r'D:\image_prediction_project\best (1).pt')  # Update with the path to your trained model

llm_prompt = PromptTemplate(
    input_variables=["summary", "question"],
    template="""
    You are an expert pulmonologist providing advice to a person without a medical background.
    Based on the findings below:
    1. Explain each detected condition simply.
    2. Indicate whether it is mild, moderate, or severe.
    3. Suggest steps they could take, like visiting a doctor, monitoring symptoms, or taking preventive measures at home.

    Findings:
    {summary}

    Question:
    {question}
    """
)

# Set up ChatGroq with provided API key and model name
llm = ChatGroq(
    temperature=0,
    groq_api_key="",
    model="llama-3.1-70b-versatile"
)

llm_chain = LLMChain(llm=llm, prompt=llm_prompt)

# Image uploader
uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded X-ray Image", use_column_width=True)
    st.write("Processing...")

    # Convert the uploaded image to a format YOLO can use
    img = Image.open(uploaded_file) 
    img_path = 'uploaded_image.jpg'
    img.save(img_path)  # Save the uploaded image

    # Run YOLO model on the uploaded image
    yolo_results = yolo_model.predict(source=img_path, save=True)

    # Process YOLO detections into a summary for the ChatGroq prompt
    detections_summary = []
    for result in yolo_results:
        for box in result.boxes:
            class_name = yolo_model.names[int(box.cls)]
            confidence = box.conf.item()
            
            # Extract coordinates from the bounding box
            x_min, y_min, x_max, y_max = box.xyxy[0].tolist()  # Convert tensor to list for individual elements

            # Update the summary to include bounding box location
            location = f"bounding box ({x_min:.0f}, {y_min:.0f}, {x_max:.0f}, {y_max:.0f})"
            detections_summary.append(f"Condition: {class_name} detected at {location} with confidence {confidence:.2f}")

    yolo_summary = "\n".join(detections_summary)

    # Generate initial diagnostic report using ChatGroq
    initial_response = llm_chain.run(summary=yolo_summary, question="")

    # Display YOLO result with bounding boxes
    for result in yolo_results:
        img_with_boxes = result.plot()  # Overlay bounding boxes on the image

    # Convert to Streamlit-compatible format
    st.image(img_with_boxes, caption="Detected Conditions with Bounding Boxes", use_column_width=True)

    # Display ChatGroq's initial diagnostic report
    st.subheader("Diagnostic Report")
    st.write(initial_response)

    # Add ChatBot interaction in the sidebar
    st.sidebar.subheader("Ask the Diagnostic Assistant")

    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Suggested prompts for the user
    st.sidebar.write("### Suggested Questions")
    suggested_questions = [
        "What does this condition mean?",
        "Is this condition serious?",
        "What steps should I take?",
        "Can this condition improve on its own?",
        "What lifestyle changes could help?"
    ]

    # Display suggested questions as buttons
    for question in suggested_questions:
        if st.sidebar.button(question):
            st.session_state.chat_history.append({"role": "user", "content": question})
            response = llm_chain.run(summary=yolo_summary, question=question)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Input for custom user question
    user_question = st.sidebar.text_input("Or, ask your own question:")

    if user_question:
        # Append user question to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        
        # Generate LLM response based on detected conditions and user question
        response = llm_chain.run(summary=yolo_summary, question=user_question)

        # Append LLM response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

    # Display chat history in the sidebar
    st.sidebar.write("### Chat History")
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.sidebar.write(f"**You:** {chat['content']}")
        else:
            st.sidebar.write(f"**Assistant:** {chat['content']}")
