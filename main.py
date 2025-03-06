import streamlit as st
import numpy as np
import PIL.Image
import google.generativeai as genai
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import chromadb
from dotenv import load_dotenv
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Configure Google Generative AI with the API key
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("GOOGLE_API_KEY not found in environment variables. Please set it in the .env file.")
    st.stop()
genai.configure(api_key=api_key)

# Custom CSS for the entire app
st.markdown(
    """
    <style>
    /* Lavender background for the entire app */
    .stApp {
        background-color: #E6E6FA;  /* Lavender */
    }

    /* Black text for all content */
    .stMarkdown, .stText, .stButton button, .stFileUploader div, .stTextInput input {
        color: #000000;  /* Black text */
    }

    /* Dark purple headings */
    h1, h2, h3, h4, h5, h6 {
        color: #4B0082;  /* Dark purple */
        font-family: 'Poppins', sans-serif;
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #4A90E2;  /* Vibrant blue */
        color: #FFFFFF;  /* White text */
    }

    /* Buttons */
    .stButton button {
        background-color: #2E86C1;  /* Blue */
        color:rgb(220, 219, 219);  /* White text */
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        font-family: 'Poppins', sans-serif;
        border: none;
        transition: background-color 0.3s ease;
    }
    .stButton button:hover {
        background-color: #1C6EA4;  /* Darker blue on hover */
    }

    /* File uploader */
    .stFileUploader div {
        background-color: #E0FFFF;  /* Light Cyan */
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #DDDDDD;
    }

    /* Text input */
    .stTextInput input {
        background-color: #E0FFFF;  /* Light Cyan */
        border-radius: 8px;
        padding: 10px;
        border: 1px solid #DDDDDD;
        color: #000000;  /* Black text */
    }

    /* Highlighted text */
    .highlight {
        background-color: #FFFFE0;  /* Light yellow */
        padding: 5px;
        border-radius: 4px;
        color: #000000;  /* Black text */
    }

    /* Image captions */
    .caption {
        color: #000000;  /* Black text */
        font-family: 'Poppins', sans-serif;
        font-size: 14px;
    }

    /* Team Members and Guide headings in bright color */
    .sidebar .sidebar-content .subheader {
        color:rgb(7, 192, 97);  /* Bright gold */
        font-family: 'Poppins', sans-serif;
    }

    /* Team Members and Guide names in thick colors */
    .team-member, .guide {
        color:rgb(212, 83, 147);  /* Tomato (thick red) */
        font-family: 'Poppins', sans-serif;
        font-weight: bold;
    }

    /* Virtual Fashion Assistant box background color */
    .stFileUploader div, .stTextInput input {
        background-color:rgb(173, 129, 159);  /* Light Cyan */
    }

    /* Full-slide image with overlay */
    .full-slide-image {
        position: relative;
        width: 100%;
        height: 100vh;  /* Full viewport height */
        overflow: hidden;
    }
    .full-slide-image img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .image-overlay {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        text-align: center;
        color: white;
        font-family: 'Poppins', sans-serif;
    }
    .image-overlay h1 {
        font-size: 4rem;  /* Larger font size */
        font-weight: bold;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.7);  /* Add shadow for better visibility */
        color: #00008B;  /* Thick blue color */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "About", "Virtual Fashion Assistant"])

# Team Members and Guide Details
st.sidebar.markdown("---")
st.sidebar.subheader("Team Members")
st.sidebar.markdown('<p class="team-member">CH.Sai Jeevana Jyothi</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="team-member">A.Jagruthi Mani</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="team-member">K.Durga Bhavani</p>', unsafe_allow_html=True)
st.sidebar.markdown('<p class="team-member">CH.Jahnavi</p>', unsafe_allow_html=True)
st.sidebar.subheader("Guide")
st.sidebar.markdown('<p class="guide">Abdul Aziz Md</p>', unsafe_allow_html=True)

# About Page
if page == "About":
    st.markdown("<h1 style='color:rgb(33, 61, 145);'>About the Project</h1>", unsafe_allow_html=True)  # Thick navy blue
    st.write("""
    This project is an AI-powered Fashion Styling Assistant that helps users get personalized fashion recommendations. 
    It uses advanced AI models like Google's Gemini and OpenCLIP to analyze images and provide styling advice.
    """)
    st.write("""
    Key Features:
    - Upload an image to find similar fashion items.
    - Enter a text query to get styling recommendations.
    - AI-generated fashion advice based on the latest trends.
    """)
    st.write("""
    Technologies Used:
    - Streamlit for the UI.
    - ChromaDB for vector storage and retrieval.
    - Google Gemini for AI-powered recommendations.
    - OpenCLIP for image embeddings.
    """)

# Virtual Fashion Assistant Page
elif page == "Virtual Fashion Assistant":
    st.markdown("<h1 style='color:rgb(141, 43, 163);'>Fashion Styling Assistant using AI</h1>", unsafe_allow_html=True)  # Thick peach color
    st.write("Enter your styling query and get image-based recommendations, or upload an image to retrieve similar images.")

    uploaded_file = st.file_uploader("Upload an image to retrieve similar images:", type=["jpg", "jpeg", "png"])
    query = st.text_input("Or, enter your styling query:")

    if st.button("Generate Styling Ideas / Retrieve Images"):
        chroma_client = chromadb.PersistentClient(path="Vector_database")
        image_loader = ImageLoader()
        CLIP = OpenCLIPEmbeddingFunction()
        image_vdb = chroma_client.get_or_create_collection(name="image", embedding_function=CLIP, data_loader=image_loader)

        if uploaded_file is not None:
            uploaded_image = np.array(PIL.Image.open(uploaded_file))
            retrieved_imgs = image_vdb.query(query_images=[uploaded_image], include=['data'], n_results=3)
            
            st.subheader("Retrieved Similar Images:")
            for i, img_data in enumerate(retrieved_imgs['data'][0]):
                try:
                    img = PIL.Image.fromarray(img_data.astype('uint8'))
                    st.image(img, caption=f"Retrieved Image {i+1}", use_container_width=True)
                    model = genai.GenerativeModel(model_name="gemini-1.5-pro")
                    prompt = ("You are a professional fashion and styling assistant with expertise in creating personalized outfit recommendations. "
                              "Analyze the provided image carefully and give detailed fashion advice, including how to style and complement this item. "
                              "Offer suggestions for pairing it with accessories, footwear, and other clothing pieces. "
                              "Focus on the specific design elements, colors, and texture of the clothing item in the image. "
                              "Based on the image, recommend how best to style this outfit to make a fashion statement.")
                    response = model.generate_content([prompt, img])
                    st.subheader("Styling Recommendations:")
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Error loading image {i+1}: {e}")
        
        if query:
            def query_db(query, results):
                return image_vdb.query(
                    query_texts=[query],
                    n_results=results,
                    include=['uris', 'distances'])
            
            results = query_db(query, results=2)
            image_paths = [results['uris'][0][0], results['uris'][0][1]]
            
            sample_file_1 = PIL.Image.open(image_paths[0])
            sample_file_2 = PIL.Image.open(image_paths[1])
            
            st.image(sample_file_1, caption="Image 1", use_container_width=True)
            st.image(sample_file_2, caption="Image 2", use_container_width=True)
            
            model = genai.GenerativeModel(model_name="gemini-1.5-pro")
            prompt = ("You are a professional fashion and styling assistant with expertise in creating personalized outfit recommendations. "
                       "Analyze the provided image carefully and give detailed fashion advice, including how to style and complement this item. "
                       "Offer suggestions for pairing it with accessories, footwear, and other clothing pieces. "
                       "Focus on the specific design elements, colors, and texture of the clothing item in the image. "
                       "This is the piece I want to wear: " + query + ". "
                       "Based on the image, recommend how best to style this outfit to make a fashion statement.")
            response = model.generate_content([prompt, sample_file_1, sample_file_2])
            st.subheader("Styling Recommendations:")
            st.write(response.text)

# Home Page
else:
    # Full-slide image with overlay
    st.markdown(
        """
        <div class="full-slide-image">
            <img src=https://th.bing.com/th/id/OIP.-ApvkQhlqqKN773G2w4nYwHaJl?w=792&h=1025&rs=1&pid=ImgDetMainalt="Fashion   Styling  Assistant">
            <div class="image-overlay">
                <h1>Fashion Styling Assistant</h1>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Description below the image
    st.write("""
    This is your one-stop solution for personalized fashion recommendations. 
    Use the navigation sidebar to explore the features of the app.
    """)