import os
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import numpy as np
import PIL.Image

# Load environment variables from .env file
load_dotenv()

# Get the API key from the environment variables
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in the .env file.")

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="Vector_database")

# Initialize OpenCLIP embedding function and image loader
CLIP = OpenCLIPEmbeddingFunction()
image_loader = ImageLoader()

# Create or get a collection in ChromaDB
image_vdb = chroma_client.get_or_create_collection(
    name="image",
    embedding_function=CLIP,
    data_loader=image_loader
)

# Function to load and process images from a directory
def load_images_from_directory(directory_path):
    """
    Load images from a directory, process them, and store them in ChromaDB.
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    # List all image files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        raise ValueError(f"No images found in directory: {directory_path}")

    # Load and process each image
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        try:
            # Load the image
            image = np.array(PIL.Image.open(image_path))

            # Add the image to ChromaDB
            image_vdb.add(
                ids=[image_file],  # Use the filename as the ID
                images=[image]     # Pass the image as a NumPy array
            )
            print(f"Processed and added image: {image_file}")
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")

# Main function to load data
def main():
    # Specify the directory containing your images
    image_directory = "path/to/your/image/folder"  # Replace with your actual directory path

    # Load images from the directory
    load_images_from_directory(image_directory)
    print("Data loading complete.")

# Run the main function
if __name__ == "__main__":
    main()

