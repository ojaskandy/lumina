import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from google import genai
import tempfile
from dotenv import load_dotenv
import re

load_dotenv()
# Initialize FastAPI app
app = FastAPI(
    title="Gemini Image Analysis API",
    description="A backend service that analyzes images using Google's Gemini AI",
    version="1.0.0"
)

# Configure Gemini AI
# Make sure to set your GOOGLE_API_KEY environment variable
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

# Initialize the Gemini client
client = genai.Client(api_key=API_KEY)

def strip_markdown(text: str) -> str:
    """Remove common markdown formatting and clean for TTS output"""

    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove bold and italic formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)   # **bold**
    text = re.sub(r'__([^_]+)__', r'\1', text)       # __bold__
    text = re.sub(r'\*([^*]+)\*', r'\1', text)       # *italic*
    text = re.sub(r'_([^_]+)_', r'\1', text)         # _italic_

    # Remove headers
    text = re.sub(r'#+\s*', '', text)

    # Remove links but keep the link text
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Remove bullet points and numbered lists
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Replace actual newline characters with a space
    text = text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ')

    # Remove extra whitespace
    text = re.sub(r'\s{2,}', ' ', text)

    # Final cleanup
    return text.strip()

class ImageAnalysisResponse(BaseModel):
    description: str
    filename: str

@app.get("/")
async def root():
    return {
        "message": "Gemini Image Analysis API", 
        "status": "running",
        "endpoints": {
            "/analyze": "POST - Analyze an image from URL",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "gemini-2.5-flash"}

@app.post("/analyze", response_model=ImageAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(..., description="Image file to analyze"),
    prompt: str = "You are assisting a visually impaired person in understanding and navigating a room based on an image. From the viewer's perspective (as if standing at the entrance), describe the overall layout clearly and naturally. Mention where key furniture pieces are placed in relation to the person. Then, identify the location of a specific object, such as a chair, and explain whether anything is blocking the path to it. Describe how the person can reach that object—whether the path is clear or if they need to move around obstacles. Keep the language simple, brief, and suitable for audio guidance."
):
    """
    Analyze an uploaded image file using Google's Gemini AI
    """
    try:
        # Validate file type
        content_type = file.content_type.lower() if file.content_type else ""
        
        # Check if content type is an image
        if not content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail=f"File is not an image. Content-Type: {content_type}")
        
        # Check if image format is supported by Gemini
        supported_formats = {
            'image/png': '.png',
            'image/jpeg': '.jpg',
            'image/jpg': '.jpg',
            'image/webp': '.webp',
            'image/heic': '.heic',
            'image/heif': '.heif'
        }
        
        extension = None
        for format_type, ext in supported_formats.items():
            if format_type in content_type:
                extension = ext
                break
        
        if not extension:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported image format: {content_type}. Supported formats: PNG, JPEG, WEBP, HEIC, HEIF"
            )
        
        # Read file content
        image_data = await file.read()
        
        # Check file size (20MB limit for inline data)
        if len(image_data) > 20 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Image file is too large. Maximum size is 20MB."
            )
        
        # Save image temporarily for Gemini processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as tmp_file:
            tmp_file.write(image_data)
            temp_path = tmp_file.name
        
        try:
            # Upload image to Gemini
            uploaded_file = client.files.upload(file=temp_path)
            
            # Generate content using Gemini
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[uploaded_file, prompt]
            )
            
            # Clean up temporary file
            os.unlink(temp_path)
            
            # Debug: Print original response
            # print("Original Gemini response:")
            # print(repr(response.text[:200]))
            
            # Strip markdown formatting from response
            clean_text = strip_markdown(response.text)
            
            # Debug: Print cleaned response
            # print("Cleaned response:")
            # print(repr(clean_text[:200]))
            
            return ImageAnalysisResponse(
                description=clean_text,
                filename=file.filename
            )
            
        except Exception as e:
            # Clean up temporary file in case of error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            raise HTTPException(status_code=500, detail=f"Gemini AI analysis failed: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 