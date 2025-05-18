import base64
import io
import os
import time
import traceback
import sys
import socket
import subprocess
from flask import Flask, request, send_file, jsonify, Blueprint
from PIL import Image
import requests
import httpx
from openai import OpenAI
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Log environment setup
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.critical("OPENAI_API_KEY environment variable is not set!")
    raise ValueError("OPENAI_API_KEY environment variable is not set!")
else:
    logger.info(f"OPENAI_API_KEY is set (length: {len(api_key)})")

# Initialize OpenAI client with custom HTTP client and increased timeout
logger.info("Initializing OpenAI client with custom HTTP client")
try:
    http_client = httpx.Client(
        timeout=120.0,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        http2=False  # Disabled HTTP/2
    )
    logger.info(f"HTTPX client initialized with timeout: 120.0s, HTTP/2: disabled")
    
    # Add request and response logging to HTTPX
    def log_request(request):
        logger.debug(f"Making request: {request.method} {request.url}")
        return request

    def log_response(response):
        logger.debug(f"Received response: {response.status_code} from {response.url}")
        return response

    http_client.event_hooks = {
        'request': [log_request],
        'response': [log_response]
    }
    
    client = OpenAI(
        api_key=api_key,
        http_client=http_client
    )
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.critical(f"Failed to initialize OpenAI client: {str(e)}")
    logger.critical(f"Traceback: {traceback.format_exc()}")
    raise

# Routes
@app.route('/', methods=['GET'])
def fun():
    logger.info("Root endpoint accessed")
    return {'message': 'Image Editor API is running!'}

@app.route('/test-openai', methods=['GET'])
def test_openai():
    """Simple endpoint to test OpenAI API connection without images"""
    logger.info("test-openai endpoint accessed")
    try:
        start_time = time.time()
        logger.info("Making test request to OpenAI API with model: gpt-4o-mini")
        
        # Log network information
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            logger.info(f"Host information - Hostname: {hostname}, IP: {ip_address}")
            
            # Test connection to OpenAI API
            openai_host = "api.openai.com"
            logger.info(f"Testing connection to {openai_host}")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(15)
            result = s.connect_ex((openai_host, 443))
            if result == 0:
                logger.info(f"Connection to {openai_host}:443 successful")
            else:
                logger.warning(f"Connection to {openai_host}:443 failed with error code {result}")
            s.close()
        except Exception as net_err:
            logger.error(f"Error checking network: {str(net_err)}")
        
        # Make a simple text request to OpenAI
        logger.debug("Creating chat completion with gpt-4o-mini")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            max_tokens=100
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        logger.info(f"OpenAI API responded in {response_time:.2f} seconds")
        
        return jsonify({
            "status": "success",
            "response": response.choices[0].message.content,
            "response_time_seconds": response_time,
            "model": "gpt-4o-mini"
        })
    except Exception as e:
        logger.error(f"Unexpected error testing OpenAI API: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e)
        }), 500

@app.route('/edit-image', methods=['POST'])
def edit_image():
    logger.info("edit-image endpoint accessed")
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            logger.warning("No image provided in request")
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        instruction = request.form.get('instruction', '')
        max_width = int(request.form.get('max_width', 1024))
        max_height = int(request.form.get('max_height', 1024))
        
        logger.info(f"Received image: {file.filename}, instruction: {instruction}")
        logger.info(f"Max dimensions: {max_width}x{max_height}")
        
        if file.filename == '':
            logger.warning("Empty filename provided")
            return jsonify({"error": "No image selected"}), 400
        
        # Read and process the uploaded image
        logger.debug("Reading uploaded image")
        image_content = file.read()
        img = Image.open(io.BytesIO(image_content))
        logger.info(f"Original image dimensions: {img.size}, format: {img.format}")
        
        # Resize image if needed (DALL·E has size limitations)
        logger.debug("Resizing image if needed")
        img = resize_image(img, max_width, max_height)
        logger.info(f"Processed image dimensions: {img.size}")
        
        # Step 1: Refine the user instruction
        logger.info("Step 1: Refining instruction")
        refined_instruction = refine_instruction(instruction)
        logger.info(f"Refined instruction: {refined_instruction}")
        
        # Step 2: Generate a new image based on the instruction
        logger.info("Step 2: Generating image based on instruction")
        
        # Try to use the image as a reference for text-to-image generation
        try:
            # Convert image to proper format for DALL-E 2
            img_png = convert_to_png(img)
            
            # Generate image using DALL-E 2
            edited_image_url = generate_image_with_dalle(img_png, refined_instruction)
            logger.info(f"Generated image URL: {edited_image_url[:50]}...")
            
            # Download and return the edited image
            logger.info("Step 3: Downloading image")
            edited_image_data = download_image(edited_image_url)
            logger.info(f"Downloaded image size: {len(edited_image_data)} bytes")
            
            # Create a BytesIO object from the image data
            result_image = io.BytesIO(edited_image_data)
            result_image.seek(0)
            
            logger.info("Returning edited image to client")
            return send_file(
                result_image,
                mimetype='image/png',
                as_attachment=True,
                download_name='edited_image.png'
            )
        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Create a placeholder image with text
            logger.info("Creating placeholder image with error message")
            placeholder_img = create_placeholder_image(instruction, str(e))
            placeholder_buffer = io.BytesIO()
            placeholder_img.save(placeholder_buffer, format="PNG")
            placeholder_buffer.seek(0)
            
            return send_file(
                placeholder_buffer,
                mimetype='image/png',
                as_attachment=True,
                download_name='error_image.png'
            )
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": str(e),
            "error_type": type(e).__name__
        }), 500

def resize_image(img, max_width, max_height):
    """Resize image while maintaining aspect ratio"""
    width, height = img.size
    logger.debug(f"resize_image called with dimensions: {width}x{height}, max: {max_width}x{max_height}")
    
    # Calculate new dimensions while maintaining aspect ratio
    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
        img = img.resize((new_width, new_height), Image.LANCZOS)
    else:
        logger.info("Image is within size limits, no resizing needed")
    
    return img

def convert_to_png(img):
    """Convert image to PNG format with transparency"""
    logger.info("Converting image to PNG format")
    
    # Create a new RGBA image with white background
    if img.mode != 'RGBA':
        # Convert to RGBA if not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Create a new image with alpha channel
        png_img = Image.new('RGBA', img.size, (255, 255, 255, 255))
        png_img.paste(img, (0, 0))
    else:
        png_img = img
    
    # Save to BytesIO
    buffer = io.BytesIO()
    png_img.save(buffer, format="PNG")
    buffer.seek(0)
    
    return buffer

def refine_instruction(original_instruction):
    """Refine the user's instruction"""
    logger.info("Refining instruction")
    try:
        logger.debug(f"Making API call to refine instruction: {original_instruction}")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating precise image editing instructions. Your task is to convert user instructions into detailed, specific prompts that will produce the best results."
                },
                {
                    "role": "user",
                    "content": f"""
                    Original user instruction: "{original_instruction}"
                    
                    Create a detailed, specific instruction for image editing that will achieve what the user wants.
                    The instruction should be precise about what objects to modify, their positions, colors, and any other relevant details.
                    Focus only on the editing task, don't include explanations or notes.
                    """
                }
            ],
            max_tokens=500
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Instruction refinement completed in {elapsed_time:.2f} seconds")
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error refining instruction: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return original_instruction

def generate_image_with_dalle(img_buffer, prompt):
    """Generate image using DALL-E 2"""
    logger.info("Generating image with DALL-E 2")
    try:
        # Try with DALL-E 2 for image generation (not edit)
        logger.info("Attempting to generate image with DALL-E 2")
        start_time = time.time()
        
        response = client.images.edit(
            model="dall-e-2",
            image=img_buffer,
            prompt=f"{prompt}",
            n=1,
            size="1024x1024"
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"DALL·E 2 generation completed in {elapsed_time:.2f} seconds")
        
        return response.data[0].url
    except Exception as e:
        logger.error(f"Error generating image with DALL·E 2: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Use a placeholder image service as fallback
        logger.info("Using placeholder image service as fallback")
        return f"https://placehold.co/1024x1024/png?text={prompt.replace(' ', '+')[:100]}"

def create_placeholder_image(instruction, error_message):
    """Create a placeholder image with text"""
    logger.info("Creating placeholder image")
    
    # Create a new image with a gradient background
    width, height = 800, 600
    image = Image.new('RGB', (width, height), color=(73, 109, 137))
    
    # Draw text on the image
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(image)
    
    # Add instruction text
    draw.text((20, 20), f"Instruction: {instruction[:100]}", fill=(255, 255, 255))
    
    # Add error message
    draw.text((20, 60), f"Error: {error_message[:200]}", fill=(255, 200, 200))
    
    # Add helpful message
    draw.text((20, 120), "Please check your OpenAI API key permissions", fill=(200, 255, 200))
    draw.text((20, 160), "and ensure you have access to image generation", fill=(200, 255, 200))
    
    return image

def download_image(url):
    """Download image from URL"""
    logger.info(f"Downloading image from URL: {url[:50]}...")
    try:
        start_time = time.time()
        
        response = requests.get(url, timeout=60)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Image download completed in {elapsed_time:.2f} seconds")
        
        if response.status_code == 200:
            logger.info(f"Successfully downloaded image, size: {len(response.content)} bytes")
            return response.content
        else:
            logger.error(f"Failed to download image: HTTP {response.status_code}")
            raise Exception(f"Failed to download image: HTTP {response.status_code}")
    except Exception as e:
        logger.error(f"Error downloading image: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create a placeholder image if download fails
        logger.info("Creating placeholder image due to download failure")
        img = Image.new('RGB', (512, 512), color=(73, 109, 137))
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return buffered.getvalue()

@app.route('/test-image-edit', methods=['GET'])
def test_image_edit():
    """Test if the account has image editing capabilities"""
    logger.info("Testing image edit capabilities")
    try:
        # Create a simple test image - pure white PNG with transparency
        test_img = Image.new('RGBA', (512, 512), (255, 255, 255, 0))
        
        # Save to BytesIO with correct format
        buffer = io.BytesIO()
        test_img.save(buffer, format="PNG")
        buffer.seek(0)
        
        # Create a temporary file with .png extension
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(buffer.getvalue())
            temp_path = temp_file.name
        
        logger.info(f"Created temporary PNG file at {temp_path}")
        
        # Open the file in binary mode with correct MIME type
        with open(temp_path, "rb") as png_file:
            # Try to use the images.edit endpoint
            response = client.images.edit(
                image=png_file,  # Pass the file object directly
                prompt="Add a blue circle in the center",
                n=1,
                size="1024x1024"
            )
        
        # Clean up the temporary file
        import os
        os.unlink(temp_path)
        
        return jsonify({
            "status": "success",
            "message": "Your account has image editing capabilities!",
            "url": response.data[0].url
        })
    except Exception as e:
        logger.error(f"Error testing image edit: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            "status": "error",
            "message": "Your account does not have image editing capabilities or there was an error.",
            "error": str(e),
            "error_type": type(e).__name__
        })  

# For local testing (not used in production)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)