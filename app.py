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
    level=logging.INFO,  # Changed to INFO for production
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)  # This variable name 'app' is important for gunicorn app:app

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
        timeout=120.0,  # Increased to 120 seconds for production
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        http2=True  # Enable HTTP/2 for better performance
    )
    logger.info(f"HTTPX client initialized with timeout: 120.0s, HTTP/2: enabled")
    
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
            s.settimeout(15)  # Increased to 15 seconds
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
        
        # Convert image to base64 for API calls
        logger.debug("Converting image to base64")
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        logger.debug(f"Base64 image size: {len(img_base64)} characters")
        
        # Step 1: Analyze the image using GPT-4o Vision
        logger.info("Step 1: Analyzing image with GPT-4o Vision")
        analysis = analyze_image(img_base64, instruction)
        logger.info(f"Image analysis completed: {len(analysis)} characters")
        
        # Step 2: Refine the user instruction based on the analysis
        logger.info("Step 2: Refining instruction based on analysis")
        refined_instruction = refine_instruction(instruction, analysis)
        logger.info(f"Refined instruction: {refined_instruction}")
        
        # Step 3: Generate the edited image using DALL·E 3
        logger.info("Step 3: Generating edited image with DALL·E 3")
        edited_image_url = generate_edited_image(img_base64, refined_instruction)
        logger.info(f"Edited image URL received: {edited_image_url[:50]}...")
        
        # Step 4: Download and return the edited image
        logger.info("Step 4: Downloading edited image")
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

def analyze_image(img_base64, instruction):
    """Analyze the image using GPT-4o Vision to identify objects and context"""
    logger.info("Analyzing image with GPT-4o Vision")
    try:
        logger.debug(f"Making API call to analyze image with instruction: {instruction}")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert image analyzer. Identify all important objects, people, colors, and context in the image. Focus on elements that might be relevant to the user's editing instruction."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze this image in detail. The user wants to: {instruction}. Identify all relevant objects, their positions, colors, and any other details that would help with precise image editing."},
                        {"type": "image", "image": f"data:image/png;base64,{img_base64}"}
                    ]
                }
            ],
            max_tokens=500
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Image analysis completed in {elapsed_time:.2f} seconds")
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return "Could not analyze the image."

def refine_instruction(original_instruction, image_analysis):
    """Refine the user's instruction based on image analysis"""
    logger.info("Refining instruction based on image analysis")
    try:
        logger.debug(f"Making API call to refine instruction: {original_instruction}")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating precise image editing instructions for DALL·E 3. Your task is to convert user instructions into detailed, specific prompts that will produce the best results."
                },
                {
                    "role": "user",
                    "content": f"""
                    Original user instruction: "{original_instruction}"
                    
                    Image analysis: {image_analysis}
                    
                    Create a detailed, specific instruction for DALL·E 3 that will achieve what the user wants.
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

def generate_edited_image(img_base64, refined_instruction):
    """Generate edited image using DALL·E 3"""
    logger.info("Generating edited image with DALL·E 3")
    try:
        # Convert base64 string to bytes
        logger.debug("Converting base64 to bytes")
        image_data = base64.b64decode(img_base64)
        
        # Create a BytesIO object
        image_bytes_io = io.BytesIO(image_data)
        
        logger.info("Attempting to edit image with DALL·E 3")
        logger.debug(f"Using instruction: {refined_instruction}")
        start_time = time.time()
        
        response = client.images.edit(
            model="dall-e-2",
            # Pass the BytesIO object instead of a string
            image=image_bytes_io,
            prompt=refined_instruction,
            n=1,
            size="1024x1024"
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"DALL·E 3 edit completed in {elapsed_time:.2f} seconds")
        
        return response.data[0].url
    except Exception as e:
        logger.error(f"Error generating edited image with DALL·E 3: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        try:
            logger.info("Attempting fallback to image generation")
            start_time = time.time()
            
            response = client.images.generate(
                model="dall-e-2",
                prompt=f"Edit this image according to these instructions: {refined_instruction}. Maintain the original style and composition as much as possible.",
                n=1,
                size="1024x1024"
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"Fallback generation completed in {elapsed_time:.2f} seconds")
            
            return response.data[0].url
        except Exception as e2:
            logger.error(f"Error with fallback generation: {str(e2)}")
            logger.error(f"Error type: {type(e2).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise Exception(f"Failed to generate edited image: {str(e)} and fallback also failed: {str(e2)}")

def download_image(url):
    """Download image from URL"""
    logger.info(f"Downloading image from URL: {url[:50]}...")
    try:
        start_time = time.time()
        
        response = requests.get(url, timeout=60)  # Increased timeout to 60 seconds
        
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
        raise Exception(f"Failed to download image: {str(e)}")

# For local testing (not used in production)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)