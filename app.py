import base64
import io
import os
import time
import traceback
import sys
import socket
import json
from flask import Flask, request, send_file, jsonify
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

# Maximum length for DALL-E 2 prompts
MAX_DALLE_PROMPT_LENGTH = 900

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
        # response = client.chat.completions.create(
        #     model="gpt-4o-mini",
        #     messages=[
        #         {"role": "system", "content": "You are a helpful assistant."},
        #         {"role": "user", "content": "What is the capital of France?"}
        #     ],
        #     max_tokens=100
        # )
        response = client.images.generate(
            # model="dall-e-2",
            prompt="Generate an image of a futuristic city skyline at sunset.",
            n=1,
            size="1024x1024"
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

@app.route('/edit-image-url', methods=['POST'])
def edit_image_url():
    """Edit an image from a URL"""
    logger.info("edit-image-url endpoint accessed")
    try:
        # Get the image URL and instruction from the request
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        image_url = data.get('image_url')
        instruction = data.get('instruction', '')
        max_width = int(data.get('max_width', 1024))
        max_height = int(data.get('max_height', 1024))
        
        if not image_url:
            logger.warning("No image URL provided")
            return jsonify({"error": "No image URL provided"}), 400
        
        logger.info(f"Received image URL: {image_url}, instruction: {instruction}")
        logger.info(f"Max dimensions: {max_width}x{max_height}")
        
        # Download the image from the URL
        logger.info(f"Downloading image from URL: {image_url}")
        try:
            response = requests.get(image_url, timeout=30)
            if response.status_code != 200:
                logger.error(f"Failed to download image: HTTP {response.status_code}")
                return jsonify({"error": f"Failed to download image: HTTP {response.status_code}"}), 400
            
            image_content = response.content
            logger.info(f"Downloaded image, size: {len(image_content)} bytes")
        except Exception as download_err:
            logger.error(f"Error downloading image: {str(download_err)}")
            return jsonify({"error": f"Error downloading image: {str(download_err)}"}), 400
        
        # Process the downloaded image
        try:
            img = Image.open(io.BytesIO(image_content))
            logger.info(f"Original image dimensions: {img.size}, format: {img.format}")
            
            # Resize image if needed (DALL·E has size limitations)
            img = resize_image(img, max_width, max_height)
            logger.info(f"Processed image dimensions: {img.size}")
            
            # Step 1: Analyze the image with GPT-4o-mini
            logger.info("Step 1: Analyzing image with GPT-4o-mini")
            img_features = extract_image_features(img)
            logger.warning(f"Extracted image features: {img_features}")
            image_analysis = analyze_image_with_gpt(img_features)
            logger.warning(f"Image analysis result: {image_analysis}...")
            logger.info(f"Image analysis: {image_analysis[:100]}...")
            
            # Step 2: Refine the user instruction with GPT-4o-mini
            logger.info("Step 2: Refining instruction with GPT-4o-mini")
            refined_instruction = refine_instruction(instruction, image_analysis)
            logger.info(f"Refined instruction: {refined_instruction[:100]}...")
            
            # Step 3: Edit the image based on the instruction using DALL-E 2
            logger.info("Step 3: Editing image with DALL-E 2")
            edited_image_data, success = edit_image_with_dalle(img, refined_instruction)
            
            if not success:
                logger.warning("Image editing failed, returning error")
                return jsonify({
                    "status": "error",
                    "message": "Failed to edit image with DALL-E 2",
                    "refined_instruction": refined_instruction
                }), 400
            
            # Create a response with both the image and the refined instructions
            logger.info("Creating response with image and refined instructions")
            
            # Create a multipart response with JSON and image
            response_data = {
                "status": "success",
                "refined_instruction": refined_instruction,
                "image_analysis": image_analysis
            }
            
            # Return the edited image and the refined instructions
            return send_file_with_json(
                edited_image_data,
                mimetype='image/png',
                as_attachment=True,
                download_name='edited_image.png',
                json_data=response_data
            )
            
        except Exception as process_err:
            logger.error(f"Error processing image: {str(process_err)}")
            logger.error(f"Error type: {type(process_err).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                "status": "error",
                "error": str(process_err),
                "error_type": type(process_err).__name__,
                "refined_instruction": refined_instruction if 'refined_instruction' in locals() else None
            }), 500
    
    except Exception as e:
        logger.error(f"Error in edit-image-url endpoint: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }), 500

def send_file_with_json(file_data, mimetype, as_attachment, download_name, json_data):
    """Send a file with JSON data in the response headers"""
    response = send_file(
        io.BytesIO(file_data),
        mimetype=mimetype,
        as_attachment=as_attachment,
        download_name=download_name
    )
    
    # Add the JSON data as a custom header
    response.headers['X-Response-Data'] = json.dumps(json_data)
    
    return response

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

def extract_image_features(img):
    """Extract basic features from the image to use as reference"""
    width, height = img.size
    aspect_ratio = width / height
    
    # Resize for faster processing
    img_small = img.resize((100, 100))
    if img_small.mode != 'RGB':
        img_small = img_small.convert('RGB')
    
    # Extract dominant colors
    colors = []
    try:
        # Get pixel data
        pixels = list(img_small.getdata())
        # Count colors
        color_count = {}
        for pixel in pixels:
            if pixel in color_count:
                color_count[pixel] += 1
            else:
                color_count[pixel] = 1
        
        # Get top 5 colors
        sorted_colors = sorted(color_count.items(), key=lambda x: x[1], reverse=True)
        for i in range(min(5, len(sorted_colors))):
            r, g, b = sorted_colors[i][0]
            colors.append(f"rgb({r},{g},{b})")
    except Exception as e:
        logger.error(f"Error extracting colors: {str(e)}")
        colors = ["unknown"]
    
    # Detect if image is mostly dark or light
    try:
        brightness_sum = sum(sum(p) for p in pixels)
        avg_brightness = brightness_sum / (len(pixels) * 3)  # 3 channels
        tone = "dark" if avg_brightness < 128 else "light"
    except:
        tone = "unknown"
    
    return {
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "dominant_colors": colors,
        "tone": tone
    }

def analyze_image_with_gpt(img_features):
    """Analyze image features using GPT-4o-mini"""
    logger.info("Analyzing image with GPT-4o-mini")
    try:
        start_time = time.time()
        
        # Create a description of the image
        img_description = (
            f"An image of size {img_features['width']}x{img_features['height']} "
            f"with aspect ratio {img_features['aspect_ratio']:.2f}, "
            f"having a {img_features['tone']} overall tone "
            f"and dominant colors: {', '.join(img_features['dominant_colors'][:3])}"
        )
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert image analyst. Describe what might be in this image based on the technical data provided. Be specific about possible objects, their positions, and the overall scene composition. Keep your analysis under 300 characters."
                },
                {
                    "role": "user",
                    "content": f"Based on this technical data, describe what might be in this image: {img_description}"
                }
            ],
            max_tokens=150  # Limiting token count for concise analysis
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Image analysis completed in {elapsed_time:.2f} seconds")
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Image analysis failed. Using basic features: {img_features['width']}x{img_features['height']}, {img_features['tone']} tone."

def refine_instruction(original_instruction, image_analysis):
    """Refine the user's instruction based on image analysis"""
    logger.info("Refining instruction with image analysis")
    try:
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at creating precise image editing instructions. Create CONCISE instructions (under 900 characters) that focus on the specific changes needed. Be direct and brief while maintaining clarity. Specify exact positions and colors when relevant."
                },
                {
                    "role": "user",
                    "content": f"""
                    Image analysis: {image_analysis}
                    
                    User wants to: "{original_instruction}"
                    
                    Create a concise, specific instruction for DALL-E 2 to edit this image.
                    Your response MUST be under 900 characters total.
                    Focus only on the essential editing steps.
                    For color changes, specify the exact colors (e.g., 'change red car to forest green').
                    For object modifications, be precise about what to modify and how.
                    """
                }
            ],
            max_tokens=300  # Limiting token count to ensure shorter response
        )
        
        refined_instruction = response.choices[0].message.content
        
        # Check if the refined instruction is too long and truncate if necessary
        if len(refined_instruction) > MAX_DALLE_PROMPT_LENGTH:
            logger.warning(f"Refined instruction too long ({len(refined_instruction)} chars), truncating to {MAX_DALLE_PROMPT_LENGTH} chars")
            refined_instruction = refined_instruction[:MAX_DALLE_PROMPT_LENGTH]
        
        elapsed_time = time.time() - start_time
        logger.info(f"Instruction refinement completed in {elapsed_time:.2f} seconds")
        
        return refined_instruction
    except Exception as e:
        logger.error(f"Error refining instruction: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Ensure the original instruction is also truncated if needed
        if len(original_instruction) > MAX_DALLE_PROMPT_LENGTH:
            return original_instruction[:MAX_DALLE_PROMPT_LENGTH]
        return original_instruction

def edit_image_with_dalle(img, prompt):
    """Generate a new image based on the original image and prompt"""
    logger.info("Generating image with DALL-E 2")
    try:
        # Ensure prompt is within valid length
        if len(prompt) > MAX_DALLE_PROMPT_LENGTH:
            logger.warning(f"Prompt too long ({len(prompt)} chars), truncating to {MAX_DALLE_PROMPT_LENGTH} chars")
            prompt = prompt[:MAX_DALLE_PROMPT_LENGTH]
        
        # Extract image features to enhance the prompt
        img_features = extract_image_features(img)
        width, height = img.size
        
        # Create an enhanced prompt that describes both the original image and the desired changes
        enhanced_prompt = f"Create a professional product photo of {prompt}. The image should be {width}x{height} with a clean background."
        
        logger.info(f"Enhanced prompt: {enhanced_prompt[:100]}...")
        
        # Generate a new image using DALL-E 2
        logger.info("Calling OpenAI images.generate API")
        start_time = time.time()
        
        response = client.images.generate(
            model="dall-e-2",
            prompt=enhanced_prompt,
            n=1,
            size="1024x1024"
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"DALL·E 2 generation completed in {elapsed_time:.2f} seconds")
        
        # Get the URL of the generated image
        generated_image_url = response.data[0].url
        logger.info(f"Generated image URL: {generated_image_url[:50]}...")
        
        # Download the generated image
        logger.info("Downloading generated image")
        generated_image_response = requests.get(generated_image_url, timeout=30)
        if generated_image_response.status_code != 200:
            logger.error(f"Failed to download image: HTTP {generated_image_response.status_code}")
            raise Exception(f"Failed to download image: HTTP {generated_image_response.status_code}")
        
        generated_image_data = generated_image_response.content
        logger.info(f"Downloaded image, size: {len(generated_image_data)} bytes")
        
        return generated_image_data, True
    except Exception as e:
        logger.error(f"Error with DALL·E 2: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # No fallback, just return the error
        return None, False

# For local testing (not used in production)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
