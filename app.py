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

from flask_cors import CORS

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

CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173"}})
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

@app.route('/test-openai', methods=['POST'])
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
        
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        prompt = data.get('prompt')
        # Make a simple text request to OpenAI
        logger.debug("Creating chat completion with gpt-4o-mini")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
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

@app.route('/generate-image-gemini', methods=['POST'])
def generate_image_gemini():
    """Generate an image using Google's Gemini model"""
    logger.info("generate-image-gemini endpoint accessed")
    try:
        # Get the prompt from the request
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        prompt = data.get('prompt')
        if not prompt:
            logger.warning("No prompt provided")
            return jsonify({"error": "No prompt provided"}), 400
        
        logger.info(f"Received prompt: {prompt}")
        
        # Initialize Gemini
        try:
            import google.generativeai as genai
            from PIL import Image
            from io import BytesIO
            import base64
            
            # Get API key from environment variable or request
            # gemini_api_key = data.get('api_key') or os.environ.get("GEMINI_API_KEY")
            # if not gemini_api_key:
            #     logger.critical("GEMINI_API_KEY not provided")
            #     return jsonify({"error": "GEMINI_API_KEY is required. Please provide it in the request or set the environment variable."}), 400
            
            # # Configure the Gemini API
            # genai.configure(api_key=gemini_api_key)
            # logger.info("Gemini API configured successfully")
            
            # Generate image with Gemini
            start_time = time.time()
            logger.info(f"Generating image with Gemini model: gemini-2.0-flash-preview-image-generation")
            
            # Get the model
            model = genai.GenerativeModel('gemini-2.0-flash-preview-image-generation')
            
            # Generate content
            response = model.generate_content(
                contents=prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_types=['text/plain', 'image/png']
                )
            )
            
            # Process response
            text_content = ""
            image_data = None
            
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'text') and part.text:
                    text_content += part.text
                    logger.info(f"Received text response: {text_content[:100]}...")
                elif hasattr(part, 'inline_data') and part.inline_data:
                    # Convert binary data to image
                    image_data = base64.b64decode(part.inline_data.data)
                    logger.info(f"Received image data, size: {len(image_data)} bytes")
            
            # Calculate response time
            response_time = time.time() - start_time
            logger.info(f"Gemini API responded in {response_time:.2f} seconds")
            
            # Prepare response
            if image_data:
                # Create a response with both the image and text
                response_data = {
                    "status": "success",
                    "text": text_content,
                    "response_time_seconds": response_time,
                    "model": "gemini-2.0-flash-preview-image-generation"
                }
                
                # Return the generated image and text
                return send_file_with_json(
                    image_data,
                    mimetype='image/png',
                    as_attachment=True,
                    download_name='gemini_generated_image.png',
                    json_data=response_data
                )
            else:
                # If no image was generated, return just the text
                return jsonify({
                    "status": "success",
                    "text": text_content,
                    "response_time_seconds": response_time,
                    "model": "gemini-2.0-flash-preview-image-generation"
                })
                
        except ImportError as import_err:
            logger.error(f"Missing required packages for Gemini: {str(import_err)}")
            return jsonify({
                "status": "error",
                "error": f"Missing required packages: {str(import_err)}",
                "error_type": "ImportError"
            }), 500
        except Exception as gemini_err:
            logger.error(f"Error with Gemini API: {str(gemini_err)}")
            logger.error(f"Error type: {type(gemini_err).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                "status": "error",
                "error": str(gemini_err),
                "error_type": type(gemini_err).__name__
            }), 500
    
    except Exception as e:
        logger.error(f"Error in generate-image-gemini endpoint: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }), 500
    



@app.route('/analyze-image-file', methods=['POST'])
def analyze_image_file():
    """Analyze an uploaded image file directly using GPT-4o-mini"""
    logger.info("analyze-image-file endpoint accessed")
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            logger.warning("No image file in request")
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        
        # If user does not select file, browser might submit an empty file
        if file.filename == '':
            logger.warning("Empty filename submitted")
            return jsonify({"error": "No image selected"}), 400
        
        # Check if the file is allowed
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}
        if not '.' in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            logger.warning(f"File type not allowed: {file.filename}")
            return jsonify({"error": "File type not allowed. Please upload an image (png, jpg, jpeg, gif, webp, bmp)"}), 400
        
        logger.info(f"Processing uploaded image: {file.filename}")
        
        # Process the image
        try:
            start_time = time.time()
            
            # Read the image
            img_data = file.read()
            img = Image.open(io.BytesIO(img_data))
            logger.info(f"Image opened successfully. Format: {img.format}, Size: {img.size}")
            
            # For reference, still extract basic features
            img_features = extract_image_features(img)
            
            # Convert image to base64 for sending to OpenAI API
            buffered = io.BytesIO()
            img.save(buffered, format=img.format if img.format else "JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            # Analyze the image directly with GPT-4o-mini
            analysis = analyze_image_directly(img_base64, img.format if img.format else "JPEG")
            logger.info(f"Direct image analysis completed")
            
            # Get detailed analysis with specific questions
            detailed_analysis = get_detailed_direct_analysis(img_base64, img.format if img.format else "JPEG")
            logger.info(f"Detailed direct analysis completed")
            
            # Calculate response time
            response_time = time.time() - start_time
            logger.info(f"Analysis completed in {response_time:.2f} seconds")
            
            # Return the analysis
            return jsonify({
                "status": "success",
                "filename": file.filename,
                "image_features": img_features,  # Still include basic features for reference
                "analysis": analysis,
                "detailed_analysis": detailed_analysis,
                "response_time_seconds": response_time
            })
            
        except Exception as process_err:
            logger.error(f"Error processing image: {str(process_err)}")
            logger.error(f"Error type: {type(process_err).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({
                "status": "error",
                "error": str(process_err),
                "error_type": type(process_err).__name__
            }), 500
    
    except Exception as e:
        logger.error(f"Error in analyze-image-file endpoint: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }), 500

def analyze_image_directly(img_base64, img_format):
    """Analyze the image directly using GPT-4o-mini's vision capabilities"""
    logger.info("Analyzing image directly with GPT-4o-mini")
    try:
        # Create the message with the image
        messages = [
            {
                "role": "system",
                "content": "You are an expert image analyst. Describe what you see in this image in detail, including objects, people, scenes, colors, and any notable elements."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this image and describe what you see in detail."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{img_format.lower()};base64,{img_base64}"
                        }
                    }
                ]
            }
        ]
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error analyzing image directly: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Failed to analyze image directly: {str(e)}"

def get_detailed_direct_analysis(img_base64, img_format):
    """Get a more detailed analysis of the image by asking specific questions"""
    logger.info("Getting detailed direct image analysis with GPT-4o-mini")
    try:
        # Create the message with the image and specific questions
        messages = [
            {
                "role": "system",
                "content": "You are an expert image analyst with deep knowledge in visual arts, photography, and cultural context. Provide a comprehensive analysis of the image."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyze this image in detail and answer the following questions:\n\n1. What are the main subjects or objects in the image?\n2. What is the overall mood or atmosphere of the image?\n3. What techniques or style is used in this image?\n4. What might be the context or story behind this image?\n5. Are there any notable details that might be easily missed?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/{img_format.lower()};base64,{img_base64}"
                        }
                    }
                ]
            }
        ]
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error getting detailed direct image analysis: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return f"Failed to get detailed direct image analysis: {str(e)}"




@app.route('/generate-image', methods=['GET'])
def generate_image():
    """Endpoint to generate an image using DALL-E"""
    logger.info("generate-image endpoint accessed")
    
    try:
        # Get the prompt from the request
        # data = request.get_json()
        # prompt = data.get('prompt')
        
        # if not prompt:
        #     return jsonify({
        #         "status": "error",
        #         "message": "No prompt provided"
        #     }), 400
        
        # logger.info(f"Generating image with prompt: {prompt}")
        start_time = time.time()
        
        # Generate the image using DALL-E
        response = client.images.generate(
            model="dall-e-2",
            prompt='blue circle.',
            size="1024x1024",
            # quality="standard",
            n=1,
            response_format="url"
        )
        
        image_url = response.data[0].url
        elapsed_time = time.time() - start_time
        
        logger.info(f"Image generated successfully. Time elapsed: {elapsed_time:.2f} seconds")
        
        return jsonify({
            "status": "success",
            "image_url": image_url,
            "time_elapsed": f"{elapsed_time:.2f} seconds"
        })
    
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error generating image: {str(e)}"
        }), 500


@app.route('/create-variation', methods=['POST'])
def create_variation():
    """Endpoint to create variations of an existing image"""
    logger.info("create-variation endpoint accessed")
    
    try:
        # Check if an image file was uploaded
        if 'image' not in request.files:
            return jsonify({
                "status": "error",
                "message": "No image file provided"
            }), 400
            
        image_file = request.files['image']
        
        # Save the uploaded file temporarily
        temp_path = "temp_upload.png"
        image_file.save(temp_path)
        
        # Get number of variations (default to 1)
        num_variations = 1
        
        # Create variations
        with open(temp_path, "rb") as image_file:
            response = client.images.create_variation(
                image=image_file,
                n=num_variations,
                size="1024x1024"
            )
            
        # Extract URLs from the response
        variation_urls = [item.url for item in response.data]
        
        # Clean up the temporary file
        os.remove(temp_path)
        
        return jsonify({
            "status": "success",
            "variation_urls": variation_urls
        })
    
    except Exception as e:
        logger.error(f"Error creating image variations: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error creating image variations: {str(e)}"
        }), 500





# hg


HF_API_KEY = os.environ.get("HF_API_KEY")

@app.route('/generate-image-hf', methods=['POST'])
def generate_image_hf():
    """Endpoint to generate an image using Hugging Face Stable Diffusion"""
    logger.info("generate-image endpoint accessed")
    
    try:
        start_time = time.time()
        
        # Get the prompt from the request
        data = request.get_json()
        prompt = data.get('prompt')
        
        if not prompt:
            logger.warning("No prompt provided")
            return jsonify({
                "status": "error",
                "message": "No prompt provided"
            }), 400
        
        logger.info(f"Generating image with prompt: {prompt}")
        
        # Using Stable Diffusion XL from Hugging Face
        API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        
        # Make the request to Hugging Face
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt}
        )
        
        # Check for errors
        if response.status_code != 200:
            error_msg = f"Error from Hugging Face API: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return jsonify({
                "status": "error",
                "message": error_msg
            }), 500
        
        # Convert the image to base64 for easy display in browser
        image_bytes = response.content
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Calculate response time
        response_time = time.time() - start_time
        logger.info(f"Image generated successfully in {response_time:.2f} seconds")
        
        return jsonify({
            "status": "success",
            "image_base64": image_base64,
            "response_time_seconds": response_time
        })
    
    except Exception as e:
        logger.error(f"Unexpected error generating image: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e)
        }), 500

@app.route('/edit-image-hf', methods=['POST'])
def edit_image_hf():
    """Endpoint to edit an image based on both the image and a text prompt"""
    logger.info("edit-image endpoint accessed")
    
    try:
        start_time = time.time()
        
        # Check if an image file was uploaded
        if 'image' not in request.files:
            logger.warning("No image file provided")
            return jsonify({
                "status": "error",
                "message": "No image file provided"
            }), 400
            
        image_file = request.files['image']
        prompt = request.form.get('prompt', '')
        
        if not prompt:
            logger.warning("No prompt provided")
            return jsonify({
                "status": "error",
                "message": "No prompt provided"
            }), 400
        
        logger.info(f"Editing image with prompt: {prompt}")
        
        # Read the image file
        image_bytes = image_file.read()
        
        # Convert to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        # Using InstructPix2Pix model from Hugging Face
        API_URL = "https://api-inference.huggingface.co/models/timbrooks/instruct-pix2pix"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        
        # Make the request to Hugging Face
        response = requests.post(
            API_URL,
            headers=headers,
            json={
                "inputs": {
                    "image": image_base64,
                    "prompt": prompt
                }
            }
        )
        
        # Check for errors
        if response.status_code != 200:
            error_msg = f"Error from Hugging Face API: {response.status_code} - {response.text}"
            logger.error(error_msg)
            return jsonify({
                "status": "error",
                "message": error_msg
            }), 500
        
        # Convert the result image to base64
        result_image_bytes = response.content
        result_image_base64 = base64.b64encode(result_image_bytes).decode('utf-8')
        
        # Calculate response time
        response_time = time.time() - start_time
        logger.info(f"Image edited successfully in {response_time:.2f} seconds")
        
        return jsonify({
            "status": "success",
            "image_base64": result_image_base64,
            "response_time_seconds": response_time
        })
    
    except Exception as e:
        logger.error(f"Unexpected error editing image: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e)
        }), 500

@app.route('/test-huggingface', methods=['POST'])
def test_huggingface():
    """Endpoint to test Hugging Face API connection with a simple prompt"""
    logger.info("test-huggingface endpoint accessed")
    try:
        start_time = time.time()
        
        # Get the prompt from the request
        data = request.get_json()
        prompt = data.get('prompt', 'a simple blue circle')  # Default prompt if none provided
        
        logger.info(f"Making test request to Hugging Face API with prompt: {prompt}")
        
        # Log network information
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            logger.info(f"Host information - Hostname: {hostname}, IP: {ip_address}")
            
            # Test connection to Hugging Face API
            hf_host = "api-inference.huggingface.co"
            logger.info(f"Testing connection to {hf_host}")
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(15)
            result = s.connect_ex((hf_host, 443))
            if result == 0:
                logger.info(f"Connection to {hf_host}:443 successful")
            else:
                logger.warning(f"Connection to {hf_host}:443 failed with error code {result}")
            s.close()
        except Exception as net_err:
            logger.error(f"Error checking network: {str(net_err)}")
        
        # Make a simple request to Hugging Face
        logger.debug("Creating test image with Stable Diffusion")
        
        # Using a smaller/faster model for testing
        API_URL = "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-v1-5"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        
        # Make the request
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt}
        )
        
        # Check response status
        if response.status_code != 200:
            logger.error(f"Hugging Face API returned status code {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return jsonify({
                "status": "error",
                "message": f"Hugging Face API returned status code {response.status_code}",
                "details": response.text
            }), 500
        
        # Convert the image to base64
        image_base64 = base64.b64encode(response.content).decode('utf-8')
        
        # Calculate response time
        response_time = time.time() - start_time
        logger.info(f"Hugging Face API responded in {response_time:.2f} seconds")
        
        return jsonify({
            "status": "success",
            "image_base64": image_base64,
            "response_time_seconds": response_time,
            "model": "runwayml/stable-diffusion-v1-5"
        })
    except Exception as e:
        logger.error(f"Unexpected error testing Hugging Face API: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e)
        }), 500


################
# הפרומפט המערכתי המדויק
SYSTEM_PROMPT = """
You are an AI system that generates JSON objects for image editing based on natural language requests.

The structure of the JSON object you must return is:
{
  "filters": {
    // Various editing options based on the request
  }
}

Available editing options:

1. textLayer - for adding text:
   {
     "text": "The text to display",
     "x": X position (number, center of canvas is 250),
     "y": Y position (number, center of canvas is 250),
     "fontSize": font size in pixels (number),
     "color": text color (HEX code like "#ff0000"),
     "fontFamily": "font name" (optional, default is "Arial"),
     "fontWeight": "font weight" (optional, default is "normal"),
     "textAlign": "text alignment" (optional, values: "left", "center", "right")
   }

2. border - for adding a border:
   {
     "width": border width in pixels (number),
     "color": border color (HEX code),
     "style": border style ("solid", "dashed", "dotted")
   }

3. filter - for general image filters:
   {
     "brightness": brightness (0-200, where 100 is normal),
     "contrast": contrast (0-200, where 100 is normal),
     "saturation": saturation (0-200, where 100 is normal),
     "blur": blur in pixels (0-20),
     "grayscale": black and white (0-100 percent)
   }

4. overlay - for adding a color layer:
   {
     "color": overlay color (HEX code with alpha, like "#ff000080"),
     "blendMode": blend mode ("multiply", "screen", "overlay", "darken", "lighten")
   }

5. transform - for geometric changes:
   {
     "rotation": rotation in degrees (0-360),
     "scale": scale factor (number, 1 is normal size),
     "flipX": horizontal flip (boolean),
     "flipY": vertical flip (boolean)
   }

6. crop - for cropping the image:
   {
     "x": starting X point (number),
     "y": starting Y point (number),
     "width": width (number),
     "height": height (number)
   }

7. shadow - for adding shadow:
   {
     "offsetX": horizontal offset (number),
     "offsetY": vertical offset (number),
     "blur": blur amount (number),
     "color": shadow color (HEX code with alpha, like "#00000080")
   }

Important instructions:
1. Return ONLY the JSON object without any explanations or additional text
2. Use only the editing options that are relevant to the request
3. For colors, always use HEX codes (e.g., "#ff0000" for red)
4. For positions, use the provided canvas dimensions (available in the currentImageParams)
5. If the request is ambiguous, make reasonable assumptions based on design best practices
6. If multiple options are requested, include all of them in the response

Important instruction for property updates:
When the user requests to change only specific properties of an existing element (like text size, color, or position), return ONLY the properties that need to change, not the entire object.

CURRENT STATE CONTEXT:
You will be provided with the current state of the image in the request. Use this information to make appropriate modifications.
For example, if the current text color is "#ff0000" (red) and the user asks to "make the text darker", you should return a darker shade of red.
Similarly, if the current font size is 14px and the user asks to "make the text a bit larger", you should return a larger font size like 18px.

Examples:

1. Request: "Make the text larger" (Current fontSize: 14)
   Response:
   {
     "filters": {
       "textLayer": {
         "fontSize": 18
       }
     }
   }

2. Request: "Make the color darker" (Current color: "#ffb6c1" - light pink)
   Response:
   {
     "filters": {
       "textLayer": {
         "color": "#ff8da1"
       }
     }
   }

3. Request: "Add a thin pink border and the text 'Hello' in a matching color"
   Response:
   {
     "filters": {
       "border": {
         "width": 2,
         "color": "#ffb6c1",
         "style": "solid"
       },
       "textLayer": {
         "text": "Hello",
         "x": 250,
         "y": 250,
         "fontSize": 30,
         "color": "#ffb6c1"
       }
     }
   }
"""

@app.route('/image-design', methods=['POST'])
def process_image_design():
    """
    An endpoint for processing natural language image formatting requests.
    Accepts a natural language request, the current state of the image, and additional parameters.
    Returns a JSON object with the required changes.
    """
    logger.info("image-design endpoint accessed")
    start_time = time.time()
    
    try:
        # קבלת הבקשה מהלקוח
        request_data = request.json
        if not request_data or 'prompt' not in request_data:
            logger.error("Missing 'prompt' in request data")
            return jsonify({
                "status": "error",
                "error": "Missing 'prompt' in request data"
            }), 400
        
        user_prompt = request_data['prompt']
        logger.info(f"Received design prompt: {user_prompt}")
        
        # קבלת המצב הנוכחי של התמונה (אם קיים)
        current_state = request_data.get('currentState', {"filters": {}})
        logger.info(f"Current state: {current_state}")
        
        # קבלת פרמטרים נוספים (אם קיימים)
        image_params = request_data.get('imageParams', {
            "width": 500,
            "height": 500,
            "format": "jpg"
        })
        logger.info(f"Image parameters: {image_params}")
        
        # בניית הפרומפט המלא עם המצב הנוכחי והפרמטרים
        full_prompt = f"""
User request: {user_prompt}

Current image state:
{json.dumps(current_state, indent=2)}

Image parameters:
{json.dumps(image_params, indent=2)}

Please provide the JSON object with the necessary changes based on the request and current state.
"""
        # שליחת הבקשה ל-OpenAI
        logger.info(f"Sending request to OpenAI API with model: gpt-4o-mini")
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # או כל מודל אחר שברצונך להשתמש
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.2,  # ערך נמוך לתוצאות יותר עקביות
            max_tokens=500
        )
        
        # חילוץ התשובה מהמודל
        content = response.choices[0].message.content
        logger.debug(f"Raw response from OpenAI: {content}")
        
        # ניסיון לפרסר את התשובה כ-JSON
        try:
            # ניקוי התשובה מתווים מיותרים
            content = content.strip()
            # הסרת תגי קוד אם קיימים
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            # פרסור ה-JSON
            filters_obj = json.loads(content)
            
            # וידוא שהאובייקט מכיל את המפתח 'filters'
            if "filters" not in filters_obj:
                if any(key in filters_obj for key in ["textLayer", "border", "filter", "overlay", "transform", "crop", "shadow"]):
                    # אם האובייקט מכיל ישירות את אפשרויות העריכה, עטוף אותו ב-filters
                    filters_obj = {"filters": filters_obj}
                else:
                    logger.warning(f"Response does not contain 'filters' key: {filters_obj}")
                    filters_obj = {"filters": {}}
            
            # # חישוב זמן התגובה
            # response_time = time.time() - start_time
            # logger.info(f"Request processed in {response_time:.2f} seconds")
            
            return jsonify({
                "status": "success",
                "data": filters_obj,
            })
            
        except json.JSONDecodeError as json_err:
            logger.error(f"JSON parsing error: {str(json_err)}")
            logger.error(f"Raw content that failed to parse: {content}")
            
            # ניסיון לחלץ את חלק ה-JSON מהתשובה
            import re
            json_match = re.search(r'({.*})', content, re.DOTALL)
            if json_match:
                try:
                    extracted_json = json_match.group(1)
                    logger.info(f"Attempting to parse extracted JSON: {extracted_json}")
                    filters_obj = json.loads(extracted_json)
                    
                    # וידוא שהאובייקט מכיל את המפתח 'filters'
                    if "filters" not in filters_obj:
                        filters_obj = {"filters": filters_obj}
                   
                    return jsonify({
                        "status": "success",
                        "data": filters_obj,
                        "note": "JSON was extracted from text response"
                    })
                    
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse extracted JSON: {extracted_json}")
                    return jsonify({
                        "status": "error",
                        "error": "Could not parse AI response as JSON",
                        "raw_response": content
                    }), 500
            else:
                logger.error("No JSON-like structure found in response")
                return jsonify({
                    "status": "error",
                    "error": "AI response did not contain valid JSON",
                    "raw_response": content
                }), 500
    
    except Exception as e:
        logger.error(f"Unexpected error processing request: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e)
        }), 500
################


# For local testing (not used in production)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
