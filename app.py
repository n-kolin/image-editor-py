import base64
import io
import os
import time
import httpx
from dotenv import load_dotenv
from flask import Flask, request, send_file, jsonify
from PIL import Image
import requests
from openai import OpenAI
import logging
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
load_dotenv()

# Initialize OpenAI client with custom HTTP client and increased timeout
http_client = httpx.Client(timeout=60.0)  # 60 seconds timeout
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    http_client=http_client
)

@app.route('/hello', methods=['GET'])
def fun():
    return {'hello': 'world'}

@app.route('/test-openai', methods=['GET'])
def test_openai():
    """Simple endpoint to test OpenAI API connection without images"""
    try:
        start_time = time.time()
        
        # Make a simple text request to OpenAI
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ],
            max_tokens=100
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        
        return jsonify({
            "status": "success",
            "response": response.choices[0].message.content,
            "response_time_seconds": response_time,
            "model": "gpt-3.5-turbo"
        })
    except Exception as e:
        logger.error(f"Error testing OpenAI API: {str(e)}")
        return jsonify({
            "status": "error",
            "error": str(e)
        }), 500

@app.route('/edit-image', methods=['POST'])
def edit_image():
    try:
        # Check if the post request has the file part
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        instruction = request.form.get('instruction', '')
        max_width = int(request.form.get('max_width', 1024))
        max_height = int(request.form.get('max_height', 1024))
        
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Read and process the uploaded image
        image_content = file.read()
        img = Image.open(io.BytesIO(image_content))
        
        # Resize image if needed (DALL·E has size limitations)
        img = resize_image(img, max_width, max_height)
        
        # Convert image to base64 for API calls
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # Step 1: Analyze the image using GPT-4o Vision
        analysis = analyze_image(img_base64, instruction)
        logger.info(f"Image analysis: {analysis}")
        
        # Step 2: Refine the user instruction based on the analysis
        refined_instruction = refine_instruction(instruction, analysis)
        logger.info(f"Refined instruction: {refined_instruction}")
        
        # Step 3: Generate the edited image using DALL·E 3
        edited_image_url = generate_edited_image(img_base64, refined_instruction)
        
        # Step 4: Download and return the edited image
        edited_image_data = download_image(edited_image_url)
        
        # Create a BytesIO object from the image data
        result_image = io.BytesIO(edited_image_data)
        result_image.seek(0)
        
        return send_file(
            result_image,
            mimetype='image/png',
            as_attachment=True,
            download_name='edited_image.png'
        )
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500

def resize_image(img, max_width, max_height):
    """Resize image while maintaining aspect ratio"""
    width, height = img.size
    
    # Calculate new dimensions while maintaining aspect ratio
    if width > max_width or height > max_height:
        ratio = min(max_width / width, max_height / height)
        new_width = int(width * ratio)
        new_height = int(height * ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
    
    return img

def analyze_image(img_base64, instruction):
    """Analyze the image using GPT-4o Vision to identify objects and context"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
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
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return "Could not analyze the image."

def refine_instruction(original_instruction, image_analysis):
    """Refine the user's instruction based on image analysis"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
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
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Error refining instruction: {str(e)}")
        return original_instruction

def generate_edited_image(img_base64, refined_instruction):
    """Generate edited image using DALL·E 3"""
    try:
        # Convert base64 string to bytes
        image_data = base64.b64decode(img_base64)
        
        # Create a BytesIO object
        image_bytes_io = io.BytesIO(image_data)
        
        logger.info("Attempting to edit image with DALL·E 3")
        response = client.images.edit(
            model="dall-e-3",
            # Pass the BytesIO object instead of a string
            image=image_bytes_io,
            prompt=refined_instruction,
            n=1,
            size="1024x1024"
        )
        logger.info("Successfully edited image with DALL·E 3")
        return response.data[0].url
    except Exception as e:
        logger.error(f"Error generating edited image with DALL·E 3: {str(e)}")
        try:
            logger.info("Attempting fallback to image generation")
            response = client.images.generate(
                model="dall-e-3",
                prompt=f"Edit this image according to these instructions: {refined_instruction}. Maintain the original style and composition as much as possible.",
                n=1,
                size="1024x1024"
            )
            logger.info("Successfully generated image with fallback method")
            return response.data[0].url
        except Exception as e2:
            logger.error(f"Error with fallback generation: {str(e2)}")
            raise Exception("Failed to generate edited image")

def download_image(url):
    """Download image from URL"""
    response = requests.get(url)
    if response.status_code == 200:
        return response.content
    else:
        raise Exception(f"Failed to download image: {response.status_code}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=True)