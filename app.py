import os
import requests
import base64
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import logging
from io import BytesIO

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is not set")

headers = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint that doesn't use OpenAI."""
    return jsonify({"status": "healthy", "message": "Server is running"}), 200

@app.route('/openai-text', methods=['POST'])
def openai_text_only():
    """Test endpoint that uses OpenAI but doesn't process images."""
    try:
        data = request.json
        if not data or 'prompt' not in data:
            return jsonify({"error": "Missing prompt in request"}), 400
        
        prompt = data['prompt']
        
        # Call OpenAI API for text completion
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500
            }
        )
        
        if response.status_code != 200:
            logger.error(f"OpenAI API error: {response.text}")
            return jsonify({"error": "OpenAI API error", "details": response.text}), 500
        
        result = response.json()
        return jsonify({"response": result["choices"][0]["message"]["content"]}), 200
    
    except Exception as e:
        logger.error(f"Error in openai-text endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/process-image', methods=['POST'])
def process_image():
    """
    Main endpoint that processes an image based on modification instructions.
    
    Expected JSON payload:
    {
        "image_url": "https://aws-bucket-url/image.jpg",
        "instructions": "Change the red car to green"
    }
    """
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Missing request data"}), 400
        
        image_url = data.get('image_url')
        instructions = data.get('instructions')
        
        if not image_url or not instructions:
            return jsonify({"error": "Missing image_url or instructions"}), 400
        
        # Step 1: Download the image from AWS
        try:
            image_response = requests.get(image_url)
            image_response.raise_for_status()
            image_data = image_response.content
            
            # Convert image to base64
            base64_image = base64.b64encode(image_data).decode('utf-8')
            
            logger.info(f"Successfully downloaded image from {image_url}")
        except Exception as e:
            logger.error(f"Error downloading image: {str(e)}")
            return jsonify({"error": f"Error downloading image: {str(e)}"}), 500
        
        # Step 2: Analyze the image with GPT-4o
        try:
            analysis_response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "Analyze this image in detail. Describe the main objects, colors, and composition."
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{base64_image}"
                                    }
                                }
                            ]
                        }
                    ],
                    "max_tokens": 500
                }
            )
            
            analysis_response.raise_for_status()
            analysis_result = analysis_response.json()
            image_analysis = analysis_result["choices"][0]["message"]["content"]
            
            logger.info("Successfully analyzed image with GPT-4o")
        except Exception as e:
            logger.error(f"Error analyzing image with GPT-4o: {str(e)}")
            return jsonify({"error": f"Error analyzing image with GPT-4o: {str(e)}"}), 500
        
        # Step 3: Refine the instructions with GPT-4o
        try:
            refinement_response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are an assistant that helps refine image editing instructions. Make the instructions precise and clear for DALL-E 3 to understand."
                        },
                        {
                            "role": "user",
                            "content": f"Image analysis: {image_analysis}\n\nOriginal instructions: {instructions}\n\nPlease refine these instructions to be precise and clear for DALL-E 3 to understand, based on the image analysis."
                        }
                    ],
                    "max_tokens": 500
                }
            )
            
            refinement_response.raise_for_status()
            refinement_result = refinement_response.json()
            refined_instructions = refinement_result["choices"][0]["message"]["content"]
            
            logger.info("Successfully refined instructions with GPT-4o")
        except Exception as e:
            logger.error(f"Error refining instructions with GPT-4o: {str(e)}")
            return jsonify({"error": f"Error refining instructions with GPT-4o: {str(e)}"}), 500
        
        # Step 4: Generate the modified image with DALL-E 3
        try:
            dalle_prompt = f"Edit this image according to these instructions: {refined_instructions}"
            
            dalle_response = requests.post(
                "https://api.openai.com/v1/images/edits",
                headers=headers,
                files={
                    "image": ("image.jpg", BytesIO(image_data), "image/jpeg"),
                },
                data={
                    "prompt": dalle_prompt,
                    "n": 1,
                    "size": "1024x1024",
                    "response_format": "b64_json"
                }
            )
            
            # If DALL-E edit fails, try DALL-E generation with the original image as reference
            if dalle_response.status_code != 200:
                logger.warning(f"DALL-E edit failed, trying DALL-E generation: {dalle_response.text}")
                
                dalle_generation_prompt = f"Create a new version of this image with the following changes: {refined_instructions}"
                
                dalle_response = requests.post(
                    "https://api.openai.com/v1/images/generations",
                    headers=headers,
                    json={
                        "model": "dall-e-3",
                        "prompt": dalle_generation_prompt,
                        "n": 1,
                        "size": "1024x1024",
                        "response_format": "b64_json"
                    }
                )
            
            dalle_response.raise_for_status()
            dalle_result = dalle_response.json()
            
            # Extract the base64 image data
            if "data" in dalle_result and len(dalle_result["data"]) > 0:
                if "b64_json" in dalle_result["data"][0]:
                    modified_image_base64 = dalle_result["data"][0]["b64_json"]
                else:
                    # If using URL response format
                    image_url = dalle_result["data"][0]["url"]
                    img_response = requests.get(image_url)
                    img_response.raise_for_status()
                    modified_image_base64 = base64.b64encode(img_response.content).decode('utf-8')
            
            logger.info("Successfully generated modified image with DALL-E 3")
        except Exception as e:
            logger.error(f"Error generating image with DALL-E 3: {str(e)}")
            return jsonify({"error": f"Error generating image with DALL-E 3: {str(e)}"}), 500
        
        # Return the results
        return jsonify({
            "original_instructions": instructions,
            "refined_instructions": refined_instructions,
            "image_analysis": image_analysis,
            "modified_image": modified_image_base64
        }), 200
    
    except Exception as e:
        logger.error(f"Unexpected error in process-image endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)