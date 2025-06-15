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

CORS(app, resources={r"/*": {"origins": "https://image-editor-amq7.onrender.com"}})

# Log environment setup
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    logger.critical("OPENAI_API_KEY environment variable is not set!")
    raise ValueError("OPENAI_API_KEY environment variable is not set!")
else:
    logger.info(f"OPENAI_API_KEY is set (length: {len(api_key)})")

logger.info("Initializing OpenAI client with custom HTTP client")
try:
    http_client = httpx.Client(
        timeout=120.0,
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
        http2=False  
    )
    logger.info(f"HTTPX client initialized with timeout: 120.0s, HTTP/2: disabled")
    
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

@app.route('/', methods=['GET'])
def fun():
    logger.info("Root endpoint accessed")
    return {'message': 'Image Editor API is running!'}


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
    
    try:
        request_data = request.json
        if not request_data or 'prompt' not in request_data:
            logger.error("Missing 'prompt' in request data")
            return jsonify({
                "status": "error",
                "error": "Missing 'prompt' in request data"
            }), 400
        
        user_prompt = request_data['prompt']
        logger.info(f"Received design prompt: {user_prompt}")
        
        current_state = request_data.get('currentState', {"filters": {}})
        logger.info(f"Current state: {current_state}")
        
        image_params = request_data.get('imageParams', {
            "width": 500,
            "height": 500,
            "format": "jpg"
        })
        logger.info(f"Image parameters: {image_params}")
        
        full_prompt = f"""
User request: {user_prompt}

Current image state:
{json.dumps(current_state, indent=2)}

Image parameters:
{json.dumps(image_params, indent=2)}

Please provide the JSON object with the necessary changes based on the request and current state.
"""
        logger.info(f"Sending request to OpenAI API with model: gpt-4o-mini")
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
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
            content = content.strip()
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
                    filters_obj = {"filters": filters_obj}
                else:
                    logger.warning(f"Response does not contain 'filters' key: {filters_obj}")
                    filters_obj = {"filters": {}}
            
           
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
