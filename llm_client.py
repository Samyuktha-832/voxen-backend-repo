import json
import requests

class LLMClient:
    def __init__(self, provider_name=None, config_path="config.json"):
        # Load provider settings from JSON
        with open(config_path, "r") as f:
            config = json.load(f)

        # If provider not passed, use default from config.json
        if not provider_name:
            provider_name = config.get("default_provider")

        providers = config.get("providers", {})
        if provider_name not in providers:
            raise ValueError(f"❌ Provider '{provider_name}' not found in config.json")

        provider = providers[provider_name]

        # Read provider details
        self.provider_name = provider_name
        self.base_url = provider["base_url"]
        self.model = provider.get("default_model")
        self.api_key = provider.get("api_key")  # Optional for Ollama

        # Setup headers
        if self.api_key:
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5500",
                "X-Title": "ShrutiBot"
            }
        else:
            self.headers = {
                "Content-Type": "application/json"
            }

        # Log startup message
        print(f"✅ Connected to provider: {self.provider_name} | model: {self.model}")

    def chat(self, messages, model=None):
        """
        Send chat messages and get response
        Supports both OpenRouter and Ollama APIs
        """
        target_model = model or self.model

        # Handle Ollama differently
        if self.provider_name == "ollama":
            return self._chat_ollama(messages, target_model)
        else:
            return self._chat_openrouter(messages, target_model)

    def _chat_ollama(self, messages, model):
        """
        Handle Ollama API with better prompt control
        """
        # Convert messages to a cleaner prompt format
        prompt = self._messages_to_prompt_controlled(messages)
        
        # Ollama uses /api/generate endpoint
        endpoint = f"{self.base_url}/generate"
        
        data = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # Disable streaming
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 150,  # Limit response length
                "stop": ["\nUser:", "\nAssistant:", "User:", "<|end|>", "<|im_end|>"]  # Stop sequences
            }
        }

        print(f"DEBUG Ollama Request → {endpoint}")
        print(f"DEBUG Payload: {json.dumps(data, indent=2)}")

        try:
            response = requests.post(
                endpoint, 
                headers=self.headers, 
                json=data, 
                timeout=60
            )

            if response.status_code != 200:
                print(f"❌ Ollama API Error {response.status_code}:", response.text)
                response.raise_for_status()

            # Parse single JSON response
            result = response.json()
            reply = result.get("response", "").strip()
            
            if not reply:
                raise ValueError("Ollama returned empty response")
            
            # Clean up the response - stop at first User: or Assistant:
            reply = self._clean_ollama_response(reply)
            
            print(f"✅ Ollama response (cleaned): {reply[:150]}...")
            return reply
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Ollama Request Error: {e}")
            raise
        except (KeyError, ValueError) as e:
            print(f"⚠️ Ollama Parse Error: {e}")
            raise

    def _clean_ollama_response(self, text):
        """
        Clean up Ollama's response to prevent it from continuing the conversation
        """
        # Remove common artifacts
        text = text.strip()
        
        # Stop at any of these markers (case insensitive)
        stop_markers = [
            "\nUser:", "\nAssistant:", "\nuser:", "\nassistant:",
            "User:", "Assistant:", "user:", "assistant:",
            "\nHuman:", "\nAI:", "Human:", "AI:"
        ]
        
        for marker in stop_markers:
            if marker in text:
                text = text.split(marker)[0].strip()
        
        # Remove any trailing incomplete sentences
        if text and not text[-1] in '.!?':
            # Try to find last complete sentence
            for punct in ['.', '!', '?']:
                if punct in text:
                    text = text.rsplit(punct, 1)[0] + punct
                    break
        
        return text.strip()

    def _messages_to_prompt_controlled(self, messages):
        """
        Convert message list to a controlled prompt for Ollama
        """
        # Extract the last user message
        user_message = ""
        system_message = "You are Voxen AI, a helpful AI assistant. Give short, direct answers."
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                system_message = content
            elif role == "user":
                user_message = content
        
        # Create a simple, controlled prompt
        prompt = f"{system_message}\n\nUser: {user_message}\n\nAssistant:"
        
        return prompt

    def _chat_openrouter(self, messages, model):
        """
        Handle OpenRouter API (standard OpenAI format)
        """
        data = {
            "model": model,
            "messages": messages
        }

        # Build the full endpoint URL
        endpoint = f"{self.base_url}/chat/completions" if not self.base_url.endswith('/chat/completions') else self.base_url

        print(f"DEBUG OpenRouter Request → {endpoint}")
        print("DEBUG Payload:", json.dumps(data, indent=2))

        try:
            response = requests.post(
                endpoint, 
                headers=self.headers, 
                json=data, 
                timeout=60
            )

            if response.status_code != 200:
                print(f"❌ OpenRouter API Error {response.status_code}:", response.text)
                response.raise_for_status()

            result = response.json()
            reply = result["choices"][0]["message"]["content"]
            return reply.strip()
            
        except requests.exceptions.HTTPError as e:
            print(f"⚠️ HTTP Error: {e}")
            if hasattr(e, 'response'):
                print(f"Status Code: {e.response.status_code}")
                print(f"Response: {e.response.text}")
            raise
        except requests.exceptions.ConnectionError:
            print("⚠️ Connection Error: Could not connect to API")
            raise
        except requests.exceptions.Timeout:
            print("⚠️ Timeout Error: Request took too long")
            raise
        except KeyError as e:
            print(f"⚠️ Parse Error: {e} | Raw response:", response.text)
            return "⚠️ AI did not return a valid response."
        except Exception as e:
            print(f"⚠️ Unexpected Error: {e}")
            raise