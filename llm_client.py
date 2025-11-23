# llm_client.py - Universal API Key Support (No Validation)

import requests
import json
import os
from typing import List, Dict, Optional

class LLMClient:
    """
    Universal LLM Client supporting ANY API provider:
    - Ollama (local)
    - OpenAI
    - Anthropic Claude
    - OpenRouter
    - ANY custom API endpoint with OpenAI-compatible format
    """
    
    def __init__(self, provider_name: Optional[str] = None, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.provider_name = provider_name or self.config.get("default_provider", "ollama")
        self.provider_config = self._get_provider_config()
        self.model = self.provider_config.get("default_model", "qwen2.5:0.5b")
        
        print(f"✅ LLMClient initialized: {self.provider_name} | model: {self.model}")
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Config file not found: {config_path}, using defaults")
            return {
                "providers": {
                    "ollama": {
                        "base_url": "http://localhost:11434/api",
                        "default_model": "qwen2.5:0.5b"
                    }
                },
                "default_provider": "ollama"
            }
    
    def _get_provider_config(self) -> dict:
        """Get configuration for the selected provider"""
        providers = self.config.get("providers", {})
        
        if self.provider_name not in providers:
            raise ValueError(f"Provider '{self.provider_name}' not found in config")
        
        return providers[self.provider_name]
    
    def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None) -> str:
        """
        Send chat messages to AI provider
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Optional model override
            
        Returns:
            AI response as string
        """
        target_model = model or self.model
        
        if self.provider_name.startswith("ollama"):
            return self._chat_ollama(messages, target_model)
        elif self.provider_name == "anthropic":
            return self._chat_anthropic(messages, target_model)
        else:
            # Universal handler for OpenAI-compatible APIs
            # This includes: OpenAI, OpenRouter, and ANY custom provider
            return self._chat_openai_compatible(messages, target_model)
    
    def _chat_ollama(self, messages: List[Dict], model: str) -> str:
        """Chat with Ollama (local)"""
        base_url = self.provider_config.get("base_url", "http://localhost:11434/api")
        
        try:
            response = requests.post(
                f"{base_url}/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_ctx": self.provider_config.get("num_ctx", 1024),
                        "num_gpu": self.provider_config.get("num_gpu", 0)
                    }
                },
                timeout=180
            )
            
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
            
        except Exception as e:
            raise ValueError(f"Ollama Error: {str(e)}")
    
    def _chat_openai_compatible(self, messages: List[Dict], model: str) -> str:
        """
        Universal handler for OpenAI-compatible APIs
        Works with: OpenAI, OpenRouter, Azure OpenAI, and ANY custom endpoint
        
        ✅ Accepts ANY API key format - no validation
        """
        api_key = self.provider_config.get("api_key", "")
        base_url = self.provider_config.get("base_url", "https://api.openai.com/v1")
        
        if not api_key:
            raise ValueError(f"{self.provider_name} API key not configured")
        
        # Log key info (first 10 chars + last 4 chars for security)
        if len(api_key) > 14:
            key_preview = f"{api_key[:10]}...{api_key[-4:]}"
        else:
            key_preview = f"{api_key[:4]}...{api_key[-2:]}"
        
        print(f"🔑 Using {self.provider_name} API key: {key_preview}")
        print(f"🌐 Endpoint: {base_url}")
        print(f"🤖 Model: {model}")
        
        # Build headers (compatible with most providers)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        # Add provider-specific headers
        if self.provider_name == "openrouter" or "openrouter" in base_url:
            headers.update({
                "HTTP-Referer": "https://voxen-ai.local",
                "X-Title": "VOXEN AI"
            })
        
        # Build request payload
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self.provider_config.get("temperature", 0.7),
            "max_tokens": self.provider_config.get("max_tokens", 1000)
        }
        
        # Make request
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            # Enhanced error messages (but accept ANY key format)
            if response.status_code == 401:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("error", {}).get("message", "Authentication failed")
                raise ValueError(
                    f"❌ {self.provider_name} Authentication Failed: {error_msg}\n"
                    f"   Please check:\n"
                    f"   1. API key is correct and complete\n"
                    f"   2. API key has not expired\n"
                    f"   3. Account has active credits/subscription"
                )
            
            elif response.status_code == 402:
                raise ValueError(
                    f"❌ {self.provider_name} Payment Required: Your account has insufficient credits. "
                    f"Please add credits or check your billing."
                )
            
            elif response.status_code == 429:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("error", {}).get("message", "Rate limit exceeded")
                raise ValueError(
                    f"❌ {self.provider_name} Rate Limit: {error_msg}\n"
                    f"   Please try again in a few moments."
                )
            
            elif response.status_code == 400:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("error", {}).get("message", "Bad request")
                raise ValueError(f"❌ {self.provider_name} Request Error: {error_msg}")
            
            elif response.status_code == 404:
                raise ValueError(
                    f"❌ {self.provider_name} Model Not Found: '{model}' is not available.\n"
                    f"   Please check the model name is correct."
                )
            
            # Raise for any other HTTP errors
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Extract content (standard OpenAI format)
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                raise ValueError(f"Unexpected response format from {self.provider_name}")
            
        except requests.exceptions.Timeout:
            raise ValueError(f"❌ {self.provider_name} Timeout: Request took too long. Please try again.")
        
        except requests.exceptions.ConnectionError:
            raise ValueError(
                f"❌ {self.provider_name} Connection Error: Cannot reach API endpoint.\n"
                f"   Check your internet connection."
            )
        
        except requests.exceptions.RequestException as e:
            raise ValueError(f"❌ {self.provider_name} Network Error: {str(e)}")
    
    def _chat_anthropic(self, messages: List[Dict], model: str) -> str:
        """
        Chat with Anthropic Claude (uses different API format)
        ✅ Accepts ANY API key format - no validation
        """
        api_key = self.provider_config.get("api_key", "")
        base_url = self.provider_config.get("base_url", "https://api.anthropic.com/v1")
        
        if not api_key:
            raise ValueError("Anthropic API key not configured")
        
        # Log key info
        if len(api_key) > 14:
            key_preview = f"{api_key[:10]}...{api_key[-4:]}"
        else:
            key_preview = f"{api_key[:4]}...{api_key[-2:]}"
        
        print(f"🔑 Using Anthropic API key: {key_preview}")
        print(f"🌐 Endpoint: {base_url}")
        print(f"🤖 Model: {model}")
        
        # Convert messages format for Claude
        system_message = ""
        claude_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                claude_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Build payload
        payload = {
            "model": model,
            "max_tokens": self.provider_config.get("max_tokens", 1000),
            "messages": claude_messages
        }
        
        if system_message:
            payload["system"] = system_message
        
        try:
            response = requests.post(
                f"{base_url}/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=60
            )
            
            # Error handling
            if response.status_code == 401:
                error_data = response.json() if response.content else {}
                error_msg = error_data.get("error", {}).get("message", "Authentication failed")
                raise ValueError(f"❌ Anthropic Authentication Failed: {error_msg}")
            
            elif response.status_code == 429:
                raise ValueError("❌ Anthropic Rate Limit Exceeded. Please try again later.")
            
            response.raise_for_status()
            data = response.json()
            
            return data["content"][0]["text"]
            
        except requests.exceptions.RequestException as e:
            raise ValueError(f"❌ Anthropic Connection Error: {str(e)}")
    
    def list_models(self) -> List[str]:
        """List available models for the current provider"""
        if self.provider_name.startswith("ollama"):
            return self._list_ollama_models()
        else:
            return [self.model]
    
    def _list_ollama_models(self) -> List[str]:
        """List available Ollama models"""
        try:
            base_url = self.provider_config.get("base_url", "http://localhost:11434/api")
            response = requests.get(f"{base_url.replace('/api', '')}/api/tags", timeout=5)
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            print(f"⚠️ Failed to list Ollama models: {e}")
            return [self.model]


# Helper function to test any API key
def test_api_key(provider: str, api_key: str, model: str, base_url: str = None):
    """
    Test any API key with any provider
    
    Args:
        provider: Provider name (openai, openrouter, anthropic, custom)
        api_key: Any API key format
        model: Model to test
        base_url: Optional custom endpoint URL
    
    Returns:
        dict with success status and message
    """
    import tempfile
    
    # Create temporary config
    config = {
        "providers": {
            provider: {
                "api_key": api_key,
                "default_model": model
            }
        },
        "default_provider": provider
    }
    
    # Add base_url if provided
    if base_url:
        config["providers"][provider]["base_url"] = base_url
    elif provider == "openai":
        config["providers"][provider]["base_url"] = "https://api.openai.com/v1"
    elif provider == "openrouter":
        config["providers"][provider]["base_url"] = "https://openrouter.ai/api/v1"
    elif provider == "anthropic":
        config["providers"][provider]["base_url"] = "https://api.anthropic.com/v1"
    
    # Save to temp file
    temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config, temp_config)
    temp_config.close()
    
    try:
        # Test the connection
        client = LLMClient(provider_name=provider, config_path=temp_config.name)
        response = client.chat([{"role": "user", "content": "Hi"}], model=model)
        
        os.unlink(temp_config.name)
        
        return {
            "success": True,
            "message": f"Successfully connected to {provider}",
            "response": response[:100] if response else None
        }
    
    except Exception as e:
        os.unlink(temp_config.name)
        return {
            "success": False,
            "message": str(e)
        }
