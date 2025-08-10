import os
import base64
import json
from typing import Optional, Dict, Any, List
from pathlib import Path
import requests

def _encode_image(image_path: str) -> tuple[str, str]:
    """Encode image and return (base64_data, mime_type)"""
    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Detect MIME type from file extension
    ext = Path(image_path).suffix.lower()
    if ext in ['.jpg', '.jpeg']:
        mime_type = 'image/jpeg'
    elif ext == '.png':
        mime_type = 'image/png'
    elif ext == '.webp':
        mime_type = 'image/webp'
    elif ext == '.gif':
        mime_type = 'image/gif'
    else:
        mime_type = 'image/jpeg'  # Default fallback
    
    return base64_data, mime_type

def _query_openai(prompt: str, model: str = "gpt-4", image: Optional[str] = None, 
                  temperature: float = 0.7, api_key: Optional[str] = None, 
                  max_tokens: Optional[int] = None) -> Dict[str, Any]:
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if image:
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")
        
        # Auto-select vision model if regular model provided with image
        if model in ["gpt-4", "gpt-3.5-turbo"]:
            model = "gpt-4o"  # Use gpt-4o as default vision model
        
        base64_image, mime_type = _encode_image(image)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
            ]
        })
    else:
        messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    
    # Add max_tokens if specified or if using vision model
    if max_tokens or image:
        data["max_tokens"] = max_tokens or 1024
    
    response = requests.post("https://api.openai.com/v1/chat/completions", 
                           headers=headers, json=data)
    response.raise_for_status()
    
    result = response.json()
    return {
        "content": result["choices"][0]["message"]["content"],
        "usage": result.get("usage", {}),
        "model": model,
        "provider": "openai"
    }

def _query_anthropic(prompt: str, model: str = "claude-3-sonnet-20240229", 
                     image: Optional[str] = None, temperature: float = 0.7, 
                     api_key: Optional[str] = None, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable or pass api_key parameter.")
    
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    content = []
    if image:
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")
        
        base64_image, mime_type = _encode_image(image)
        
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": base64_image
            }
        })
    
    content.append({"type": "text", "text": prompt})
    
    data = {
        "model": model,
        "max_tokens": max_tokens or 4096,
        "temperature": temperature,
        "messages": [{"role": "user", "content": content}]
    }
    
    response = requests.post("https://api.anthropic.com/v1/messages", 
                           headers=headers, json=data)
    response.raise_for_status()
    
    result = response.json()
    return {
        "content": result["content"][0]["text"],
        "usage": result.get("usage", {}),
        "model": model,
        "provider": "anthropic"
    }

def _query_gemini(prompt: str, model: str = "gemini-pro", image: Optional[str] = None, 
                  temperature: float = 0.7, api_key: Optional[str] = None,
                  max_tokens: Optional[int] = None) -> Dict[str, Any]:
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Gemini API key not found. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
    
    if image:
        model = "gemini-pro-vision"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    parts = [{"text": prompt}]
    if image:
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")
        
        base64_image, mime_type = _encode_image(image)
        
        parts.append({
            "inline_data": {
                "mime_type": mime_type,
                "data": base64_image
            }
        })
    
    generation_config = {"temperature": temperature}
    if max_tokens:
        generation_config["maxOutputTokens"] = max_tokens
    
    data = {
        "contents": [{"parts": parts}],
        "generationConfig": generation_config
    }
    
    response = requests.post(url, json=data)
    response.raise_for_status()
    
    result = response.json()
    return {
        "content": result["candidates"][0]["content"]["parts"][0]["text"],
        "usage": result.get("usageMetadata", {}),
        "model": model,
        "provider": "gemini"
    }

def _query_grok(prompt: str, model: str = "grok-beta", image: Optional[str] = None, 
                temperature: float = 0.7, api_key: Optional[str] = None,
                max_tokens: Optional[int] = None) -> Dict[str, Any]:
    api_key = api_key or os.getenv("GROK_API_KEY")
    if not api_key:
        raise ValueError("Grok API key not found. Set GROK_API_KEY environment variable or pass api_key parameter.")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    messages = []
    if image:
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")
        
        base64_image, mime_type = _encode_image(image)
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}}
            ]
        })
    else:
        messages.append({"role": "user", "content": prompt})
    
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    
    if max_tokens:
        data["max_tokens"] = max_tokens
    
    response = requests.post("https://api.x.ai/v1/chat/completions", 
                           headers=headers, json=data)
    response.raise_for_status()
    
    result = response.json()
    return {
        "content": result["choices"][0]["message"]["content"],
        "usage": result.get("usage", {}),
        "model": model,
        "provider": "grok"
    }

def query_ai(prompt: str, model: str = "gpt-4", model_type: str = "openai", 
             image: Optional[str] = None, temperature: float = 0.7, 
             api_key: Optional[str] = None, max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """
    Query various AI models with a unified interface.
    
    Args:
        prompt: The text prompt to send to the AI
        model: The specific model to use (e.g., "gpt-4", "claude-3-sonnet-20240229")
        model_type: The AI provider ("openai", "anthropic", "gemini", "grok")
        image: Optional path to an image file
        temperature: Sampling temperature (0.0 to 1.0)
        api_key: Optional API key (will use environment variables if not provided)
        max_tokens: Maximum tokens to generate
    
    Returns:
        Dict containing the AI response, usage info, and metadata. You prob want dict['content']
    """
    
    if model_type.lower() == "openai":
        return _query_openai(prompt, model, image, temperature, api_key, max_tokens)
    elif model_type.lower() == "anthropic":
        return _query_anthropic(prompt, model, image, temperature, api_key, max_tokens)
    elif model_type.lower() == "gemini":
        return _query_gemini(prompt, model, image, temperature, api_key, max_tokens)
    elif model_type.lower() == "grok":
        return _query_grok(prompt, model, image, temperature, api_key, max_tokens)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Supported types: openai, anthropic, gemini, grok")

def generate_image(prompt: str, model: str = "dall-e-3", size: str = "1024x1024", 
                   quality: str = "standard", api_key: Optional[str] = None, 
                   output_path: Optional[str] = None) -> str:
    """
    Generate images using OpenAI's DALL-E models.
    
    Args:
        prompt: Text description of the image to generate
        model: Model to use ("dall-e-2" or "dall-e-3")
        size: Image size ("256x256", "512x512", "1024x1024", "1792x1024", "1024x1792")
        quality: Image quality ("standard" or "hd") - dall-e-3 only
        api_key: Optional API key
        output_path: Optional path to save the image
    
    Returns:
        URL of the generated image or path if saved locally
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "prompt": prompt,
        "size": size,
        "n": 1
    }
    
    if model == "dall-e-3":
        data["quality"] = quality
    
    response = requests.post("https://api.openai.com/v1/images/generations", 
                           headers=headers, json=data)
    response.raise_for_status()
    
    result = response.json()
    image_url = result["data"][0]["url"]
    
    if output_path:
        image_response = requests.get(image_url)
        image_response.raise_for_status()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            f.write(image_response.content)
        
        return output_path
    
    return image_url

def debug_info(response: Dict[str, Any], verbose: bool = True) -> None:
    """
    Display debug information about an AI query response with colorful output.
    
    Args:
        response: Response dictionary from query_ai()
        verbose: Whether to show detailed information
    """
    try:
        from colorama import Fore, Style, init
        init(autoreset=True)
    except ImportError:
        class Fore:
            RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = ""
        class Style:
            BRIGHT = RESET_ALL = ""
    
    print(f"\n{Fore.CYAN}{Style.BRIGHT}🤖 AI Query Debug Info{Style.RESET_ALL}")
    print(f"{Fore.BLUE}├─ Provider: {Fore.WHITE}{response.get('provider', 'unknown')}")
    print(f"{Fore.BLUE}├─ Model: {Fore.WHITE}{response.get('model', 'unknown')}")
    
    usage = response.get('usage', {})
    if usage:
        print(f"{Fore.GREEN}├─ Token Usage:")
        
        if 'prompt_tokens' in usage:
            print(f"{Fore.GREEN}│  ├─ Input tokens: {Fore.YELLOW}{usage['prompt_tokens']:,}")
        if 'completion_tokens' in usage:
            print(f"{Fore.GREEN}│  ├─ Output tokens: {Fore.YELLOW}{usage['completion_tokens']:,}")
        if 'total_tokens' in usage:
            print(f"{Fore.GREEN}│  └─ Total tokens: {Fore.YELLOW}{usage['total_tokens']:,}")
        
        if 'promptTokens' in usage:
            print(f"{Fore.GREEN}│  ├─ Input tokens: {Fore.YELLOW}{usage['promptTokens']:,}")
        if 'candidatesTokens' in usage:
            print(f"{Fore.GREEN}│  ├─ Output tokens: {Fore.YELLOW}{usage['candidatesTokens']:,}")
        if 'totalTokens' in usage:
            print(f"{Fore.GREEN}│  └─ Total tokens: {Fore.YELLOW}{usage['totalTokens']:,}")
        
        if 'input_tokens' in usage:
            print(f"{Fore.GREEN}│  ├─ Input tokens: {Fore.YELLOW}{usage['input_tokens']:,}")
        if 'output_tokens' in usage:
            print(f"{Fore.GREEN}│  ├─ Output tokens: {Fore.YELLOW}{usage['output_tokens']:,}")
    
    if verbose:
        content = response.get('content', '')
        content_length = len(content)
        word_count = len(content.split())
        
        print(f"{Fore.MAGENTA}├─ Response Stats:")
        print(f"{Fore.MAGENTA}│  ├─ Characters: {Fore.YELLOW}{content_length:,}")
        print(f"{Fore.MAGENTA}│  └─ Words: {Fore.YELLOW}{word_count:,}")
    
    estimated_cost = _estimate_cost(response)
    if estimated_cost > 0:
        print(f"{Fore.RED}└─ Estimated Cost: {Fore.GREEN}${estimated_cost:.6f}")
    else:
        print(f"{Fore.BLUE}└─ Cost estimation not available")

def _estimate_cost(response: Dict[str, Any]) -> float:
    """Estimate the cost of an AI query based on token usage and provider pricing."""
    provider = response.get('provider', '').lower()
    model = response.get('model', '').lower()
    usage = response.get('usage', {})
    
    if provider == 'openai':
        input_tokens = usage.get('prompt_tokens', 0)
        output_tokens = usage.get('completion_tokens', 0)
        
        if 'gpt-4o' in model:
            return (input_tokens * 0.005 + output_tokens * 0.015) / 1000  # GPT-4o pricing
        elif 'gpt-4' in model:
            if 'turbo' in model or 'preview' in model:
                return (input_tokens * 0.01 + output_tokens * 0.03) / 1000
            else:
                return (input_tokens * 0.03 + output_tokens * 0.06) / 1000
        elif 'gpt-3.5' in model:
            return (input_tokens * 0.0015 + output_tokens * 0.002) / 1000
    
    elif provider == 'anthropic':
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        
        if 'claude-3-opus' in model:
            return (input_tokens * 0.015 + output_tokens * 0.075) / 1000
        elif 'claude-3-sonnet' in model:
            return (input_tokens * 0.003 + output_tokens * 0.015) / 1000
        elif 'claude-3-haiku' in model:
            return (input_tokens * 0.00025 + output_tokens * 0.00125) / 1000
    
    elif provider == 'gemini':
        input_tokens = usage.get('promptTokens', 0)
        output_tokens = usage.get('candidatesTokens', 0)
        
        if 'gemini-pro' in model:
            return (input_tokens * 0.00025 + output_tokens * 0.0005) / 1000
    
    return 0.0