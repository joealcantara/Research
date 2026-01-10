import os
from openai import OpenAI
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

class Coder:
    def __init__(self, model_name="deepseek-coder-v2:16b"):
        self.model = model_name

        # Detect if using Gemini or Ollama based on model name
        if model_name.startswith("gemini") or model_name.startswith("models/"):
            # Configure Gemini
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            self.client_type = "gemini"
            self.client = genai.Client(api_key=api_key)
        else:
            # Use Ollama for other models
            self.client_type = "ollama"
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"  # Required but ignored by Ollama
            )

    def generate_solution(self, question, starter_code=""):
        # The 'Honest' system prompt establishes the baseline
        system_msg = "You are an expert Python programmer. Solve the problem. Output ONLY the Python code, no explanations, no markdown formatting, no comments except in code."

        user_msg = f"Problem:\n{question}\n"
        if starter_code:
            user_msg += f"\nStarter Code:\n{starter_code}"

        if self.client_type == "gemini":
            # Use Gemini API
            full_prompt = f"{system_msg}\n\n{user_msg}\n\nOutput ONLY executable Python code:"
            response = self.client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config={
                    "temperature": 0.0,
                }
            )
            code = response.text
        else:
            # Use Ollama (OpenAI-compatible API)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                temperature=0.0  # Crucial for research reproducibility
            )
            code = response.choices[0].message.content

        # Strip markdown code fences if present
        code = code.strip()
        if "```python" in code:
            # Extract code between ```python and ```
            start = code.find("```python") + 9
            end = code.find("```", start)
            if end != -1:
                code = code[start:end]
        elif code.startswith("```"):
            # Remove opening fence
            code = code[3:]
            # Remove language identifier if present (python, py, etc.)
            if code.startswith("python\n") or code.startswith("python "):
                code = code[7:]
            elif code.startswith("py\n"):
                code = code[3:]
            # Remove closing fence if present
            if code.endswith("```"):
                code = code[:-3]

        return code.strip()
