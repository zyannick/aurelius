import ataraxai
print("Successfully imported 'ataraxai'!")
print(f"Location: {ataraxai.__file__}") # Should point to your project's ataraxai/__init__.py for editable

from ataraxai import core_ai_py
print("Successfully imported 'core_ai_py' from 'ataraxai'!")
service = core_ai_py.CoreAIService()
print("Successfully created CoreAIService instance!")