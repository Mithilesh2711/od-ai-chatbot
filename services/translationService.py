"""
Translation Service - Multilingual Support for Hindi, English, and Hinglish
Provides middleware for translating input/output in any service
"""

from typing import Dict, Any, Optional, Tuple
import re


class TranslationService:
    """
    Translation service using dynamic LLM for multilingual support
    Supports: Hindi, English, and Hinglish
    Uses the same LLM instance as the RAG pipeline
    """

    def __init__(self, llm_instance=None):
        """
        Initialize translation service with optional LLM instance

        Args:
            llm_instance: LangChain LLM instance (ChatOpenAI, ChatAnthropic, etc.)
                         If None, will need to be provided in method calls
        """
        self.llm_instance = llm_instance

        # Language detection prompts
        self.lang_detection_prompt = """Detect the language of the following text and respond with ONLY one word: 'english', 'hindi', or 'hinglish'.

Rules:
- 'english': Text is purely in English
- 'hindi': Text is purely in Hindi/Devanagari script
- 'hinglish': Text is a mix of Hindi and English (Roman script with Hindi words)

Text: {text}

Language:"""

        self.translation_prompts = {
            "to_english": """Translate the following {source_lang} text to English. Preserve the meaning and intent accurately.
If the text is already in English, return it as is.

Text: {text}

English translation:""",

            "from_english": """Translate the following English text to {target_lang}.
Maintain a natural, conversational tone appropriate for the target language.

Text: {text}

{target_lang} translation:"""
        }


    def detect_language(self, text: str, llm_instance=None) -> str:
        """
        Detect if text is in English, Hindi, or Hinglish

        Args:
            text: Input text to detect language
            llm_instance: Optional LLM instance to use (overrides instance llm)

        Returns:
            str: 'english', 'hindi', or 'hinglish'
        """
        try:
            # Quick heuristic check before using LLM (optimization)
            # Check for Devanagari script (Hindi)
            if re.search(r'[\u0900-\u097F]', text):
                return 'hindi'

            # Check if purely ASCII English (common case)
            if text.isascii() and not self._has_hinglish_patterns(text):
                return 'english'

            # Use LLM for accurate detection (Hinglish and edge cases)
            llm = llm_instance or self.llm_instance
            if not llm:
                print("⚠️ No LLM instance available, defaulting to 'english'")
                return 'english'

            prompt = self.lang_detection_prompt.format(text=text)

            # Use LangChain invoke method
            response = llm.invoke(prompt)

            # Extract content from response
            if hasattr(response, 'content'):
                detected_lang = response.content.strip().lower()
            else:
                detected_lang = str(response).strip().lower()

            # Validate response
            if detected_lang not in ['english', 'hindi', 'hinglish']:
                print(f"⚠️ Unexpected language detection result: {detected_lang}, defaulting to 'english'")
                return 'english'

            print(f"🌐 Detected language: {detected_lang}")
            return detected_lang

        except Exception as e:
            print(f"❌ Error detecting language: {str(e)}, defaulting to 'english'")
            return 'english'


    def _has_hinglish_patterns(self, text: str) -> bool:
        """
        Check for common Hinglish patterns (heuristic)
        """
        hinglish_words = [
            'kya', 'hai', 'hain', 'kar', 'ke', 'ki', 'ko', 'se', 'me', 'mein',
            'aap', 'aapka', 'tumhara', 'mera', 'tera', 'kaise', 'kahan', 'kyun',
            'bahut', 'thoda', 'accha', 'theek', 'zaroor', 'bilkul', 'nahi', 'haan'
        ]

        text_lower = text.lower()
        for word in hinglish_words:
            if re.search(rf'\b{word}\b', text_lower):
                return True
        return False


    def translate_to_english(self, text: str, source_lang: str, llm_instance=None) -> str:
        """
        Translate text from Hindi/Hinglish to English

        Args:
            text: Text to translate
            source_lang: Source language ('hindi' or 'hinglish')
            llm_instance: Optional LLM instance to use (overrides instance llm)

        Returns:
            str: Translated English text
        """
        # If already English, return as is
        if source_lang == 'english':
            print("✅ Text already in English, skipping translation")
            return text

        try:
            llm = llm_instance or self.llm_instance
            if not llm:
                print("⚠️ No LLM instance available, returning original text")
                return text

            prompt = self.translation_prompts["to_english"].format(
                source_lang=source_lang.capitalize(),
                text=text
            )

            # Use LangChain invoke method
            response = llm.invoke(prompt)

            # Extract content from response
            if hasattr(response, 'content'):
                translated_text = response.content.strip()
            else:
                translated_text = str(response).strip()

            print(f"🔄 Translated to English: {translated_text[:100]}...")
            return translated_text

        except Exception as e:
            print(f"❌ Error translating to English: {str(e)}, returning original text")
            return text


    def translate_from_english(self, text: str, target_lang: str, llm_instance=None) -> str:
        """
        Translate text from English to Hindi/Hinglish

        Args:
            text: English text to translate
            target_lang: Target language ('hindi' or 'hinglish')
            llm_instance: Optional LLM instance to use (overrides instance llm)

        Returns:
            str: Translated text in target language
        """
        # If target is English, return as is
        if target_lang == 'english':
            print("✅ Target language is English, skipping translation")
            return text

        try:
            llm = llm_instance or self.llm_instance
            if not llm:
                print("⚠️ No LLM instance available, returning original text")
                return text

            prompt = self.translation_prompts["from_english"].format(
                target_lang=target_lang.capitalize(),
                text=text
            )

            # Use LangChain invoke method
            response = llm.invoke(prompt)

            # Extract content from response
            if hasattr(response, 'content'):
                translated_text = response.content.strip()
            else:
                translated_text = str(response).strip()

            print(f"🔄 Translated to {target_lang}: {translated_text[:100]}...")
            return translated_text

        except Exception as e:
            print(f"❌ Error translating from English: {str(e)}, returning original text")
            return text


    def process_multilingual_input(self, text: str, llm_instance=None) -> Tuple[str, str]:
        """
        Process multilingual input: detect language and translate to English if needed

        Args:
            text: Input text in any language (Hindi/English/Hinglish)
            llm_instance: Optional LLM instance to use (overrides instance llm)

        Returns:
            Tuple[str, str]: (english_text, detected_language)
        """
        # Detect language
        detected_lang = self.detect_language(text, llm_instance)

        # Translate to English if needed
        english_text = self.translate_to_english(text, detected_lang, llm_instance)

        return english_text, detected_lang


    def process_multilingual_output(self, text: str, target_lang: str, llm_instance=None) -> str:
        """
        Process multilingual output: translate English response to target language

        Args:
            text: English text to translate
            target_lang: Target language ('english', 'hindi', or 'hinglish')
            llm_instance: Optional LLM instance to use (overrides instance llm)

        Returns:
            str: Translated text in target language
        """
        return self.translate_from_english(text, target_lang, llm_instance)


# Global translation service instance (no LLM - will be provided per request)
translation_service = TranslationService()


def with_translation(enabled: bool = True):
    """
    Decorator to add translation middleware to any async service function

    Usage:
        @with_translation(enabled=True)
        async def my_service(entity: str, query: str, model_name: str = None, **kwargs) -> Dict[str, Any]:
            # Your service logic here
            return {"answer": "...", "query": query}

    The decorator expects the service function to accept 'query' and optionally 'llm_instance' in kwargs.

    Args:
        enabled: Whether translation is enabled (from config)

    Returns:
        Decorated function with translation support
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # If translation disabled, call original function
            if not enabled:
                return await func(*args, **kwargs)

            # Extract query from kwargs or args
            # Assuming query is either in kwargs['query'] or second positional arg
            query = kwargs.get('query')
            if query is None and len(args) >= 2:
                query = args[1]

            if query is None:
                print("⚠️ No query found in function args, skipping translation")
                return await func(*args, **kwargs)

            # Extract or initialize LLM instance
            llm_instance = kwargs.get('llm_instance')

            # If no LLM instance, initialize it based on model_name
            if not llm_instance:
                model_name = kwargs.get('model_name') or (args[2] if len(args) > 2 else None)
                if not model_name:
                    # Try to import and use default model
                    try:
                        from config.settings import DEFAULT_MODEL
                        model_name = DEFAULT_MODEL
                    except:
                        model_name = None

                if model_name:
                    try:
                        from graphs.leadsGraph import get_llm_instance
                        llm_instance = get_llm_instance(model_name)
                        kwargs['llm_instance'] = llm_instance
                        print(f"🔧 Translation middleware initialized LLM: {model_name}")
                    except Exception as e:
                        print(f"⚠️ Could not initialize LLM for translation: {str(e)}")
                        llm_instance = None

            print(f"\n🌐 === Translation Middleware Active ===")
            print(f"Original query: {query}")

            # Step 1: Detect language and translate query to English
            english_query, detected_lang = translation_service.process_multilingual_input(
                query,
                llm_instance=llm_instance
            )

            # Update query in kwargs or args
            if 'query' in kwargs:
                kwargs['query'] = english_query
            else:
                # Reconstruct args with translated query
                args_list = list(args)
                args_list[1] = english_query
                args = tuple(args_list)

            # Step 2: Call original function with English query
            result = await func(*args, **kwargs)

            # Step 3: Translate response back to detected language
            if isinstance(result, dict) and 'answer' in result:
                original_answer = result['answer']
                translated_answer = translation_service.process_multilingual_output(
                    original_answer,
                    detected_lang,
                    llm_instance=llm_instance
                )
                result['answer'] = translated_answer
                result['detected_language'] = detected_lang
                result['original_query'] = query

                print(f"✅ Translation complete - Response in {detected_lang}")
                print(f"🌐 === Translation Middleware End ===\n")

            return result

        return wrapper
    return decorator
