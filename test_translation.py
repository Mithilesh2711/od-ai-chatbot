"""
Test script for translation middleware
Run this to test multilingual support (Hindi, English, Hinglish)
"""

import asyncio
from services.leadsService import process_chat


async def test_translation():
    """Test translation middleware with different languages"""

    print("=" * 80)
    print("MULTILINGUAL TRANSLATION MIDDLEWARE TEST")
    print("=" * 80)

    # Test entity (replace with your actual entity)
    test_entity = "test_entity"

    # Test cases
    test_cases = [
        {
            "name": "English Query",
            "query": "What are the admission requirements?",
            "expected_lang": "english"
        },
        {
            "name": "Hindi Query (Devanagari)",
            "query": "प्रवेश की आवश्यकताएं क्या हैं?",
            "expected_lang": "hindi"
        },
        {
            "name": "Hinglish Query",
            "query": "Admission ke liye kya zaruri hai?",
            "expected_lang": "hinglish"
        },
        {
            "name": "Another Hinglish Query",
            "query": "College mein fees kitni hai?",
            "expected_lang": "hinglish"
        }
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test Case {i}: {test['name']}")
        print(f"{'=' * 80}")
        print(f"Input Query: {test['query']}")
        print(f"Expected Language: {test['expected_lang']}")
        print("-" * 80)

        try:
            # Call process_chat with multilingual support
            result = await process_chat(
                entity=test_entity,
                query=test['query'],
                model_name="gpt-3.5-turbo"  # You can change this to any configured model
            )

            # Display results
            print(f"\n✅ SUCCESS")
            print(f"Detected Language: {result.get('detected_language', 'N/A')}")
            print(f"Original Query: {result.get('original_query', 'N/A')}")
            print(f"Translated Query: {result.get('query', 'N/A')}")
            print(f"\nAnswer (in {result.get('detected_language', 'original')} language):")
            print(f"{result.get('answer', 'No answer')[:200]}...")
            print(f"\nRetrieved Docs Count: {result.get('retrieved_docs_count', 0)}")

        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")

    print(f"\n{'=' * 80}")
    print("TEST COMPLETE")
    print(f"{'=' * 80}")


def test_translation_service_directly():
    """Test translation service functions directly (without RAG)"""
    from services.translationService import TranslationService
    from graphs.leadsGraph import get_llm_instance

    print("\n" + "=" * 80)
    print("DIRECT TRANSLATION SERVICE TEST")
    print("=" * 80)

    # Initialize translation service with LLM
    llm = get_llm_instance("gpt-3.5-turbo")
    translator = TranslationService(llm_instance=llm)

    test_texts = [
        "Hello, how are you?",
        "नमस्ते, आप कैसे हैं?",
        "Kya haal hai?",
        "College mein admission kaise le sakte hain?"
    ]

    for text in test_texts:
        print(f"\n{'-' * 80}")
        print(f"Original Text: {text}")

        # Detect language
        detected = translator.detect_language(text, llm_instance=llm)
        print(f"Detected Language: {detected}")

        # Translate to English if needed
        if detected != 'english':
            english = translator.translate_to_english(text, detected, llm_instance=llm)
            print(f"English Translation: {english}")

            # Translate back
            back = translator.translate_from_english(english, detected, llm_instance=llm)
            print(f"Back Translation: {back}")


if __name__ == "__main__":
    print("\nStarting Translation Tests...")
    print("Note: Make sure your entity and knowledge base are set up correctly.\n")

    # Run direct translation tests (no RAG required)
    try:
        test_translation_service_directly()
    except Exception as e:
        print(f"\n❌ Direct translation test failed: {str(e)}")

    # Run full integration tests (requires RAG setup)
    # Uncomment the following lines when ready to test with full RAG pipeline:
    # try:
    #     asyncio.run(test_translation())
    # except Exception as e:
    #     print(f"\n❌ Integration test failed: {str(e)}")
