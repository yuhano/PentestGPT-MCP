import dataclasses
import importlib
import os
import sys

module_mapping = {
    "gpt-4o": {
        "config_name": "GPT4O",
        "module_name": "chatgpt_api",
        "class_name": "ChatGPTAPI",
    },
    "o3": {  # Added GPT-o3
        "config_name": "GPT4O3",
        "module_name": "chatgpt_api",
        "class_name": "ChatGPTAPI",
    },
    "o4-mini": {  # Added GPT-o4-mini
        "config_name": "GPT4O4Mini",
        "module_name": "chatgpt_api",
        "class_name": "ChatGPTAPI",
    },
    "gpt4all": {
        "config_name": "GPT4ALLConfigClass",
        "module_name": "gpt4all_api",
        "class_name": "GPT4ALLAPI",
    },
    "gemini-2.5-flash": {  # Added Gemini 2.5
        "config_name": "Gemini25ConfigClass",
        "module_name": "gemini_api",
        "class_name": "GeminiAPI",
    },
    "gemini-2.5-pro": {  # Added Gemini 2.5
        "config_name": "Gemini25ProConfigClass",
        "module_name": "gemini_api",
        "class_name": "GeminiAPI",
    },
    "deepseek-r1": {  # Added Deepseek R1
        "config_name": "DeepseekR1ConfigClass",
        "module_name": "deepseek_api",
        "class_name": "DeepseekAPI",
    },
    "deepseek-v3": {  # Added Deepseek v3
        "config_name": "DeepseekV3ConfigClass",
        "module_name": "deepseek_api",
        "class_name": "DeepseekAPI",
    },
}


@dataclasses.dataclass
class GPT4O:
    model: str = "gpt-4o"
    api_base: str = os.getenv("OPENAI_BASEURL", "https://api.openai.com/v1")
    # set up the openai key
    openai_key = os.getenv("OPENAI_API_KEY", None)
    if openai_key is None:
        print(
            "Your OPENAI_API_KEY is not set. Please set it in the environment variable."
        )
    error_wait_time: float = 10
    is_debugging: bool = False
    log_dir: str = None


@dataclasses.dataclass
class GPT4O3:  # Added GPT-o3 configuration
    model: str = "o3"
    api_base: str = os.getenv("OPENAI_BASEURL", "https://api.openai.com/v1")
    # set up the openai key
    openai_key = os.getenv("OPENAI_API_KEY", None)
    if openai_key is None:
        print(
            "Your OPENAI_API_KEY is not set. Please set it in the environment variable."
        )
    error_wait_time: float = 10
    is_debugging: bool = False
    log_dir: str = None


@dataclasses.dataclass
class GPT4O4Mini:  # Added GPT-o4-mini configuration
    model: str = "o4-mini"
    api_base: str = os.getenv("OPENAI_BASEURL", "https://api.openai.com/v1")
    # set up the openai key
    openai_key = os.getenv("OPENAI_API_KEY", None)
    if openai_key is None:
        print(
            "Your OPENAI_API_KEY is not set. Please set it in the environment variable."
        )
    error_wait_time: float = 10
    is_debugging: bool = False
    log_dir: str = None


@dataclasses.dataclass
class GPT4ALLConfigClass:
    model: str = "mistral-7b-openorca.Q4_0.gguf"


@dataclasses.dataclass
class TitanConfigClass:
    model: str = "amazon.titan-tg1-large"


@dataclasses.dataclass
class Gemini25ConfigClass:  # Added Gemini 2.5 flash configuration
    model: str = "gemini-2.5-flash-preview-04-17"
    gemini_key = os.getenv(
        "GEMINI_API_KEY", None
    )  # Assuming environment variable for API key
    if gemini_key is None:
        print(
            "Your GEMINI_API_KEY is not set. Please set it in the environment variable."
        )
    error_wait_time: float = 20
    is_debugging: bool = False
    log_dir: str = None


class Gemini25ProConfigClass:  # Added Gemini 2.5 Pro configuration
    model: str = "gemini-2.5-pro-preview-03-25"
    gemini_key = os.getenv(
        "GEMINI_API_KEY", None
    )  # Assuming environment variable for API key
    if gemini_key is None:
        print(
            "Your GEMINI_API_KEY is not set. Please set it in the environment variable."
        )
    error_wait_time: float = 20
    is_debugging: bool = False
    log_dir: str = None


@dataclasses.dataclass
class DeepseekR1ConfigClass:  # Added Deepseek configuration
    model: str = "deepseek-reasoner"
    api_base: str = os.getenv("DEEPSEEK_BASEURL", "https://api.deepseek.com")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", None)
    if deepseek_key is None:
        print(
            "Your DEEPSEEK_API_KEY is not set. Please set it in the environment variable."
        )
    error_wait_time: float = 20
    is_debugging: bool = False
    log_dir: str = None


@dataclasses.dataclass
class DeepseekV3ConfigClass:  # Added Deepseek configuration
    model: str = "deepseek-chat"
    api_base: str = os.getenv("DEEPSEEK_BASEURL", "https://api.deepseek.com")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", None)
    if deepseek_key is None:
        print(
            "Your DEEPSEEK_API_KEY is not set. Please set it in the environment variable."
        )
    error_wait_time: float = 20
    is_debugging: bool = False
    log_dir: str = None


def dynamic_import(module_name, log_dir, use_langfuse_logging=False) -> object:
    if module_name in module_mapping:
        module_config_name = module_mapping[module_name]["config_name"]
        module_import_name = module_mapping[module_name]["module_name"]
        class_name = module_mapping[module_name]["class_name"]
        module_config = getattr(sys.modules[__name__], module_config_name)
        module_config.log_dir = log_dir

        # import the module
        LLM_module = importlib.import_module(
            "pentestgpt.utils.APIs." + module_import_name
        )
        LLM_class = getattr(LLM_module, class_name)
        # initialize the class
        LLM_class_initialized = LLM_class(
            module_config, use_langfuse_logging=use_langfuse_logging
        )

        return LLM_class_initialized

    else:
        print(
            "Module not found: "
            + module_name
            + ". Falling back to use the default gpt-4o"
        )
        # fall back to gpt-3.5-turbo-16k
        LLM_class_initialized = dynamic_import("gpt-4o", log_dir)
        return LLM_class_initialized


if __name__ == "__main__":
    # Quick test for each model
    import time

    # Define test models based on the revised list
    test_models = [
        # "gpt-4o",
        # "o3",
        # "o4-mini",
        # "gemini-2.5-flash",
        # "gemini-2.5-pro",
        "deepseek-r1",
        "deepseek-v3",
    ]

    # Test message
    test_message = "Explain briefly what makes you unique as an AI model."

    print("=== Starting Model Tests ===")

    for model_name in test_models:
        try:
            print(f"\n\nTesting model: {model_name}")
            print("=" * 50)

            # Initialize the model
            llm = dynamic_import(model_name, "logs")

            # Print model information
            print(f"Model initialized: {llm.__class__.__name__}")

            # Send a message and get response
            print(f"\nSending test message: '{test_message}'")
            print("-" * 50)

            # Measure response time
            start_time = time.time()
            response = llm.send_new_message(test_message)
            end_time = time.time()

            # Print response and timing
            print(f"Response received in {end_time - start_time:.2f} seconds:")
            print("-" * 50)
            print(response)
            print("-" * 50)

            # Add a small delay between API calls
            time.sleep(2)

        except Exception as e:
            print(f"Error testing {model_name}: {str(e)}")

    print("\n=== Model Tests Complete ===")
