from dataclasses import dataclass
import sys
from typing import Optional, Dict, List
import argparse
from loguru import logger
from pentestgpt.test_connection import test_connection
from pentestgpt.utils.pentest_gpt import pentestGPT
from pentestgpt.utils.APIs.module_import import module_mapping


@dataclass
class PentestConfig:
    log_dir: str
    reasoning_model: str
    parsing_model: str
    use_logging: bool
    use_api: bool
    show_models: bool
    base_url: Optional[str] = None


class PentestGPTCLI:
    DEFAULT_CONFIG = {
        "log_dir": "logs",
        "reasoning_model": "gpt-4o",  # Updated to gpt-4o as default
        "parsing_model": "gpt-4o",  # Updated to gpt-4o as default
    }

    # Updated with the latest model options
    VALID_MODELS = list(module_mapping.keys())

    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="PentestGPT")

        # Use --logDir instead of --log_dir to match your current interface
        parser.add_argument(
            "--logDir",
            type=str,
            default=self.DEFAULT_CONFIG["log_dir"],
            help="Path to the log directory for storing conversations",
        )

        # Adding baseUrl as it appears in your current interface
        parser.add_argument(
            "--baseUrl", type=str, default=None, help="Base URL for API requests"
        )

        # Adding models command to show available models
        parser.add_argument(
            "--models", action="store_true", help="List all available models"
        )

        # Add model selection arguments
        parser.add_argument(
            "--reasoning",
            type=str,
            default=self.DEFAULT_CONFIG["reasoning_model"],
            choices=self.VALID_MODELS,
            help="Model for higher-level cognitive tasks",
        )

        parser.add_argument(
            "--parsing",
            type=str,
            default=self.DEFAULT_CONFIG["parsing_model"],
            choices=self.VALID_MODELS,
            help="Model for structural and grammatical language processing",
        )

        # Main arguments
        parser.add_argument(
            "--logging",
            action="store_true",
            default=False,
            help="Enable data collection through langfuse logging",
        )

        parser.add_argument(
            "--useAPI",
            action="store_true",
            default=True,
            help="Deprecated: Set to False only for testing with cookie",
        )

        return parser

    def parse_args(self) -> PentestConfig:
        args = self.parser.parse_args()

        return PentestConfig(
            log_dir=args.logDir,
            reasoning_model=args.reasoning,
            parsing_model=args.parsing,
            use_logging=args.logging,
            use_api=args.useAPI,
            show_models=args.models if hasattr(args, "models") else False,
            base_url=args.baseUrl,
        )

    def display_available_models(self) -> None:
        """Display all available models with their details"""
        print("\n=== AVAILABLE MODELS ===")
        print("\nOpenAI Models:")
        openai_models = [
            m
            for m in self.VALID_MODELS
            if any(prefix in m for prefix in ["gpt", "o3", "o4"])
        ]
        for model in openai_models:
            print(f"  • {model}")

        print("\nGemini Models:")
        gemini_models = [m for m in self.VALID_MODELS if "gemini" in m]
        for model in gemini_models:
            print(f"  • {model}")

        print("\nDeepseek Models:")
        deepseek_models = [m for m in self.VALID_MODELS if "deepseek" in m]
        for model in deepseek_models:
            print(f"  • {model}")

        print("\nOther Models:")
        other_models = [
            m
            for m in self.VALID_MODELS
            if m not in openai_models + gemini_models + deepseek_models
        ]
        for model in other_models:
            print(f"  • {model}")

        print("\nDefault Configuration:")
        print(f"  • Reasoning Model: {self.DEFAULT_CONFIG['reasoning_model']}")
        print(f"  • Parsing Model: {self.DEFAULT_CONFIG['parsing_model']}")
        print("\nUsage examples:")
        print("  • List models: pentestgpt --models")
        print("  • Use specific models: pentestgpt --reasoning o3 --parsing gpt-4o")
        print("")


def run_pentest(config: PentestConfig) -> None:
    try:
        # Pass base_url if provided
        kwargs = {
            "reasoning_model": config.reasoning_model,
            "parsing_model": config.parsing_model,
            "useAPI": config.use_api,
            "log_dir": config.log_dir,
            "use_langfuse_logging": config.use_logging,
        }

        # Only add base_url if it's provided
        if config.base_url:
            kwargs["base_url"] = config.base_url

        pentest_handler = pentestGPT(**kwargs)
        pentest_handler.main()
    except Exception as e:
        logger.error(f"PentestGPT execution failed: {e}")
        sys.exit(1)


def main():
    cli = PentestGPTCLI()
    config = cli.parse_args()

    # Handle show-models flag
    if config.show_models:
        cli.display_available_models()
        sys.exit(0)

    run_pentest(config)


if __name__ == "__main__":
    main()
