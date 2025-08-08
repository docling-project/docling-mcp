import logging

from smolagents import (
    TransformersModel,
    LiteLLMModel,
)
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field, validator
from smolagents.models import MessageRole, ChatMessage, Model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Configuration for the LLM model."""

    type: Literal["transformers", "ollama"] = Field(
        default="transformers", description="Type of model to use"
    )
    model_id: str = Field(
        default="HuggingFaceTB/SmolLM2-1.7B-Instruct", description="Model identifier"
    )
    device: str = Field(default="cuda", description="Device to run on (cuda/cpu)")
    torch_dtype: str = Field(default="auto", description="Torch dtype for model")
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Base URL for Ollama server"
    )


def setup_local_model(config: ModelConfig) -> Model:
    """Setup local model based on configuration."""
    logger.info(f"Setting up {config} model")

    if config.type == "ollama":
        logger.info(f"Connecting to Ollama at {config.ollama_base_url}")
        # return OllamaModel(
        return LiteLLMModel(
            model_id=config.model_id,
            base_url=config.ollama_base_url,
            num_ctx=4 * 8192,  # ollama default is 2048 which will often fail horribly.
        )
    else:
        logger.info(f"Loading transformers model: {config.model_id}")
        return TransformersModel(
            model_id=config.model_id,
            device=config.device,
            torch_dtype=config.torch_dtype,
            trust_remote_code=True,
            max_new_tokens=4 * 8192,
        )


def test_ollama_smollm2():
    """Main function to run the demonstrations."""

    model_config = ModelConfig(
        type="ollama", model_id="ollama/smollm2", device="cpu", torch_dtype="auto"
    )
    model = setup_local_model(config=model_config)

    chat_messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=[{"type": "text", "text": "You are a helpful assistant"}],
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=[{"type": "text", "text": "What is a polymer?"}],
        ),
    ]

    output = model.generate(messages=chat_messages)
    print(output)


def test_ollama_gptoss():
    """Main function to run the demonstrations."""

    model_config = ModelConfig(
        type="ollama",
        model_id="ollama/gpt-oss:20b",  # , device="cpu", torch_dtype="auto"
    )
    model = setup_local_model(config=model_config)

    chat_messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=[{"type": "text", "text": "You are a helpful assistant"}],
        ),
        ChatMessage(
            role=MessageRole.USER,
            content=[{"type": "text", "text": "What is a polymer?"}],
        ),
    ]

    output = model.generate(messages=chat_messages)
    print(output)


def main():
    """Main function to run the demonstrations."""

    test_ollama_smollm2()

    test_ollama_gptoss()


if __name__ == "__main__":
    main()
