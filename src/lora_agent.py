"""
Simple LoRA-backed LLM loader and generator.

This module attempts to load a HuggingFace-compatible base model and then apply
PEFT LoRA adapters if available. It exposes `generate_response` which accepts a
prompt and returns text output. It's intentionally lightweight and falls back
to a simple template rewriter if large LLMs are not available.
"""
from typing import Optional
import os
import logging

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from peft import PeftModel, PeftConfig, get_peft_model, prepare_model_for_kbit_training
    import torch
    _HAS_TF = True
except Exception:
    _HAS_TF = False

LOGGER = logging.getLogger(__name__)


class LoRAAgent:
    def __init__(self, base_model_name: str = None, adapter_dir: str = None, device: Optional[str] = None):
        self.base_model_name = base_model_name
        self.adapter_dir = adapter_dir
        self.device = device or ("cuda" if (
            torch and torch.cuda.is_available()) else "cpu")
        self.tokenizer = None
        self.model = None
        self._loaded = False

    def load(self):
        if not _HAS_TF:
            LOGGER.warning(
                "Transformers/PEFT not available; LoRAAgent will use fallback behavior.")
            return

        if self._loaded:
            return

        if not self.base_model_name:
            raise ValueError(
                "base_model_name must be provided to load a model")

        # Load tokenizer and base model
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

        # prepare for LoRA
        try:
            model = prepare_model_for_kbit_training(model)
        except Exception:
            pass

        # apply peft adapter if available
        if self.adapter_dir and os.path.exists(self.adapter_dir):
            try:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, self.adapter_dir)
                LOGGER.info("Loaded LoRA adapter from %s", self.adapter_dir)
            except Exception as e:
                LOGGER.warning("Failed to load LoRA adapter: %s", e)

        self.model = model.to(self.device)
        self._loaded = True

    def generate(self, prompt: str, max_length: int = 256, temperature: float = 0.7):
        """Generate a response using the loaded model. If not loaded, return a simple fallback."""
        if not _HAS_TF or self.model is None or self.tokenizer is None:
            # Fallback: do a light template rewrite
            return self._fallback(prompt)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_length, do_sample=True, temperature=temperature)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # strip prompt prefix if model echoes
        if text.startswith(prompt):
            return text[len(prompt):].strip()
        return text.strip()

    def _fallback(self, prompt: str) -> str:
        # Very simple summarization-style fallback when model is not available.
        lines = [l.strip() for l in prompt.splitlines() if l.strip()]
        # Return last 2 lines or a join of notable bits.
        if len(lines) >= 2:
            return " ".join(lines[-2:])
        return lines[-1] if lines else ""


def make_default_agent(adapter_dir: str = None):
    # pick a small default conversational model; prefer a causal model available in environment
    default_base = os.environ.get("PROFSEEK_BASE_LLM", "gpt2")
    agent = LoRAAgent(base_model_name=default_base, adapter_dir=adapter_dir)
    try:
        agent.load()
    except Exception:
        LOGGER.exception("Failed to load agent model")
    return agent


__all__ = ["LoRAAgent", "make_default_agent"]
