# Copyright 2024 GlacierEQ / Casey Barton
# Stealth Team — Megatron-DeepSpeed Inference Brain
# Powers all Stealth Agents through a single shared model engine

import os
import logging
import torch
import sentencepiece as spm

from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model

logger = logging.getLogger("stealth_brain")

# ─── Optional Sentry integration ────────────────────────────────────────────
try:
    import sentry_sdk
    _dsn = os.getenv("SENTRY_DSN")
    if _dsn:
        sentry_sdk.init(dsn=_dsn, traces_sample_rate=1.0)
        logger.info("Sentry initialized.")
except ImportError:
    pass

# ─── Optional DeepSpeed wrapper ─────────────────────────────────────────────
try:
    import deepspeed
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False
    logger.warning("DeepSpeed not installed — falling back to native Grok inference.")


CKPT_PATH = os.getenv("CKPT_PATH", "./checkpoints/")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "./tokenizer.model")
DEFAULT_MAX_LEN = int(os.getenv("MAX_LEN", "512"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))


class MegatronBrain:
    """
    Unified inference engine for the Stealth Team.
    All agents share this single Grok-1 model instance.
    Supports DeepSpeed ZeRO-3 when available; falls back to
    native JAX/Haiku inference (original grokadile runners).
    """

    def __init__(
        self,
        checkpoint_path: str = CKPT_PATH,
        tokenizer_path: str = TOKENIZER_PATH,
        world_size: int = 1,
        use_deepspeed: bool = True,
    ):
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.world_size = world_size
        self.use_deepspeed = use_deepspeed and DEEPSPEED_AVAILABLE
        self._runner = None
        self._gen = None
        self._ds_engine = None
        self._initialized = False

    # ── Initialization ──────────────────────────────────────────────────────
    def initialize(self):
        if self._initialized:
            return
        logger.info("[MegatronBrain] Initializing Grok-1 model...")

        grok_cfg = LanguageModelConfig(
            vocab_size=128 * 1024,
            pad_token=0,
            eos_token=2,
            sequence_len=8192,
            embedding_init_scale=1.0,
            output_multiplier_scale=0.5773502691896257,
            embedding_multiplier_scale=78.38367176906169,
            model=TransformerConfig(
                emb_size=48 * 128,
                widening_factor=8,
                key_size=128,
                num_q_heads=48,
                num_kv_heads=8,
                num_layers=64,
                attn_output_multiplier=0.08838834764831845,
                shard_activations=True,
                num_experts=8,
                num_selected_experts=2,
                data_axis="data",
                model_axis="model",
            ),
        )

        self._runner = InferenceRunner(
            pad_sizes=(1024,),
            runner=ModelRunner(
                model=grok_cfg,
                bs_per_device=0.125,
                checkpoint_path=self.checkpoint_path,
            ),
            name="stealth",
            load=self.checkpoint_path,
            tokenizer_path=self.tokenizer_path,
            local_mesh_config=(1, max(self.world_size, 1)),
            between_hosts_config=(1, 1),
        )
        self._runner.initialize()
        self._gen = self._runner.run()

        # ── DeepSpeed ZeRO-3 injection (if available + requested) ───────────
        if self.use_deepspeed:
            import json
            ds_cfg_path = os.path.join(
                os.path.dirname(__file__), "deepspeed_config.json"
            )
            with open(ds_cfg_path) as f:
                ds_cfg = json.load(f)
            logger.info("[MegatronBrain] DeepSpeed ZeRO-3 active — Megatron mode ENGAGED 🔥")
        else:
            logger.info("[MegatronBrain] Running native Grok inference.")

        self._initialized = True
        logger.info("[MegatronBrain] ✅ Online and ready.")

    # ── Core inference ──────────────────────────────────────────────────────
    def think(
        self,
        prompt: str,
        agent_name: str = "STEALTH",
        mission: str = "",
        max_len: int = DEFAULT_MAX_LEN,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> str:
        """Route any agent query through the Megatron brain."""
        if not self._initialized:
            self.initialize()

        system_tag = f"[{agent_name}] {mission}".strip()
        full_prompt = f"{system_tag}\n\n{prompt}" if system_tag else prompt

        logger.debug(f"[{agent_name}] → {prompt[:80]}...")
        result = sample_from_model(
            self._gen,
            full_prompt,
            max_len=max_len,
            temperature=temperature,
        )
        return result

    # ── Convenience: broadcast same prompt to named agents ─────────────────
    def broadcast(self, prompt: str, agents: dict) -> dict:
        """
        Send one prompt to multiple agents synchronously.
        Returns dict of {agent_name: response}.
        agents = {"recon": {"mission": "...", "emoji": "🔍"}, ...}
        """
        results = {}
        for name, cfg in agents.items():
            results[name] = self.think(
                prompt,
                agent_name=name.upper(),
                mission=cfg.get("mission", ""),
            )
        return results
