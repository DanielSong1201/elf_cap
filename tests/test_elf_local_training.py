from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ELF_ROOT = REPO_ROOT / "elf"
sys.path.insert(0, str(ELF_ROOT))

try:
    import torch
    from configs.config import load_config_from_yaml
    from modules.model import ELF
    from train_step import train_step
    from utils.checkpoint_utils import load_model_weights
    from utils.train_utils import TrainState
except ImportError:
    torch = None
    load_config_from_yaml = None
    load_model_weights = None


@unittest.skipIf(torch is None, "PyTorch ELF dependencies are not installed")
class ElfLocalTrainingTest(unittest.TestCase):
    def test_overfit_config_uses_weight_only_initialization(self) -> None:
        config = load_config_from_yaml(
            str(REPO_ROOT / "configs" / "train_caption_overfit_100_ELF-B.yml")
        )
        self.assertEqual(config.data_path, "outputs/experiments/elf_caption_overfit_100/data/train")
        self.assertEqual(config.max_length, 48)
        self.assertEqual(config.init_checkpoint, "embedded-language-flows/ELF-B-owt-torch")
        self.assertIsNone(config.resume)
        self.assertEqual(config.global_batch_size, 10)

    def test_load_model_weights_does_not_require_training_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = torch.nn.Linear(3, 2)
            target = torch.nn.Linear(3, 2)
            with torch.no_grad():
                source.weight.fill_(2.0)
                source.bias.fill_(3.0)
                target.weight.zero_()
                target.bias.zero_()
            checkpoint = Path(temporary) / "checkpoint_99"
            torch.save(
                {
                    "params": source.state_dict(),
                    "opt_state": {},
                    "step": 99,
                    "epoch": 7,
                },
                checkpoint,
            )
            source_step = load_model_weights(str(checkpoint), target)
            self.assertEqual(source_step, 99)
            self.assertTrue(torch.equal(target.weight, source.weight))
            self.assertTrue(torch.equal(target.bias, source.bias))

    def test_tiny_elf_train_step_is_finite_and_updates_parameters(self) -> None:
        class FakeEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = torch.nn.Embedding(32, 8)

            def forward(self, input_ids, attention_mask=None, deterministic=True):
                del attention_mask, deterministic
                return self.embedding(input_ids)

        model = ELF(
            text_encoder_dim=8,
            max_length=6,
            hidden_size=16,
            depth=2,
            num_heads=2,
            bottleneck_dim=4,
            num_time_tokens=1,
            num_self_cond_cfg_tokens=0,
            num_model_mode_tokens=1,
            vocab_size=32,
        )
        encoder = FakeEncoder().eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        state = TrainState(
            model=model,
            optimizer=optimizer,
            ema_params1=TrainState.init_ema(model),
            dropout_generator=torch.Generator().manual_seed(42),
        )
        config = load_config_from_yaml(None)
        config.use_bf16 = False
        config.latent_mean = 0.0
        config.latent_std = 1.0
        config.t_eps = 0.05
        config.self_cond_prob = 0.0
        config.num_self_cond_cfg_tokens = 0
        config.decoder_prob = 0.5
        config.decoder_noise_scale = 1.0
        config.decoder_p_mean = 0.8
        config.decoder_p_std = 0.8
        config.denoiser_p_mean = 0.8
        config.denoiser_p_std = 0.8
        config.denoiser_noise_scale = 1.0
        config.time_schedule = "logit_normal"
        config.label_drop_prob = 0.0
        config.pad_token = "pad"
        config.grad_accum_steps = 1
        config.ema_decay1 = 0.99

        input_ids = torch.randint(0, 32, (2, 6))
        batch = {
            "input_ids": input_ids,
            "encoder_attention_mask": torch.ones(2, 6, 6),
            "attention_mask": torch.ones(2, 6),
            "cond_seq_mask": torch.zeros(2, 6),
            "label_drop_mask": torch.zeros(2, dtype=torch.bool),
        }
        before = model.proj_kernel.detach().clone()
        _, metrics = train_step(state, encoder, batch, config)
        self.assertTrue(torch.isfinite(metrics["loss"]))
        self.assertFalse(torch.equal(before, model.proj_kernel.detach()))


if __name__ == "__main__":
    unittest.main()
