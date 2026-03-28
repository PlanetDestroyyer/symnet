"""Tests for SymNetModel."""

import pytest
import torch
from symnet.models.symnet_model import SymNetModel, COMM_DIM, N_ACTIONS, HIDDEN_DIM


@pytest.fixture
def model():
    """Small model for fast CPU tests."""
    return SymNetModel(hidden_dim=64, comm_dim=32, n_mamba_layers=1)


@pytest.fixture
def full_model():
    """Spec-accurate model — used for shape validation only."""
    return SymNetModel()  # defaults: hidden=512, comm=128


# ── Shape tests ────────────────────────────────────────────────────────────

class TestModelShapes:
    def test_action_logits_shape(self, model):
        obs     = torch.zeros(1, 75)
        comm_in = torch.zeros(1, model.comm_dim)
        logits, comm_out, value = model(obs, comm_in)
        assert logits.shape == (1, model.n_actions), \
            f"Expected (1, {model.n_actions}), got {logits.shape}"

    def test_comm_out_shape(self, model):
        obs     = torch.zeros(1, 75)
        comm_in = torch.zeros(1, model.comm_dim)
        _, comm_out, _ = model(obs, comm_in)
        assert comm_out.shape == (1, model.comm_dim), \
            f"Expected (1, {model.comm_dim}), got {comm_out.shape}"

    def test_value_shape(self, model):
        obs     = torch.zeros(1, 75)
        comm_in = torch.zeros(1, model.comm_dim)
        _, _, value = model(obs, comm_in)
        assert value.shape == (1, 1), f"Expected (1, 1), got {value.shape}"

    def test_batch_forward(self, model):
        B       = 8
        obs     = torch.zeros(B, 75)
        comm_in = torch.zeros(B, model.comm_dim)
        logits, comm_out, value = model(obs, comm_in)
        assert logits.shape   == (B, model.n_actions)
        assert comm_out.shape == (B, model.comm_dim)
        assert value.shape    == (B, 1)

    def test_full_model_spec_shapes(self, full_model):
        """Verify spec-exact dimensions."""
        obs     = torch.zeros(1, 75)
        comm_in = torch.zeros(1, COMM_DIM)
        logits, comm_out, value = full_model(obs, comm_in)
        assert logits.shape   == (1, N_ACTIONS)
        assert comm_out.shape == (1, COMM_DIM)


# ── Comm vector constraints ────────────────────────────────────────────────

class TestCommVector:
    def test_comm_out_bounded(self, model):
        """tanh output must be in (-1, 1)."""
        obs     = torch.randn(4, 75)
        comm_in = torch.randn(4, model.comm_dim)
        _, comm_out, _ = model(obs, comm_in)
        assert comm_out.min() > -1.0 - 1e-6
        assert comm_out.max() <  1.0 + 1e-6

    def test_zero_comm_shape(self, model):
        comm = model.zero_comm(batch_size=4)
        assert comm.shape   == (4, model.comm_dim)
        assert (comm == 0).all()


# ── Weight sharing ─────────────────────────────────────────────────────────

class TestWeightSharing:
    def test_two_passes_use_same_weights(self, model):
        """Both agent passes should use exactly the same parameter tensors."""
        obs_a   = torch.randn(1, 75)
        obs_b   = torch.randn(1, 75)
        comm_a  = torch.zeros(1, model.comm_dim)
        comm_b  = torch.zeros(1, model.comm_dim)

        # Collect parameter data pointers before and after both passes
        params_before = [p.data_ptr() for p in model.parameters()]
        _ = model(obs_a, comm_b)
        _ = model(obs_b, comm_a)
        params_after  = [p.data_ptr() for p in model.parameters()]

        assert params_before == params_after, \
            "Parameter memory addresses changed — weights are not shared!"

    def test_different_inputs_produce_different_outputs(self, model):
        """Different observations should produce different outputs (not identical agents)."""
        obs_a   = torch.randn(1, 75)
        obs_b   = torch.randn(1, 75)  # different
        comm    = torch.zeros(1, model.comm_dim)

        logits_a, _, _ = model(obs_a, comm)
        logits_b, _, _ = model(obs_b, comm)

        assert not torch.allclose(logits_a, logits_b), \
            "Different observations should produce different logits"

    def test_same_inputs_produce_same_outputs(self, model):
        """Same obs + comm should always produce same output (deterministic in eval)."""
        model.eval()
        obs     = torch.randn(1, 75)
        comm    = torch.randn(1, model.comm_dim)

        out1 = model(obs, comm)
        out2 = model(obs, comm)

        torch.testing.assert_close(out1[0], out2[0])
        torch.testing.assert_close(out1[1], out2[1])
        torch.testing.assert_close(out1[2], out2[2])


# ── Gradient flow ──────────────────────────────────────────────────────────

class TestGradientFlow:
    def test_gradients_accumulate_from_both_agents(self, model):
        """
        Core SymNet property: loss_A + loss_B in a single backward().
        All parameters should receive nonzero gradients.
        """
        obs_a   = torch.randn(2, 75)
        obs_b   = torch.randn(2, 75)
        comm_a  = torch.zeros(2, model.comm_dim)
        comm_b  = torch.zeros(2, model.comm_dim)

        logits_a, comm_a_out, val_a = model(obs_a, comm_b)
        logits_b, comm_b_out, val_b = model(obs_b, comm_a)

        # Fake losses
        loss_a = logits_a.mean() + val_a.mean() + comm_a_out.mean()
        loss_b = logits_b.mean() + val_b.mean() + comm_b_out.mean()
        total  = loss_a + loss_b
        total.backward()

        # Every parameter should have a gradient
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for parameter '{name}'"

    def test_action_logits_are_differentiable(self, model):
        obs     = torch.randn(1, 75, requires_grad=False)
        comm_in = torch.randn(1, model.comm_dim)
        logits, _, _ = model(obs, comm_in)
        loss = logits.sum()
        loss.backward()   # should not raise
