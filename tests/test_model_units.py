"""Model-level checks that need no pretrained weights (random init)."""
import pytest
import torch
import torch.nn.functional as F

import main
from conftest import EXPECTED_TRAINABLE_NO_HEAD
from experiment import build_model
from model.vision_transformer import VisionTransformerPETL

METHOD_ARGS = {
    'lora_r8': ['--lora_bottleneck', '8'],
    'adaptformer_d16': ['--ft_mlp_module', 'adapter', '--ft_mlp_mode', 'parallel', '--ft_mlp_ln', 'before',
                        '--adapter_bottleneck', '16', '--adapter_init', 'lora_kaiming', '--adapter_scaler', '0.1'],
    'vpt_deep_p32': ['--vpt_mode', 'deep', '--vpt_num', '32'],
    'vpt_shallow_p10': ['--vpt_mode', 'shallow', '--vpt_num', '10'],
    'full': ['--full'],
}
NUM_PATCH_TOKENS = 197  # 14 x 14 patches + CLS for ViT-B/16 at 224 px


@pytest.fixture(autouse=True)
def no_pretrained_weights(monkeypatch):
    """Skip loading the npz: parameter counts and wiring do not depend on the weights."""
    monkeypatch.setattr(VisionTransformerPETL, 'load_pretrained', lambda self, *a, **k: None)


def build(method_args, class_num=10):
    params = main.setup_parser().parse_args(method_args)
    params.class_num = class_num
    params.debug = False
    model, _, no_head = build_model.get_model(params)
    return model, no_head, params


@pytest.mark.parametrize('method', list(METHOD_ARGS))
def test_trainable_parameter_counts(method):
    model, no_head, _ = build(METHOD_ARGS[method])
    assert no_head == EXPECTED_TRAINABLE_NO_HEAD[method]
    with_head = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert with_head == no_head + 768 * 10 + 10


def test_lora_adapts_q_and_v_but_not_k(monkeypatch):
    model, _, _ = build(METHOD_ARGS['lora_r8'])
    model.eval()
    attn = model.blocks[0].attn
    lora = attn.lora
    names = sorted(n for n, _ in lora.named_parameters())
    assert names == ['lora_a_q.weight', 'lora_a_v.weight', 'lora_b_q.weight', 'lora_b_v.weight']
    # B starts at zero; make it non-zero so a delta on k would show.
    torch.manual_seed(0)
    for p in (lora.lora_b_q.weight, lora.lora_b_v.weight):
        torch.nn.init.normal_(p)

    captured = {}

    def capture_sdpa(q, k, v, **kwargs):
        captured.update(q=q, k=k, v=v)
        return torch.zeros_like(q)

    monkeypatch.setattr(F, 'scaled_dot_product_attention', capture_sdpa)
    attn.fused_attn = True
    x = torch.randn(2, 5, 768)
    with torch.no_grad():
        attn(x, 0)
        q_ref, k_ref, v_ref = attn.qkv(x).reshape(2, 5, 3, 12, 64).permute(2, 0, 3, 1, 4).unbind(0)
    assert torch.equal(captured['k'], k_ref), 'LoRA must leave W_k untouched'
    assert not torch.allclose(captured['q'], q_ref)
    assert not torch.allclose(captured['v'], v_ref)


def _last_block_input_len(model):
    seen = {}
    handle = model.blocks[-1].register_forward_pre_hook(lambda mod, args: seen.update(n=args[0].shape[1]))
    with torch.no_grad():
        out = model.forward_features(torch.randn(1, 3, 224, 224))
    handle.remove()
    return seen['n'], out.shape[1]


def test_vpt_shallow_prompts_reach_last_block():
    model, _, _ = build(METHOD_ARGS['vpt_shallow_p10'])
    model.eval()
    assert model.vpt.prompt_embeddings.shape == (1, 10, 768)
    last_in, out_len = _last_block_input_len(model)
    assert last_in == NUM_PATCH_TOKENS + 10, 'shallow prompts must still be in the last block input'
    assert out_len == NUM_PATCH_TOKENS, 'prompts are removed after the last block'

    # The prompt tokens entering the last block depend on the learned prompts.
    seen = {}
    model.blocks[-1].register_forward_pre_hook(lambda mod, args: seen.setdefault('x', []).append(args[0]))
    img = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model.forward_features(img)
        model.vpt.prompt_embeddings.add_(1.0)
        model.forward_features(img)
    assert not torch.allclose(seen['x'][0][:, :10], seen['x'][1][:, :10])


def test_vpt_deep_unchanged():
    model, _, _ = build(METHOD_ARGS['vpt_deep_p32'])
    model.eval()
    assert model.vpt.prompt_embeddings.shape == (12, 32, 768)
    last_in, out_len = _last_block_input_len(model)
    # Fresh prompts are prepended at every layer and dropped after it.
    assert last_in == NUM_PATCH_TOKENS + 32
    assert out_len == NUM_PATCH_TOKENS


def test_lora_alpha_and_dropout_options():
    model, _, _ = build(METHOD_ARGS['lora_r8'] + ['--lora_alpha', '16', '--lora_dropout', '0.0'])
    lora = model.blocks[0].attn.lora
    assert lora.scaling == 2.0
    assert lora.lora_drop.p == 0.0
    default_lora = build(METHOD_ARGS['lora_r8'])[0].blocks[0].attn.lora
    assert default_lora.scaling == 1.0 and default_lora.lora_drop.p == 0.1


def test_normalized_defaults_to_false():
    assert main.setup_parser().parse_args([]).normalized is False
    assert main.setup_parser().parse_args(['--normalized', 'true']).normalized is True
