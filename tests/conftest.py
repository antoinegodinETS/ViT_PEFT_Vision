import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# Trainable parameters without the classification head, at the ~0.3M budget.
EXPECTED_TRAINABLE_NO_HEAD = {
    'lora_r8': 294_912,
    'adaptformer_d16': 304_320,
    'vpt_deep_p32': 294_912,
    'vpt_shallow_p10': 7_680,
    'full': 85_798_656,
}
