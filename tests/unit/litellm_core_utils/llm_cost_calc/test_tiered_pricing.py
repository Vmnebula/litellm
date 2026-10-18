import pytest

from litellm.litellm_core_utils.llm_cost_calc.tiered_pricing import select_tier_for_input


@pytest.mark.parametrize(
    "malformed_tier",
    [{"range": []}, {"range": None}, {"range": [32_000]}, {"input_cost_per_token": 1e-6}],
)
def test_select_tier_for_input_skips_tier_with_malformed_range(malformed_tier):
    priced_tier = {"range": [0, 32_000], "input_cost_per_token": 1e-6}

    assert select_tier_for_input([malformed_tier, priced_tier], input_tokens=100) is priced_tier
