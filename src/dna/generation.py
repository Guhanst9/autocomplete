import torch

from src.dna.prediction import TripletCodec, normalize_prediction_unit


@torch.no_grad()
def generate_bases(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    max_new_bases: int,
    sampling_temperature: float | None = None,
) -> torch.Tensor:
    prediction_unit = normalize_prediction_unit(getattr(model, "prediction_unit", "base"))
    if prediction_unit == "triplet":
        codec = TripletCodec(getattr(model, "output_tokens", None))
        return model.generate_triplets(
            prompt_ids,
            codec.base_ids(tokenizer, prompt_ids.device),
            max_new_bases,
            sampling_temperature=sampling_temperature,
        )

    forbidden = {
        tokenizer.pad_token_id,
        tokenizer.unk_token_id,
        tokenizer.eos_token_id,
    }
    if "N" in tokenizer.vocab:
        forbidden.add(tokenizer.vocab["N"])
    return model.generate(
        prompt_ids,
        max_new_tokens=max_new_bases,
        use_recurrent=True,
        stop_at_eos=False,
        forbidden_token_ids=tuple(sorted(forbidden)),
        min_new_tokens=max_new_bases,
        sampling_temperature=sampling_temperature,
    )
