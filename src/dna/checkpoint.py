import torch

from src.dna.data import DnaTokenizer
from src.models.s4_model import S4SequenceModel
from src.models.transformer_model import TransformerSequenceModel


def tokenizer_from_checkpoint(checkpoint: dict) -> DnaTokenizer:
    vocab = checkpoint.get("tokenizer_vocab")
    if vocab is None:
        return DnaTokenizer(include_n=True)
    return DnaTokenizer(vocab=vocab)


def build_model_from_config(config: dict, tokenizer: DnaTokenizer) -> torch.nn.Module:
    model_type = config.get("model_type", "s4d")
    if model_type == "transformer":
        return TransformerSequenceModel(
            vocab_size=tokenizer.vocab_size,
            d_model=config.get("d_model", 384),
            n_heads=config.get("n_heads", 6),
            n_layers=config.get("n_layers", 9),
            ffn_dim=config.get("ffn_dim", 1600),
            dropout=config.get("dropout", 0.1),
            pad_token_id=tokenizer.pad_token_id,
            mask_token_id=tokenizer.unk_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_length=config.get("max_length", config.get("l_max", 1024)),
        )
    if model_type != "s4d":
        raise ValueError(f"unsupported checkpoint model_type: {model_type}")
    return S4SequenceModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config.get("d_model", 448),
        d_state=config.get("d_state", 64),
        n_layers=config.get("n_layers", 10),
        dropout=config.get("dropout", 0.1),
        kernel_type=config.get("kernel_type", "diag"),
        bidirectional=False,
        model_variant=config.get("model_variant", "legacy"),
        l_max=config.get("l_max"),
        pad_token_id=tokenizer.pad_token_id,
        mask_token_id=tokenizer.unk_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_length=config.get("l_max", 1024),
    )


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint: str):
    device = get_device()
    ckpt = torch.load(checkpoint, map_location=device)
    tokenizer = tokenizer_from_checkpoint(ckpt)
    config = dict(ckpt.get("model_config", {}))
    config.setdefault("model_type", ckpt.get("model_type", "s4d"))
    model = build_model_from_config(config, tokenizer).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, tokenizer, device
