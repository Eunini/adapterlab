from dataclasses import replace

import pytest
from adapterlab.api import create_app
from adapterlab.config import Settings
from fastapi.testclient import TestClient


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}", artifacts_dir=tmp_path / "artifacts"
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client


@pytest.fixture(scope="session")
def tiny_model(tmp_path_factory):
    """An offline random-weight fixture; never presented as a pretrained benchmark."""
    import torch
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors
    from transformers import BertConfig, BertForPreTraining, PreTrainedTokenizerFast

    path = tmp_path_factory.mktemp("tiny-model")
    words = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "i",
        "my",
        "the",
        "please",
        "account",
        "invoice",
        "cancel",
        "order",
        "app",
        "login",
        "payment",
        "not",
        "is",
        "a",
        "for",
        "to",
        "and",
        "with",
        "on",
        "need",
        "change",
        "help",
        "error",
        "delivery",
        "phone",
        "billing",
        "technical",
    ]
    backend = Tokenizer(
        models.WordPiece({word: index for index, word in enumerate(words)}, unk_token="[UNK]")
    )
    backend.normalizer = normalizers.BertNormalizer(lowercase=True)
    backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    backend.post_processor = processors.BertProcessing(("[SEP]", 3), ("[CLS]", 2))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
    )
    tokenizer.save_pretrained(path)
    torch.manual_seed(7)
    model = BertForPreTraining(
        BertConfig(
            vocab_size=len(words),
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=48,
            max_position_embeddings=256,
        )
    )
    model.save_pretrained(path, safe_serialization=True)
    return str(path)


@pytest.fixture
def training_settings(settings, tiny_model):
    return replace(settings, model_path=tiny_model, model_revision=None)
