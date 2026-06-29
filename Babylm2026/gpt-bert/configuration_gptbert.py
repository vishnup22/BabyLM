from transformers import PretrainedConfig


class GptBertConfig(PretrainedConfig):
    model_type = "gptbert"

    def __init__(
        self,
        vocab_size=16384,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=2560,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        position_bucket_size=32,
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.position_bucket_size = position_bucket_size
        self.layer_norm_eps = layer_norm_eps
