"""
GPT-2 model code used to load the fine-tuned checkpoint (best_model/model.pt).
Copied from default_final_project.ipynb; the pretrained-weight loader and training code are left out.
"""
import torch
from torch import nn, dtype
import torch.nn.functional as F
from einops import rearrange
from config import PretrainedConfig, GPT2Config
from utils import get_extended_attention_mask, get_parameter_dtype


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):
    """
    TODO-1: Compute scaled dot-product attention for GPT-2.

    Implementation hints:
    1. Compute raw attention scores using QK^T, and scale them by sqrt(d_k).
    2. Apply a causal mask (lower-triangular) to prevent attending to future tokens.
    3. Optionally add the external attention_mask (e.g., padding positions).
    4. Normalize the scores with softmax to obtain attention probabilities.
    5. Apply dropout on the probabilities.
    6. Use them to weight the values (V) and obtain the context vectors.
    7. Finally, merge all attention heads back into a single hidden representation.
    """
    ### YOUR CODE HERE
    # key/query/value: [bs, num_heads, seq_len, head_size]
    d_k = query.size(-1)
    seq_len = query.size(-2)

    # 1. Raw scores QK^T, scaled by sqrt(d_k) so the softmax doesn't saturate
    scores = torch.matmul(query, key.transpose(-1, -2)) / (d_k ** 0.5)   # [bs, h, seq_len, seq_len]

    # 2. Causal mask: position i may only look at positions <= i (lower triangle).
    #    Future positions get a huge negative score, so softmax gives them ~0 weight.
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=scores.device))
    scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)

    # 3. Padding mask from get_extended_attention_mask: [bs, 1, 1, seq_len],
    #    0 for real tokens and -10000 for padding, so it is simply added to the scores
    if attention_mask is not None:
        scores = scores + attention_mask

    # 4. Softmax over the key dimension -> attention probabilities
    attn_probs = F.softmax(scores, dim=-1)

    # 5. Dropout on the attention probabilities
    attn_probs = self.dropout(attn_probs)

    # 6. Weighted sum of the values
    context = torch.matmul(attn_probs, value)                            # [bs, h, seq_len, head_size]

    # 7. Merge heads back: [bs, h, seq_len, head_size] -> [bs, seq_len, h * head_size]
    context = rearrange(context, 'b h t d -> b t (h d)')
    return context


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention using the self.attention function.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    TODO-2: Residual connection with dense projection and dropout.
    
    Implementation hints:
    1. Project the 'output' through dense_layer.
    2. Apply dropout to prevent overfitting.
    3. Add the original 'input' (residual connection) to the processed output.
    """

    ### YOUR CODE HERE
    # Project the sublayer output, apply dropout, then add the residual input
    return input + dropout(dense_layer(output))


  def forward(self, hidden_states, attention_mask):
    """
    TODO-3: Forward pass of a GPT-2 layer.

    Implementation hints:
    ---- Self-Attention Block ----
    1. LayerNorm the input for stability using self.attention_layer_norm.
    2. Compute multi-head causal self-attention using self.self_attention.
    3. Apply residual connection using self.add (dense_layer=self.attention_dense, dropout=self.attention_dropout).

    ---- Feed Forward Block ----
    4. LayerNorm the hidden_states from attention block using self.out_layer_norm.
    5. Pass through a two-layer feed-forward network with activation:
       self.interm_dense -> self.interm_af -> self.out_dense
    6. Apply residual connection again using self.add (dense_layer=self.out_dense, dropout=self.out_dropout).
    """
    ### YOUR CODE HERE
    # ---- Self-Attention Block (pre-LayerNorm) ----
    # H' = H + Dropout(W_O * SelfAttn(LayerNorm(H)))
    normed = self.attention_layer_norm(hidden_states)
    attn_output = self.self_attention(normed, attention_mask)
    hidden_states = self.add(hidden_states, attn_output, self.attention_dense, self.attention_dropout)

    # ---- Feed Forward Block (pre-LayerNorm) ----
    # H_next = H' + Dropout(W_2 * GELU(W_1 * LayerNorm(H')))
    # (self.add applies W_2 = self.out_dense, so we only apply W_1 and GELU here)
    normed = self.out_layer_norm(hidden_states)
    interm = self.interm_af(self.interm_dense(normed))
    hidden_states = self.add(hidden_states, interm, self.out_dense, self.out_dropout)
    return hidden_states

class GPTPreTrainedModel(nn.Module):

  def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
    super().__init__()
    self.config = config
    self.name_or_path = config.name_or_path

  def init_weights(self):
    # Initialize weights
    self.apply(self._init_weights)

  def _init_weights(self, module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
      # Slightly different from the TF version which uses truncated_normal for initialization
      # cf https://github.com/pytorch/pytorch/pull/5617
      module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
    elif isinstance(module, nn.LayerNorm):
      module.bias.data.zero_()
      module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
      module.bias.data.zero_()

  @property
  def dtype(self) -> dtype:
    return get_parameter_dtype(self)

class GPT2Model(GPTPreTrainedModel):
  """
  The GPT model returns the final embeddings for each token in a sentence.

  The model consists of:
  1. Embedding layers (used in self.embed).
  2. A stack of n GPT layers (used in self.encode).
  3. A linear transformation layer for the [CLS] token (used in self.forward, as given).
  """

  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # Embedding layers.
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)

    # Register position_ids (1, len position emb) to buffer because it is a constant.
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # GPT-2 layers.
    self.gpt_layers = nn.ModuleList([GPT2Layer(config) for _ in range(config.num_hidden_layers)])

    # [CLS] token transformations.
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    # Final layer norm.
    self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    self.init_weights()

  def embed(self, input_ids):
    """
    TODO-4: Embedding layer of the GPT-2 model.

    Implementation hints:
    1. Use self.word_embedding to convert input_ids to embeddings.
    2. Generate position ids and convert to embeddings using self.pos_embedding.
    3. Sum token and position embeddings.
    4. Apply self.embed_dropout to the sum.
    """
    ### YOUR CODE HERE
    seq_len = input_ids.shape[1]

    # 1. Token embeddings: [bs, seq_len, hidden]
    inputs_embeds = self.word_embedding(input_ids)

    # 2. Position embeddings for positions 0..seq_len-1: [1, seq_len, hidden]
    #    (broadcast over the batch when added)
    pos_ids = self.position_ids[:, :seq_len]
    pos_embeds = self.pos_embedding(pos_ids)

    # 3 + 4. Sum and apply dropout
    return self.embed_dropout(inputs_embeds + pos_embeds)

  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # Get the extended attention mask for self-attention.
    # Returns extended_attention_mask of size [batch_size, 1, 1, seq_len].
    # Distinguishes between non-padding tokens (with a value of 0) and padding tokens
    # (with a value of a large negative number).
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # Pass the hidden states through the encoder layers.
    for i, layer_module in enumerate(self.gpt_layers):
      # Feed the encoding from the last bert_layer to the next.
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # Get the embedding for each input token.
    embedding_output = self.embed(input_ids=input_ids)

    # Feed to a transformer (a stack of GPTLayers).
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)
    sequence_output = self.final_layer_norm(sequence_output)

    # Get the hidden state of the final token.
    last_non_pad_idx = attention_mask.sum(dim=1) - 1  # Subtract 1 to get last index
    last_token = sequence_output[torch.arange(sequence_output.shape[0]), last_non_pad_idx]

    return {'last_hidden_state': sequence_output, 'last_token': last_token}

  def hidden_state_to_token(self, hidden_state):
    """
    TODO-5: Convert hidden states back to token logits.

    Implementation hints: 
    - GPT-2 uses weight tying with the input word embeddings. 
    - The logits are the dot product between output hidden states and the word embedding weights: hidden_state(s) * E^T
    """
    ### YOUR CODE HERE
    # Weight tying: reuse the input embedding matrix E [vocab, hidden] as the output layer.
    # [..., hidden] @ [hidden, vocab] -> [..., vocab]
    return torch.matmul(hidden_state, self.word_embedding.weight.T)


@torch.no_grad()
def generate_gpt2(model, tokenizer, input_ids, max_gen_length=50, device="cuda"):
    """
    Generate text from a GPT-2 model given a single input sequence (greedy decoding).

    Note:
        - Currently only supports batch_size=1 (single input sequence).
        - Using greedy decoding, so each run with the same input produces the same output.
        - Other sampling-based decoding methods (e.g., top-k, top-p, temperature) can introduce randomness and yield different outputs each run.

    Args:
        model: GPT-2 model (pretrained or fine-tuned)
        tokenizer: GPT-2 tokenizer
        input_ids: torch.LongTensor of shape [1, seq_len], input token IDs
        max_gen_length: int, maximum number of tokens to generate
        device: str, "cuda" or "cpu"

    Returns:
        str: Generated text
    """
    model.eval()
    input_ids = input_ids.to(device)  # move input to device
    output_ids = input_ids.clone()

    """
    TODO-8: Greedy next-token generation loop

    Implementation hints:
    Repeat the below steps up to max_gen_length:
    1. Construct an attention mask based on current output_ids (non-pad tokens).
    2. Pass output_ids and attention_mask through the model to get hidden states.
    3. Convert the last hidden state to logits over the vocabulary using model.hidden_state_to_token.
    4. Select the next token using greedy decoding (argmax over logits).
    5. Append the next token to output_ids.
    6. Stop the loop early if the EOS token is generated.
    
    """
    ### YOUR CODE HERE
    for _ in range(max_gen_length):
        # GPT-2 only has 1024 learned position embeddings, so stop before running out of positions
        if output_ids.shape[1] >= model.config.max_position_embeddings:
            break

        # 1. Attention mask: 1 = real token, 0 = padding.
        #    We only handle a single, unpadded sequence, so every position is a real token.
        #    (We cannot test `output_ids != tokenizer.pad_token_id` here: GPT-2's pad token is
        #    the EOS token, and the collate_fn pads with id 0, which is the real token "!".)
        attention_mask = torch.ones_like(output_ids)                     # [1, cur_len]

        # 2. Forward pass. 'last_token' is the final-layer hidden state at the last
        #    non-pad position, i.e. the representation that predicts the NEXT token.
        last_hidden = model(output_ids, attention_mask)['last_token']    # [1, hidden]

        # 3. Hidden state -> vocabulary logits (weight tying: h @ E^T)
        logits = model.hidden_state_to_token(last_hidden)                 # [1, vocab]

        # 4. Greedy decoding: take the single most likely token
        next_token = torch.argmax(logits, dim=-1, keepdim=True)           # [1, 1]

        # 5. Append it so it becomes part of the context for the next step
        output_ids = torch.cat([output_ids, next_token], dim=1)          # [1, cur_len + 1]

        # 6. Stop early once the model emits end-of-sequence
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode generated tokens to string
    ids = output_ids[0]
    text = tokenizer.decode(ids, skip_special_tokens=True)

    return text
