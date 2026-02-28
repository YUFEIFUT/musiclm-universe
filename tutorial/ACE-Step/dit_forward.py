#    DiT模型分析 forward
#    class AceStepDiTModel(AceStepPreTrainedModel)
#    用于 AceStep 的 DiT（扩散Transformer）模型。

#    这是主扩散模型，基于文本、歌词和音色条件，生成音频隐变量。
#    采用基于补丁（patch）的处理方式，搭配 Transformer 层、时间步条件注入，
#    并通过交叉注意力机制与编码器输出进行交互。

def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_r: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        context_latents: torch.Tensor,
        use_cache: Optional[bool] = None,
        past_key_values: Optional[EncoderDecoderCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        return_hidden_states: int = None,
        custom_layers_config: Optional[dict] = None,
        enable_early_exit: bool = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # Disable cache during training or when gradient checkpointing is enabled
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False
        if self.training:
            use_cache = False
    
        # Initialize cache if needed (only during inference for auto-regressive generation)
        if not self.training and use_cache and past_key_values is None:
            past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())

        # Compute timestep embeddings for diffusion conditioning
        # Two embeddings: one for timestep t, one for timestep difference (t - r)
        temb_t, timestep_proj_t = self.time_embed(timestep)
        temb_r, timestep_proj_r = self.time_embed_r(timestep - timestep_r)
        # Combine embeddings
        temb = temb_t + temb_r
        timestep_proj = timestep_proj_t + timestep_proj_r

        # Concatenate context latents (source latents + chunk masks) with hidden states
        hidden_states = torch.cat([context_latents, hidden_states], dim=-1)
        # Record original sequence length for later restoration after padding
        original_seq_len = hidden_states.shape[1]
        # Apply padding if sequence length is not divisible by patch_size
        # This ensures proper patch extraction
        pad_length = 0
        if hidden_states.shape[1] % self.patch_size != 0:
            pad_length = self.patch_size - (hidden_states.shape[1] % self.patch_size)
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_length), mode='constant', value=0)

        # Project input to patches and project encoder states
        hidden_states = self.proj_in(hidden_states)
        encoder_hidden_states = self.condition_embedder(encoder_hidden_states)
        
        # Cache positions
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + hidden_states.shape[1], device=hidden_states.device
            )
        
        # Position IDs
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)


        seq_len = hidden_states.shape[1]
        encoder_seq_len = encoder_hidden_states.shape[1]
        dtype = hidden_states.dtype
        device = hidden_states.device
        
        # 判断是否使用 Flash Attention 2
        is_flash_attn = (self.config._attn_implementation == "flash_attention_2")

        # 初始化 Mask 变量
        full_attn_mask = None
        sliding_attn_mask = None
        encoder_attention_mask = None
        attention_mask = None

        full_attn_mask = attention_mask
        sliding_attn_mask = attention_mask if self.config.use_sliding_window else None

        # 构建 Mapping
        self_attn_mask_mapping = {
            "full_attention": full_attn_mask,
            "sliding_attention": sliding_attn_mask,
            "encoder_attention_mask": encoder_attention_mask,
        }

        # Create position embeddings to be shared across all decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_cross_attentions = () if output_attentions else None

        # Handle early exit for custom layer configurations
        max_needed_layer = float('inf')
        if custom_layers_config is not None and enable_early_exit:
            max_needed_layer = max(custom_layers_config.keys())
            # Force output_attentions to True when early exit is enabled for attention extraction
            output_attentions = True
            if all_cross_attentions is None:
                all_cross_attentions = ()

        # Process through transformer layers
        for index_block, layer_module in enumerate(self.layers):

            layer_outputs = layer_module(
                hidden_states,
                position_embeddings,
                timestep_proj,
                self_attn_mask_mapping[layer_module.attention_type],
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                encoder_hidden_states,
                self_attn_mask_mapping["encoder_attention_mask"],
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]

            if output_attentions and self.layers[index_block].use_cross_attention:
                # layer_outputs structure: (hidden_states, self_attn_weights, cross_attn_weights)
                # Extract the last element which is cross_attn_weights
                if len(layer_outputs) >= 3:
                    all_cross_attentions += (layer_outputs[2],)
        
        if return_hidden_states:
            return hidden_states

        # Extract scale-shift parameters for adaptive output normalization
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        # Apply adaptive layer norm: norm(x) * (1 + scale) + shift
        hidden_states = (self.norm_out(hidden_states) * (1 + scale) + shift).type_as(hidden_states)
        # Project output: de-patchify back to original sequence format
        hidden_states = self.proj_out(hidden_states)
        
        # Crop back to original sequence length to ensure exact length match (remove padding)
        hidden_states = hidden_states[:, :original_seq_len, :]
        
        outputs = (hidden_states, past_key_values)

        if output_attentions:
            outputs += (all_cross_attentions,)
        return outputs
