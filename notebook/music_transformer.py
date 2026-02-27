import torch
import torch.nn as nn

class MusicTransformer(nn.Module):
    def __init__(
        self, vocab_size=1024, 
        num_codebooks=4, 
        embed_dim=256, 
        max_seq_len=4096,
        num_layers=6, 
        num_heads=8, 
        dropout=0.2
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embed_dim)
            for _ in range(num_codebooks)
        ])

        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, 
            dim_feedforward=embed_dim * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)

        self.output_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):

        B, T, K = x.shape
        
        h_list = []
        for k in range(K):
            h_list.append(self.embeddings[k](x[:, :, k]))

        h = torch.stack(h_list, dim=2).reshape(B, T * K, -1)

        total_len = T * K
        pos_ids = torch.arange(total_len, device=x.device)
        h = h + self.pos_embedding(pos_ids)[None, :, :]

        mask = torch.triu(
            torch.full((total_len, total_len), float('-inf'), device=x.device),
            diagonal=1
        )

        h = self.transformer(h, mask)

        logits = self.output_head(h)

        logits = logits.reshape(B, T, K, -1)
        
        return logits