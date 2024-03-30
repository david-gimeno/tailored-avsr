import torch

class InterCTCResidualModule(torch.nn.Module):

    def __init__(self, dim_model, vocab_size):
        super().__init__()

        self.proj_1 = torch.nn.Linear(dim_model, vocab_size)
        self.proj_2 = torch.nn.Linear(vocab_size, dim_model)

    def forward(self, x):

        logits = self.proj_1(x)
        x = x + self.proj_2(logits.softmax(dim=-1))

        return x, logits
