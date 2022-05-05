"""Network modules of Conditioning"""


from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from extorch import Conv1dEx


@dataclass
class ConfGlobalCondNet:
    """
    Args:
        integration_type - Type of input-conditioning integration
        dim_io - Dimension size of input/output
        dim_cond - Dimension size of conditioning vector
    """
    integration_type: str # "add" | "concat"
    dim_io: int
    dim_cond: int

class GlobalCondNet(nn.Module):
    """Global conditioning

    Add    mode: o_series = i_series + project(expand(cond_vec))
    Concat mode: o_series = project(cat(i_series, expand(cond_vec)))
    """

    def __init__(self, conf: ConfGlobalCondNet):
        super().__init__()
        self.integration_type = conf.integration_type

        # Determine dimension size of integration product
        if conf.integration_type == "add":
            assert conf.dim_cond == conf.dim_io
            # [Batch, T_max, hidden] => [Batch, T_max, hidden==emb] => [Batch, T_max, hidden]
            dim_integrated = conf.dim_io
        elif conf.integration_type == "concat":
            # [Batch, T_max, hidden] => [Batch, T_max, hidden+emb] => [Batch, T_max, hidden]
            dim_integrated = conf.dim_io + conf.dim_cond
        else:
            raise ValueError(f"Integration type '{conf.integration_type}' is not supported.")

        self.projection = torch.nn.Linear(dim_integrated, conf.dim_io)

    def forward(self, i_series, cond_vector):
        """Integrate a conditioning vector with a input series.

        Args:
            i_series (Batch, T, dim_i) - input series
            cond_vector (Batch, dim_cond) - conditioning vector
        Returns:
            Integrated series
        """

        cond_normed = F.normalize(cond_vector)

        if self.integration_type == "add":
            return i_series + self.projection(cond_normed).unsqueeze(1)
        elif self.integration_type == "concat":
            cond_series = cond_normed.unsqueeze(1).expand(-1, i_series.size(1), -1)
            return self.projection(torch.cat([i_series, cond_series], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")
