"""Exact collision support with bounded temporary memory for H32 batches."""

import torch

from tensor_geometry import TensorSole


class BoundedTensorSole(TensorSole):
    chunk_size = 1024

    @classmethod
    def from_existing(cls, original):
        value = cls.__new__(cls)
        torch.nn.Module.__init__(value)
        value.body_count = original.body_count
        value.body_ids = list(original.body_ids)
        value.contract = {**original.contract,
            "support_selection": "All original points, chunked argmin; one minimizing material point at ties",
            "maximum_query_rows_per_chunk": cls.chunk_size}
        for foot in range(2):
            for kind in ("points", "radii"):
                name = f"{kind}_{foot}"
                value.register_buffer(name, getattr(original, name))
        return value

    def offsets(self, rotations):
        if rotations.shape[-3:] != (self.body_count, 3, 3):
            raise ValueError("Expected full-body rotations")
        selected = []
        for foot, body in enumerate(self.body_ids):
            points = getattr(self, f"points_{foot}").to(rotations)
            radii = getattr(self, f"radii_{foot}").to(rotations)
            rows = rotations[..., body, 2, :].reshape(-1, 3)
            # The minimizing index is piecewise constant. Only the chosen
            # support point needs a gradient, not a batch-by-mesh allocation.
            with torch.no_grad():
                indices = torch.cat([(part @ points.T - radii).argmin(-1)
                                     for part in rows.split(self.chunk_size)])
            local = points[indices].reshape(*rotations.shape[:-3], 3)
            offset = (rotations[..., body, :, :] @ local.unsqueeze(-1)).squeeze(-1)
            radius = radii[indices].reshape(rotations.shape[:-3])
            offset = offset - torch.stack((torch.zeros_like(radius), torch.zeros_like(radius), radius), -1)
            selected.append(offset)
        return torch.stack(selected, -2)

    def forward(self, positions, rotations):
        if positions.shape[-2:] != (self.body_count, 3) or positions.shape[:-2] != rotations.shape[:-3]:
            raise ValueError("Full-body position/rotation dimensions disagree")
        return positions[..., self.body_ids, 2] + self.offsets(rotations)[..., 2]
