"""Full-ten-slot human policy input from an explicitly audited causal forecast."""

from causal_tail import validate_forecast


class HumanForecastPolicy:
    def __init__(self, native):
        self.native = native

    def __call__(self, history, forecast, robot_quat):
        import torch
        from morphmimic.model.human import fixed_human_masks, pack_human_goal
        from morphmimic.train.imagination import act_human

        forecast = validate_forecast(forecast)
        if history.shape != (1, 930) or robot_quat.shape != (1, 4):
            raise ValueError("Forecast policy needs one canonical history and torso orientation")
        with torch.inference_mode():
            position = torch.tensor([pose.positions for pose in forecast.poses],
                                    dtype=history.dtype, device=history.device)[None]
            orientation = torch.tensor([pose.orientations for pose in forecast.poses],
                                       dtype=history.dtype, device=history.device)[None]
            raw = pack_human_goal(position, orientation, robot_quat)
            action = act_human(self.native.policy, self.native.norms["obs_norm"],
                self.native.norms["human_norm"], self.native.norms["heading_norm"], history, raw,
                *fixed_human_masks(1, "full", 10, history.device, heading=True))
        if action.shape != (1, 29) or not torch.isfinite(action).all().item() or \
                torch.count_nonzero(action[..., ~self.native.present]).item():
            raise RuntimeError("Invalid forecast policy action")
        return action
