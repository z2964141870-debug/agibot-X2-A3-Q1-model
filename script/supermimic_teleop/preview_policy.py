"""Independent full-point human preview adapter; existing inference stays frozen."""

from stream_contract import DT_NS, HumanPose, PreviewWindow, SLOTS


def validate_window(window):
    if not isinstance(window, PreviewWindow) or not window.can_step:
        raise ValueError("Human policy needs a fresh available preview; caller must handle fallback")
    count = window.prefix_frames
    if not 1 <= count <= window.requested_frames <= SLOTS:
        raise ValueError("Invalid human policy preview prefix")
    if window.policy_frame_mask != tuple(i < count for i in range(SLOTS)):
        raise ValueError("Human frame mask must be a continuous prefix")
    if len(window.poses) != SLOTS or not all(isinstance(pose, HumanPose) for pose in window.poses):
        raise ValueError("Human policy requires ten valid packing poses")
    targets = tuple(window.origin_ns + (i+1)*DT_NS for i in range(SLOTS))
    if window.target_ns != targets or window.capture_cutoff_ns > window.control_ns:
        raise ValueError("Human preview time convention changed")
    if len(window.used_sequences) != count or len(window.provenance) != SLOTS:
        raise ValueError("Missing preview provenance")
    if len(window.measured_mask) != SLOTS or any(type(value) is not bool for value in window.measured_mask):
        raise ValueError("Measured mask must contain ten boolean values")
    if window.terminal_capture_ns is not None:
        if window.source.preprocessing != "offline_unverified":
            raise ValueError("Offline terminal capture cannot claim live measured input")
        if type(window.terminal_capture_ns) is not int or \
                not 0 <= window.terminal_capture_ns <= window.capture_cutoff_ns:
            raise ValueError("Terminal capture must be an available explicit offline event")
    for i, kind in enumerate(window.provenance):
        if i < count:
            if kind not in ("measured_exact", "declared_clip_terminal_hold"):
                raise ValueError("Active goal slot has unknown provenance")
            if kind == "declared_clip_terminal_hold" and window.source.preprocessing != "offline_unverified":
                raise ValueError("Offline terminal padding cannot claim live measured input")
            if window.measured_mask[i] != (kind == "measured_exact"):
                raise ValueError("A held/forecast frame cannot claim to be measured")
        elif kind != "masked_hold_placeholder" or window.measured_mask[i]:
            raise ValueError("Hidden tail must be an unmeasured masked placeholder")
    return count


class HumanPreviewPolicy:
    def __init__(self, native):
        self.native = native

    def __call__(self, history, window, robot_quat):
        import torch
        from morphmimic.model.human import fixed_human_masks, pack_human_goal
        from morphmimic.train.imagination import act_human

        count = validate_window(window)
        if history.shape != (1, 930) or robot_quat.shape != (1, 4):
            raise ValueError("Independent preview replay requires one native Q1 history and torso quaternion")
        with torch.inference_mode():
            position = torch.tensor([pose.positions for pose in window.poses],
                                    dtype=history.dtype, device=history.device)[None]
            orientation = torch.tensor([pose.orientations for pose in window.poses],
                                       dtype=history.dtype, device=history.device)[None]
            raw = pack_human_goal(position, orientation, robot_quat)
            masks = fixed_human_masks(1, "full", count, history.device, heading=True)
            action = act_human(self.native.policy, self.native.norms["obs_norm"],
                self.native.norms["human_norm"], self.native.norms["heading_norm"], history, raw, *masks)
        if action.shape != (1, 29) or not torch.isfinite(action).all().item():
            raise RuntimeError("Invalid preview policy action")
        if torch.count_nonzero(action[..., ~self.native.present]).item():
            raise RuntimeError("Preview action did not mask absent canonical joints")
        return action
