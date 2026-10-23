"""Versioned Q1 contact correction, preserving the original R01 adapter."""

from supermimic.q1_assets import Q1Assets, identity


class ContactCorrectedAssets(Q1Assets):
    def configure_model(self, model):
        super().configure_model(model)
        floor = model.geom("floor").id
        model.geom_priority[self.foot_geoms] = model.geom_priority[floor] + 1

    def manifest(self):
        result = super().manifest()
        floor = self.model.geom("floor").id
        result["contact_protocol"] = {
            "version": "Q1_FOOT_PRIORITY_V2",
            "source": identity(__file__),
            "floor_priority": int(self.model.geom_priority[floor]),
            "foot_priorities": self.model.geom_priority[self.foot_geoms].tolist(),
            "semantics": "Foot contact friction follows the episode context; R01 assets remain separate.",
        }
        return result
