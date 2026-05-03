from typing import *


class ClassifierFreeGuidanceSamplerMixin:
    """
    A mixin class for samplers that apply classifier-free guidance.
    """

    def _inference_model(self, model, x_t, t, cond, neg_cond, guidance_strength, guidance_rescale=0.0, **kwargs):
        if guidance_strength == 1:
            return super()._inference_model(model, x_t, t, cond, **kwargs)
        elif guidance_strength == 0:
            return super()._inference_model(model, x_t, t, neg_cond, **kwargs)
        else:
            pred_pos = super()._inference_model(model, x_t, t, cond, **kwargs)
            pred_neg = super()._inference_model(model, x_t, t, neg_cond, **kwargs)
            print(f"[DIAG] CFG._inference_model: t={t:.4f}, guidance_strength={guidance_strength}, guidance_rescale={guidance_rescale}")
            print(f"[DIAG] CFG._inference_model: pred_pos min/max/mean={pred_pos.min().item():.4f}/{pred_pos.max().item():.4f}/{pred_pos.mean().item():.4f}")
            print(f"[DIAG] CFG._inference_model: pred_neg min/max/mean={pred_neg.min().item():.4f}/{pred_neg.max().item():.4f}/{pred_neg.mean().item():.4f}")
            pred = guidance_strength * pred_pos + (1 - guidance_strength) * pred_neg
            
            # CFG rescale
            if guidance_rescale > 0:
                x_0_pos = self._pred_to_xstart(x_t, t, pred_pos)
                x_0_cfg = self._pred_to_xstart(x_t, t, pred)
                std_pos = x_0_pos.std(dim=list(range(1, x_0_pos.ndim)), keepdim=True)
                std_cfg = x_0_cfg.std(dim=list(range(1, x_0_cfg.ndim)), keepdim=True)
                x_0_rescaled = x_0_cfg * (std_pos / std_cfg)
                x_0 = guidance_rescale * x_0_rescaled + (1 - guidance_rescale) * x_0_cfg
                pred = self._xstart_to_pred(x_t, t, x_0)
                
            print(f"[DIAG] CFG._inference_model: final pred min/max/mean={pred.min().item():.4f}/{pred.max().item():.4f}/{pred.mean().item():.4f}")
            return pred
