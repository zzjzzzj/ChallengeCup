from __future__ import annotations


def combine_continual_losses(
    *,
    l_box: float,
    l_cls: float,
    l_dfl: float,
    l_kd: float = 0.0,
    l_feat: float = 0.0,
    l_proto: float = 0.0,
    l_rel: float = 0.0,
    l_anchor: float = 0.0,
    lambda_box: float = 1.0,
    lambda_cls: float = 1.0,
    lambda_dfl: float = 1.0,
    lambda_kd: float = 1.0,
    lambda_feat: float = 0.5,
    lambda_proto: float = 0.3,
    lambda_rel: float = 0.1,
    lambda_anchor: float = 0.1,
) -> dict[str, float]:
    l_det = lambda_box * l_box + lambda_cls * l_cls + lambda_dfl * l_dfl
    total = (
        l_det
        + lambda_kd * l_kd
        + lambda_feat * l_feat
        + lambda_proto * l_proto
        + lambda_rel * l_rel
        + lambda_anchor * l_anchor
    )
    return {
        "L_box": float(l_box),
        "L_cls": float(l_cls),
        "L_dfl": float(l_dfl),
        "L_det": float(l_det),
        "L_KD": float(l_kd),
        "L_feat": float(l_feat),
        "L_proto": float(l_proto),
        "L_rel": float(l_rel),
        "L_anchor": float(l_anchor),
        "lambda_box": float(lambda_box),
        "lambda_cls": float(lambda_cls),
        "lambda_dfl": float(lambda_dfl),
        "lambda_KD": float(lambda_kd),
        "lambda_feat": float(lambda_feat),
        "lambda_proto": float(lambda_proto),
        "lambda_rel": float(lambda_rel),
        "lambda_anchor": float(lambda_anchor),
        "L_total": float(total),
    }
