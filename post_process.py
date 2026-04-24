import numpy as np
import torch
import torch.nn.functional as F


def post_process(c, size_list, outputs_list, return_maps: bool = True):
    print('Multi-scale sizes:', size_list)

    top_k = max(1, int(c.input_size[0] * c.input_size[1] * c.top_k))

    # ---------------------------------------------------------
    # Detection-only mode: do NOT store all 512x512 maps at once
    # ---------------------------------------------------------
    if not return_maps:
        anomaly_scores = []

        num_batches = len(outputs_list[0])

        with torch.no_grad():
            for bidx in range(num_batches):
                logp_map = None

                for lvl in range(len(outputs_list)):
                    outputs = outputs_list[lvl][bidx].to(c.device)

                    up = F.interpolate(
                        outputs.unsqueeze(1),
                        size=c.input_size,
                        mode='bilinear',
                        align_corners=True
                    ).squeeze(1)

                    if logp_map is None:
                        logp_map = up
                    else:
                        logp_map = logp_map + up

                logp_map = logp_map - logp_map.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
                prop_map_mul = torch.exp(logp_map)

                anomaly_score_map_mul = (
                    prop_map_mul.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
                    - prop_map_mul
                )

                batch = anomaly_score_map_mul.shape[0]

                score = (
                    anomaly_score_map_mul.reshape(batch, -1)
                    .topk(top_k, dim=-1)[0]
                    .mean(dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                )

                anomaly_scores.append(score)

                del logp_map, prop_map_mul, anomaly_score_map_mul

        anomaly_score = np.concatenate(anomaly_scores, axis=0)
        return anomaly_score, None, None

    # ---------------------------------------------------------
    # Original localization mode
    # ---------------------------------------------------------
    logp_maps = [list() for _ in size_list]
    prop_maps = [list() for _ in size_list]

    for l, outputs in enumerate(outputs_list):
        outputs = torch.cat(outputs, 0)

        logp_maps[l] = F.interpolate(
            outputs.unsqueeze(1),
            size=c.input_size,
            mode='bilinear',
            align_corners=True
        ).squeeze(1)

        output_norm = outputs - outputs.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        prob_map = torch.exp(output_norm)

        prop_maps[l] = F.interpolate(
            prob_map.unsqueeze(1),
            size=c.input_size,
            mode='bilinear',
            align_corners=True
        ).squeeze(1)

    logp_map = sum(logp_maps)
    logp_map = logp_map - logp_map.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]

    prop_map_mul = torch.exp(logp_map)
    anomaly_score_map_mul = (
        prop_map_mul.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        - prop_map_mul
    )

    batch = int(anomaly_score_map_mul.shape[0])

    anomaly_score = (
        anomaly_score_map_mul.reshape(batch, -1)
        .topk(top_k, dim=-1)[0]
        .mean(dim=1)
        .detach()
        .cpu()
        .numpy()
    )

    prop_map_add = sum(prop_maps)
    prop_map_add = prop_map_add.detach().cpu().numpy()
    anomaly_score_map_add = prop_map_add.max(axis=(1, 2), keepdims=True) - prop_map_add

    return anomaly_score, anomaly_score_map_add, anomaly_score_map_mul.detach().cpu().numpy()
