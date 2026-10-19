import torch
import torch.nn.functional as F

def supcon_loss(z_i, z_j, z_m, cfg):
    """
    SupCon Loss function.
    Parameters
    ----------
    z_i : torch.tensor
        embedding of original samples (batch_size x emb_size)
    z_j : torch.tensor
        embedding of augmented samples (batch_size x emb_size)
    z_m : torch.tensor
        embedding of another augmented samples (batch_size x emb_size)
    Returns
    -------
    loss
    """
    # tau = cfg['supcon_tau']
    tau = cfg.get('supcon_tau', 0.05)  

    B = z_i.shape[0]
    n_views = 3

    # (3B, D)
    z = torch.cat([z_i, z_j, z_m], dim=0)

    # similarity matrix
    logits = torch.matmul(z, z.T) / tau

    losses = []

    for anchor_idx in range(n_views * B):

        sample_idx = anchor_idx % B

        # remove self from denominator
        mask = torch.ones(n_views * B, dtype=torch.bool,
                          device=z.device)
        mask[anchor_idx] = False

        logits_anchor = logits[anchor_idx][mask]

        log_probs = F.log_softmax(logits_anchor, dim=0)

        # map original indices after self-removal
        kept_indices = torch.arange(
            n_views * B,
            device=z.device
        )[mask]

        # positives = same sample, different views
        pos_mask = (kept_indices % B == sample_idx)

        pos_log_probs = log_probs[pos_mask]

        # Eq. (2): average over positives outside log
        losses.append(-pos_log_probs.mean())

    loss = torch.stack(losses).mean()

    return loss