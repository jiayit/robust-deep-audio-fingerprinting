import torch
import torch.nn.functional as F

def triplet_loss(z_i, z_j, cfg):
    """
    Triplet loss with hardest negative in batch.

    Parameters
    ----------
    z_i : (B, D)
        embeddings of original samples
    z_j : (B, D)
        embeddings of augmented samples

    Returns
    -------
    loss : scalar
    """
    margin = cfg.get("margin", 0.2)
    # margin = cfg['margin']
    # margin = cfg["loss"].get("margin", 0.2)

    B = z_i.shape[0]

    # same layout as ntxent
    z = torch.stack((z_i, z_j), dim=1).view(2 * B, z_i.shape[1])

    # similarity matrix
    sim = torch.matmul(z, z.T)

    losses = []

    for i in range(2 * B):

        # positive index
        pos_idx = i + 1 if i % 2 == 0 else i - 1

        pos_sim = sim[i, pos_idx]

        # mask out self and positive
        neg_mask = torch.ones(2 * B,
                              dtype=torch.bool,
                              device=z.device)
        neg_mask[i] = False
        neg_mask[pos_idx] = False

        neg_sims = sim[i][neg_mask]

        # hardest negative
        neg_sim = neg_sims.max()

        loss_i = F.relu(neg_sim - pos_sim + margin)

        losses.append(loss_i)

    loss = torch.stack(losses).mean()

    return loss


def triplet_loss1(z_i, z_j, cfg):
    """
    Triplet Loss with configurable negative mining.

    Parameters
    ----------
    z_i : torch.tensor
        embeddings of original samples (B, D)

    z_j : torch.tensor
        embeddings of augmented samples (B, D)

    cfg : dict
        {
            'margin': float,
            'mining': 'hard' | 'semi' | 'random'
        }

    Returns
    -------
    loss : scalar tensor
    """

    margin = cfg.get("margin", 0.2)
    mining = cfg.get("mining", "hard")

    B = z_i.shape[0]

    z = torch.stack((z_i, z_j), dim=1).view(
        2 * B,
        z_i.shape[1]
    )

    # similarity matrix
    sim = torch.matmul(z, z.T)

    losses = []

    for i in range(2 * B):


        pos_idx = i + 1 if i % 2 == 0 else i - 1

        pos_sim = sim[i, pos_idx]

        # remove self and positive
        neg_mask = torch.ones(
            2 * B,
            dtype=torch.bool,
            device=z.device
        )

        neg_mask[i] = False
        neg_mask[pos_idx] = False

        neg_sims = sim[i][neg_mask]

        if mining == "hard":

            # largest similarity
            neg_sim = neg_sims.max()

        elif mining == "semi":

            lower = pos_sim - margin
            upper = pos_sim

            semi_mask = (
                (neg_sims > lower)
                & (neg_sims < upper)
            )

            if semi_mask.any():

                neg_sim = neg_sims[semi_mask].max()

            else:
                # fallback
                neg_sim = neg_sims.max()

        elif mining == "random":

            idx = torch.randint(
                0,
                neg_sims.shape[0],
                (1,),
                device=z.device
            )

            neg_sim = neg_sims[idx]

        else:

            raise ValueError(
                f"Unknown mining strategy: {mining}"
            )

        #
        # max(0, neg - pos + margin)
        #
        loss_i = F.relu(
            neg_sim - pos_sim + margin
        )

        losses.append(loss_i)

    loss = torch.stack(losses).mean()

    return loss