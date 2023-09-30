from .triplet_margin_loss import TripletMarginLoss


class SelectivelyContrastiveTripletLoss(TripletMarginLoss):
    """ Implementation of Selectively Contrastive Triplet Loss from https://arxiv.org/pdf/2007.12749.pdf

    Brifely: To address the issue of difficult negative triplets.
    Instead of treating them as three connected points, we split them into two sets:
        - ones with the anchor and positive,
        - and others with the anchor and negative.

    We then only focus on the anchor-negative pairs.

    LSC(S_ap, S_an) = penalty * S_an, if S_an > S_ap
    LSC(S_ap, S_an) = L(Sa_ap, S_an), otherwise

    Usually, triplet loss methods aim to make anchor-positive pairs from the same class very close to each other.
    With LSC, these pairs aren't adjusted, leading to less tight groupings within a class.
    This change helps create features that generalize better and aren't overly tailored to the training data.
    Most importantly, the network can then focus on separating the challenging negative examples.

    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        smooth_loss: Use the log-exp version of the triplet loss
        penalty: Penalizes anchor-negative similarity
    """

    def __init__(
        self,
        margin=0.05,
        swap=False,
        smooth_loss=False,
        triplets_per_anchor="all",
        penalty=1.0,
        **kwargs
    ):
        super(TripletMarginLoss, self).__init__(margin, swap, smooth_loss, triplets_per_anchor, **kwargs)
        self.penalty = penalty

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        pass