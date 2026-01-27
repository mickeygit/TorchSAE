class ModelOutput:
    def __init__(
        self,
        aa,
        bb,
        ab,
        ba,
        mask_a_pred,
        mask_b_pred,
        lm_a_pred,
        lm_b_pred,
        lm_ab_pred,
        aa_exp_only=None,   # ★追加
        bb_exp_only=None,   # ★追加
    ):
        self.aa = aa
        self.bb = bb
        self.ab = ab
        self.ba = ba
        self.mask_a_pred = mask_a_pred
        self.mask_b_pred = mask_b_pred
        self.lm_a_pred = lm_a_pred
        self.lm_b_pred = lm_b_pred
        self.lm_ab_pred = lm_ab_pred

        # ★追加
        self.aa_exp_only = aa_exp_only
        self.bb_exp_only = bb_exp_only
