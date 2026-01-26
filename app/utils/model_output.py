class ModelOutput:
    def __init__(
        self,
        aa=None,
        bb=None,
        ab=None,
        ba=None,
        mask_a_pred=None,
        mask_b_pred=None,
        lm_a_pred=None,
        lm_b_pred=None,
        lm_ab_pred=None,   # ★ 追加
    ):
        self.aa = aa
        self.bb = bb
        self.ab = ab
        self.ba = ba
        self.mask_a_pred = mask_a_pred
        self.mask_b_pred = mask_b_pred
        self.lm_a_pred = lm_a_pred
        self.lm_b_pred = lm_b_pred
        self.lm_ab_pred = lm_ab_pred   # ★ 追加
