class cebnm:
    def __init__(self,
                 mixture_type="normal",
                 betahat=None,
                 sebetahat=None,
                 max_class=None,
                 mult=None,
                 tt=1.5):
        self.type = mixture_type
        if betahat is not None and sebetahat is not None:
            if self.type == "normal":
                 autoselect_scales_mix_norm(betahat=betahat,
                                            sebetahat=sebetahat,
                                            mult=mult,
                                            max_class=max_class)
        if self.type=="exp":
                autoselect_scales_mix_norm(betahat=betahat,
                                            sebetahat=sebetahat,
                                            mult=mult,
                                            max_class=max_class,
                                            tt=tt)
                
                    
