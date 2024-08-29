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
            # Common arguments
            args = {
                'betahat': betahat,
                'sebetahat': sebetahat
            }
            
            # Optional arguments
            if mult is not None:
                args['mult'] = mult
            if max_class is not None:
                args['max_class'] = max_class

            # Call the appropriate function based on the mixture type
            if self.type == "normal":
                autoselect_scales_mix_norm(**args)
            elif self.type == "exp":
                args['tt'] = tt  # Add 'tt' for "exp" type
                autoselect_scales_mix_exp(**args)
