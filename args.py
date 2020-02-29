class Arguments:
    def __init__(self):
        # Path setting
        self.model_path = "./checkpoint"
        self.resume_model = None

        # Hyper params 
        self.d_model = 512
        self.n_head = 8
        self.num_enc_layers = 6
        self.num_dec_layers = 6
        self.dim_feedforword = 2048
        self.dropout = 0.5
        self.activation = 'relu'

        # options
        self.dataset = 'multi30k'
        self.model_type = 'transformer'
        self.store_name = '_'.join([self.model_type,self.dataset])
        self.device_list = '0,2'
        self.log_interval = 100
    