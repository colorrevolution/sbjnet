gpu = 0

lt = 1.
lt_alpha = 1.
lb = 1.
lb_beta = 10.
lf = 1.
lf_theta_1 = 10.
lf_theta_2 = 1.
lf_theta_3 = 500.
epsilon = 1e-8

# train
learning_rate = 1e-3
decay_rate = 0.9
beta1 = 0.9
beta2 = 0.999 
max_iter = 500000
show_loss_interval = 2
write_log_interval = 1
save_ckpt_interval = 20
gen_example_interval = 1000
checkpoint_savedir = 'logs/'
# ckpt_path = 'content/trained_final_5M_.model'
#

ckpt_path = 'content/train_step-180.model'

# data
batch_size = 16
data_shape = [64, None]
data_dir = '/media/mmsys9/系统/syl/SRNet-master-bj/SRNet-master/datasets/ceshi'
i_t_dir = 'i_t'
i_s_dir = 'i_s'
t_sk_dir = 't_sk'
t_t_dir = 't_t'
t_b_dir = 't_b'
t_f_dir = 't_f'
mask_t_dir = 'mask_t'
full_image_dir = "full_image"
example_data_dir = '/media/mmsys9/系统/syl/SRNet-master-bj/SRNet-master/custom_feed/label3'
example_result_dir = '/media/mmsys9/系统/syl/SRNet-master-bj/SRNet-master/custom_feed/gen_logs'

# predict
predict_ckpt_path = None
predict_data_dir = None
predict_result_dir = '/media/mmsys9/系统/syl/SRNet-master-bj/SRNet-master/custom_feed/result'
