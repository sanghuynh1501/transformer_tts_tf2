alphabet = " ,a,á,à,ả,ã,ạ,â,ấ,ầ,ẩ,ẫ,ậ,ă,ắ,ằ,ẳ,ẵ,ặ,b,c,d,đ,e,é,è,ẻ,ẽ,ẹ,ê,ế,ề,ể,ễ,ệ,f,g,h,i,í,ì,ỉ,ĩ,ị,j,k,l,m,n,o,ó,ò,ỏ," \
           "õ,ọ,ơ,ớ,ờ,ở,ỡ,ợ,ô,ố,ồ,ổ,ỗ,ộ,p,q,r,s,t,u,ú,ù,ủ,ũ,ụ,ư,ứ,ừ,ử,ữ,ự,v,w,x,y,ý,ỳ,ỷ,ỹ,ỵ,z,0,1,2,3,4,5,6,7,8,9"
alphabet = alphabet.split(",")

debug = False
with_stress = True

dropout_rate = 0.1

mel_end_value = -.5
mel_start_value = .5

mel_channels = 80
stop_loss_scaling = 8.0

postnet_conv_layers = 5
postnet_kernel_size = 5
encoder_dense_blocks = 4
decoder_dense_blocks = 4

postnet_conv_filters = 256

decoder_prenet_dimension = 256
encoder_prenet_dimension = 512

encoder_model_dimension = 512
decoder_model_dimension = 256

decoder_num_heads = [4, 4, 4, 4]
encoder_num_heads = [4, 4, 4, 4]
encoder_attention_conv_kernel = 3
decoder_attention_conv_kernel = 3
encoder_attention_conv_filters = 512
decoder_attention_conv_filters = 512
encoder_max_position_encoding = 1000
decoder_max_position_encoding = 10000

encoder_feed_forward_dimension = 1024
decoder_feed_forward_dimension = 1024

reduction_factor_schedule = [[0, 10], [80_000, 5], [150_000, 3], [250_000, 1]]

# AUDIO
sampling_rate = 22050
n_fft = 1024
mel_channels = 80
hop_length = 256
win_length = 1024
f_min = 0
f_max = 8000
normalizer = 'MelGAN'

data_path = 'data'
