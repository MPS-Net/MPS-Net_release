import os
import torch
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F

from lib.core.config import BASE_DATA_DIR
from lib.models.spin import Regressor

from torch.autograd import Variable   ## 

class TemporalAttention(nn.Module):
    def __init__(self, attention_size, seq_len, non_linearity='tanh'):
        super(TemporalAttention, self).__init__()

        if non_linearity == "relu":
            activation = nn.ReLU()
        else:
            activation = nn.Tanh()

        self.fc = nn.Linear(attention_size, 256)
        self.relu = nn.ReLU()
        self.attention = nn.Sequential(
            nn.Linear(256 * seq_len, 256),
            activation,
            nn.Linear(256, 256),
            activation,
            nn.Linear(256, seq_len),
            activation
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch = x.shape[0]
        x = self.fc(x)
        x = x.view(batch, -1)

        scores = self.attention(x)
        scores = self.softmax(scores)

        return scores

class TemporalEncoder(nn.Module):                 
    def __init__(self, channel):
        super(TemporalEncoder, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv1d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv1d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv1d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.conv_mask = nn.Conv1d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_mask_forR = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)

        self.attention = TemporalAttention(attention_size=2048, seq_len=3, non_linearity='tanh')     

    def forward(self, x, is_train=False): 
        # NTF -> NFT
        x = x.permute(0,2,1) 
        b, c, thw = x.size()                                                                          	# N x 2048 x 16

        x_=x.permute(0, 2, 1)
        xx = torch.matmul(x_, x)
        xx = self.softmax(xx)                                                                         	# N x 16 x 16
        
        x_phi = self.conv_phi(x).view(b, self.inter_channel, -1)                                     	# N x 2048/2 x 16
        x_theta = self.conv_theta(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()  	# N x 16 x 2048/2 
        x_g = self.conv_g(x).view(b, self.inter_channel, -1).permute(0, 2, 1).contiguous()           	# N x 16 x 2048/2  

        mul_theta_phi = torch.matmul(x_theta, x_phi)
        mul_theta_phi = self.softmax(mul_theta_phi)                                                   	# N x 16 x 16

        R = torch.cat((xx,mul_theta_phi),dim=0).view(xx.size(0),-1,xx.size(1),xx.size(2))             	# 2 x N x 16 x 16
        Y = self.conv_mask_forR(R).reshape(b, thw, thw)     
        Y = self.softmax(Y)                                                                           	# N x 16 x 16 

        mul_theta_phi_g = torch.matmul(Y, x_g)                                                        	# N x 16 x 2048/2
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,self.inter_channel, thw) 	# N x 2048/2 x 16
        
        mask = self.conv_mask(mul_theta_phi_g)                                                        	# N x 2048 x 16   

        out_ = mask + x       #                                                     

        y_cur_2=out_[:,:,8]   #
        y_cur_1=out_[:,:,7]   #
        y_cur_3=out_[:,:,9]   #

        y_bef_2=out_[:,:,5]   #
        y_bef_1=out_[:,:,4]   #
        y_bef_3=out_[:,:,6]   #

        y_aft_2=out_[:,:,11]  #
        y_aft_1=out_[:,:,10]  #
        y_aft_3=out_[:,:,12]  #          

        y_cur_ = torch.cat((y_cur_1[:, None, :], y_cur_2[:, None, :], y_cur_3[:, None, :]), dim=1)   #
        y_bef_ = torch.cat((y_bef_1[:, None, :], y_bef_2[:, None, :], y_bef_3[:, None, :]), dim=1)   #
        y_aft_ = torch.cat((y_aft_1[:, None, :], y_aft_2[:, None, :], y_aft_3[:, None, :]), dim=1)   #

        scores = self.attention(y_cur_)                 #  
        y_cur = torch.mul(y_cur_, scores[:, :, None])   #
        y_cur = torch.sum(y_cur, dim=1)                 #

        scores = self.attention(y_bef_)                 # 
        y_bef = torch.mul(y_bef_, scores[:, :, None])   #             
        y_bef = torch.sum(y_bef, dim=1)                 #         

        scores = self.attention(y_aft_)                 #
        y_aft = torch.mul(y_aft_, scores[:, :, None])   #      
        y_aft = torch.sum(y_aft, dim=1)                 #         

        y = torch.cat((y_bef[:, None, :], y_cur[:, None, :], y_aft[:, None, :]), dim=1)  

        scores = self.attention(y)                  
        out = torch.mul(y, scores[:, :, None])    
        out = torch.sum(out, dim=1)  # N x 2048      

        if not is_train:
            return out, scores, out_       
        else:
            y = torch.cat((out[:, None, :], out[:, None, :], out[:, None, :]), dim=1)     
            return y, scores, out_      


class MPSnet(nn.Module):
    def __init__(
            self,
            seqlen,
            batch_size=64,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(BASE_DATA_DIR, 'spin_model_checkpoint.pth.tar'),
    ):

        super(MPSnet, self).__init__()

        self.seqlen = seqlen
        self.batch_size = batch_size

        self.nonlocalblock = TemporalEncoder(channel=2048)	#nonlocalblock --> real name: MoCA+HAFI

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)
            print(f'=> loaded pretrained model from \'{pretrained}\'')

    def forward(self, input, is_train=False, J_regressor=None):                     
        # input size NTF
        batch_size, seqlen = input.shape[:2] 
   
        feature, scores, feature_seqlen = self.nonlocalblock(input, is_train=is_train)
        feature = Variable(feature.reshape(-1, feature.size(-1)))                    
        feature_seqlen = Variable(feature_seqlen.reshape(-1, feature_seqlen.size(1)))    # 

        smpl_output = self.regressor(feature, is_train=is_train, J_regressor=J_regressor)
        smpl_output_Dm = self.regressor(feature_seqlen, is_train=is_train, J_regressor=J_regressor)     #

        if not is_train:
            for s in smpl_output:
                s['theta'] = s['theta'].reshape(batch_size, -1)
                s['verts'] = s['verts'].reshape(batch_size, -1, 3)
                s['kp_2d'] = s['kp_2d'].reshape(batch_size, -1, 2)
                s['kp_3d'] = s['kp_3d'].reshape(batch_size, -1, 3)
                s['rotmat'] = s['rotmat'].reshape(batch_size, -1, 3, 3)
                s['scores'] = scores

        else:
            repeat_num = 3  
            for s in smpl_output:                
                s['theta'] = s['theta'].reshape(batch_size, repeat_num, -1)               
                s['verts'] = s['verts'].reshape(batch_size, repeat_num, -1, 3)    
                s['kp_2d'] = s['kp_2d'].reshape(batch_size, repeat_num, -1, 2)   
                s['kp_3d'] = s['kp_3d'].reshape(batch_size, repeat_num, -1, 3)   
                s['rotmat'] = s['rotmat'].reshape(batch_size, repeat_num, -1, 3, 3) 
                s['scores'] = scores

            for s_Dm in smpl_output_Dm:                 #
                s_Dm['theta_forDM'] = s_Dm['theta'].reshape(batch_size, seqlen, -1)   

        return smpl_output, scores, smpl_output_Dm      #
