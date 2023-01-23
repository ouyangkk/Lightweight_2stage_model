import torch
import torch.nn as nn
import hparams as hp
from GrouedGRU import GroupedGRU
from GroupLinear import GroupedLinear
import numpy as np
from thop import profile
from thop import clever_format
from ptflops import get_model_complexity_info
import math

class Stage1(nn.Module):
    #input:batch channel T F
    def __init__(self, padding=2):
        super().__init__()
        self.fc_in = nn.Sequential(nn.Linear(padding*32, 32) ) ##64>>32  loss1   band_mask

        self.emb_gru = GroupedGRU(32,
                                  32,
                                  num_layers=2,
                                  batch_first=False,
                                  groups=1,
                                  shuffle=False,
                                  add_outputs=True)

        self.fc_out_band  = nn.Sequential(GroupedLinear(32, 32,groups=1),  ##60>>20  loss1   band_mask
                                    nn.Sigmoid())
    def gen_inverse_filter(self,hp):
        if(hp.bandmode == "fbank"):
            eband5ms = [0, 3, 5, 8, 10, 13, 15, 18, 22, 25, 29, 33, 38, 42, 48, 53, 59, 66, 73, 80, 89, 97, 107, 117,
                        128,140, 153, 167, 183, 199, 216, 256]  #fbank
        elif(hp.bandmode == "erb"):
            eband5ms = [0, 1, 2, 3, 5, 6, 8, 10, 12, 14, 16, 19, 22, 26, 30, 34, 39, 44, 50, 56, 54, 72, 81, 91,
                        103,116, 130, 145, 163, 183, 205, 256]  #erb
        if(hp.bandmode == "mfcc"):
            eband5ms = [0, 3, 5, 8, 10, 13, 15, 18, 22, 25, 29, 33, 38, 42, 48, 53, 59, 66, 73, 80, 89, 97, 107, 117,
                        128,140, 153, 167, 183, 199, 216, 256]  #fbank
        inverse_filter = torch.zeros([len(eband5ms), eband5ms[-1] + 1])
        for i in range(len(eband5ms) - 1):
            width = (eband5ms[i + 1] - eband5ms[i])
            for j in range(width):
                inverse_filter[i, eband5ms[i] + j] = 1 - j / width
                inverse_filter[i + 1, eband5ms[i] + j] = j / width
        return inverse_filter
    def forward(self,hp,main_input,spec,lps,device): ### 257*3
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        main_input = torch.nn.ZeroPad2d((0, 0, hp.n_expand_fbank, hp.n_expand_fbank))(main_input) #[1, 1, 254, 32]
        main_input= main_input.unfold(2, 2*hp.n_expand_fbank+1, 1)#([1, 1, 252, 32, 3])
        main_input = main_input.float()
        main_input = main_input[...,:hp.n_expand_fbank+1]#([1, 1, 252, 32, 2])
        main_input = torch.reshape(main_input,(hp.batch_size,main_input.shape[2],-1)) # B T F(64)
        x0 = self.fc_in(main_input)
        x0 = x0.permute(1,0,2) # T B F
        x1, _ = self.emb_gru(x0)
        x1 = x1.permute(1,0,2)
        band_mask = self.fc_out_band(x1)  # # B T F
        filter_torch = self.gen_inverse_filter(hp).to(device)
        full_mask = torch.matmul( band_mask,filter_torch)
        full_mask = full_mask.permute(0,2,1)
        full_mask_spec = torch.unsqueeze(full_mask,dim=-1)
        out = spec * full_mask_spec
        return band_mask,full_mask, out

class lps_real(nn.Module):
    def __init__(self, band_size=32, input_size=64, mask_size=257):
        super().__init__()
        '''
        BANKpart
        '''
        self.stage1 = Stage1()

        '''
        full part
        '''
        self.fc_in_low = nn.Sequential(GroupedLinear(128 + 32, 64,groups=1),  ##60>>20  loss1   band_mask
                                   nn.PReLU())
        self.fc_in_high = nn.Sequential(GroupedLinear(128 + 32, 32,groups=2),  ##60>>20  loss1   band_mask
                                   nn.PReLU())
        self.emb_gru2_low = GroupedGRU(64,
                                   64,
                                   num_layers=2,
                                   batch_first=False,
                                   groups=2,
                                   shuffle=True,
                                   add_outputs=True)
        self.emb_gru2_high = GroupedGRU(32,
                                   32,
                                   num_layers=1,
                                   batch_first=False,
                                   groups=2,
                                   shuffle=True,
                                   add_outputs=True)
        self.relu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc_out_low = nn.Sequential(GroupedLinear(64, 128,groups=1),  ##60>>20  loss1   band_mask
                                    nn.Sigmoid())
        self.fc_out_high = nn.Sequential(GroupedLinear(32, 128,groups=2),  ##60>>20  loss1   band_mask
                                    nn.Sigmoid())

    def forward(self,hp,main_input,spec,log_power_input,device):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_power_input_low = log_power_input[:, :, :hp.freq_cut]  # 4 252 257
        log_power_input_high = log_power_input[:, :, hp.freq_cut:256]

        band_mask, full_mask, out = self.stage1(hp,main_input,spec,log_power_input,device)  #band mask 4 252 32
        # full_mask 4 257 252
        mask = torch.ones_like(full_mask[:, :, :256])

        log_power_input_low = torch.cat((log_power_input_low, band_mask), dim=-1)
        log_power_input_low = self.fc_in_low(log_power_input_low)
        log_power_input_low = log_power_input_low.permute(1, 0, 2)  # t B, F
        DF_low, _ = self.emb_gru2_low(log_power_input_low)  # 128
        DF_low = DF_low.permute(1, 0, 2)  # B t, F
        mask_low = self.fc_out_low(DF_low) # B t, F(128)

        log_power_input_high = torch.cat((log_power_input_high, band_mask), dim=-1)
        log_power_input_high= self.fc_in_low(log_power_input_high)
        log_power_input_high = log_power_input_high.permute(1, 0, 2)  # t B, F
        DF_high, _ = self.emb_gru2_high(log_power_input_high)  # 128
        DF_high = DF_high.permute(1, 0, 2)  # B t, F
        mask_high = self.fc_out_high(DF_high) # B t, F(128)

        mask[:,:hp.freq_cut] = mask_low.permute(0,2,1) * full_mask[:,:hp.freq_cut]
        mask[:,hp.freq_cut:-1] = mask_high.permute(0,2,1) * full_mask[:,hp.freq_cut:-1]
        mask[:,256:257] = full_mask[:,256:257]
        mask = torch.unsqueeze(mask,dim=-1)  # B F T 1
        spec_out = mask * out
        real = spec_out[...,0]
        imag = spec_out[...,1]
        spec_mag = torch.abs(torch.complex(real,imag))
        mask = torch.squeeze(mask, dim=-1)
        return full_mask, mask,spec_mag, spec_out

class spec_complex(nn.Module):
    def __init__(self,band_size=32, input_size=64,mask_size=257):
        super().__init__()
        '''
        BANKpart
        '''
        self.stage1 = Stage1()
        '''
        full part
        '''
        self.fc_in_real_low = nn.Sequential(GroupedLinear(128+32, 64,groups=1))

        self.fc_in_real_high = nn.Sequential(GroupedLinear(128+32, 32,groups=2))

        self.fc_in_imag_low = nn.Sequential(GroupedLinear(128+32, 64,groups=1))

        self.fc_in_imag_high = nn.Sequential(GroupedLinear(128+32, 32,groups=2))


        self.gru_real_low = GroupedGRU(64,64,num_layers= 2,batch_first=False,groups=2,shuffle=True,add_outputs=True)

        self.gru_real_high = GroupedGRU(32,32,num_layers= 1 ,batch_first=False,groups=2,shuffle=True,add_outputs=True)

        self.gru_imag_low = GroupedGRU(64,64,num_layers= 2,batch_first=False,groups=2,shuffle=True,add_outputs=True)

        self.gru_imag_high = GroupedGRU(32,32,num_layers= 1 ,batch_first=False,groups=2,shuffle=True,add_outputs=True)

        self.relu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc_out_real_low = nn.Sequential(GroupedLinear(64, 128,groups=1),nn.PReLU())
        self.fc_out_imag_low = nn.Sequential(GroupedLinear(64, 128, groups=1), nn.PReLU())
        self.fc_out_real_high = nn.Sequential(GroupedLinear(32, 128,groups=2),nn.PReLU())
        self.fc_out_imag_high = nn.Sequential(GroupedLinear(32, 128, groups=2), nn.PReLU())
    def forward(self,hp,main_input,specs,lps,device):  ## torch F X T X 2
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ##spec =  4  257 252 2
        real_low = specs[:,:hp.freq_cut,:,0].permute(0,2,1)  # 4  257 252  >> 4 252 257
        imag_low = specs[:,:hp.freq_cut,:,1].permute(0,2,1)   # 4  257 252  >> 4 252 257
        real_high= specs[:,hp.freq_cut:-1,:,0].permute(0,2,1) # 4  257 252  >> 4 252 257
        imag_high = specs[:,hp.freq_cut:-1,:,1].permute(0,2,1)  # 4  257 252  >> 4 252 257


        band_mask, full_mask, out_specs = self.stage1(hp,main_input,specs,lps,device)

        dense_low_r2r = self.fc_in_real_low(torch.cat((real_low,band_mask),dim=-1)) # B T F
        dense_low_r2r = dense_low_r2r.permute(1,0,2)
        r2r_gru_low, _= self.gru_real_low(dense_low_r2r)    ##gru low
        r2r_gru_low = r2r_gru_low.permute(1,0,2)
        out_r2r_low =  self.fc_out_real_low(r2r_gru_low)

        dense_low_r2i = self.fc_in_imag_low(torch.cat((imag_low,band_mask),dim=-1)) # B T F
        dense_low_r2i = dense_low_r2i.permute(1,0,2)
        r2i_gru_low, _= self.gru_imag_low(dense_low_r2i)    ##gru low
        r2i_gru_low = r2i_gru_low.permute(1,0,2)
        out_r2i_low =  self.fc_out_imag_low(r2i_gru_low) ##B T F

        dense_high_r2r = self.fc_in_real_high(torch.cat((real_high, band_mask), dim=-1))
        dense_high_r2r = dense_high_r2r.permute(1,0,2)
        r2r_gru_high,_= self.gru_real_high(dense_high_r2r)  ##gru high
        r2r_gru_high= r2r_gru_high.permute(1, 0, 2)
        out_r2r_high = self.fc_out_real_high(r2r_gru_high)

        dense_high_r2i = self.fc_in_imag_high(torch.cat((imag_high, band_mask), dim=-1))
        dense_high_r2i = dense_high_r2i.permute(1,0,2)
        r2i_gru_high, _= self.gru_imag_high(dense_high_r2i)  ##gru high
        r2i_gru_high= r2i_gru_high.permute(1, 0, 2)
        out_r2i_high = self.fc_out_imag_high(r2i_gru_high)  ##B T F


        real_out = torch.cat((out_r2r_low,out_r2r_high),dim=-1)
        imag_out = torch.cat((out_r2i_low, out_r2i_high), dim=-1) #[4, 252, 256]
        real_out = real_out.permute(0,2,1)
        imag_out = imag_out.permute(0, 2, 1)

        if(hp.mask_mode == "C"):
            mask_mags = (real_out ** 2 + imag_out ** 2) ** 0.5
            real_phase = real_out/ (mask_mags + 1e-8)
            imag_phase = imag_out / (mask_mags + 1e-8)
            enhance_phase = torch.atan2(
                imag_phase,
                real_phase
            )   #[1, 256, 252]
            mask = torch.tanh(mask_mags)  ### 0-1
            mask_final = torch.cat((mask,full_mask[:,256:257]),dim=1)
            real = out_specs[...,0]
            imag = out_specs[...,1]
            spec_complex = torch.complex(real, imag)
            enhance_spec_mag = mask_final * torch.abs(spec_complex)
            enhance_spec_phase =  enhance_phase + torch.angle(spec_complex[:,:256])
            enhance_spec_phase = torch.cat((enhance_spec_phase,torch.angle(spec_complex[:,256:257])),dim=1)
            enhance_spec_real =  enhance_spec_mag * torch.cos(enhance_spec_phase)
            enhance_spec_imag =  enhance_spec_mag * torch.sin(enhance_spec_phase)
            enhance_spec_complex = torch.complex(enhance_spec_real, enhance_spec_imag)
            spec_out = torch.view_as_real(enhance_spec_complex)

        return full_mask, mask_final,enhance_spec_mag, spec_out


class lps_complex(nn.Module):
    def __init__(self,band_size=32, input_size=64,mask_size=257):
        super().__init__()
        '''
        BANKpart
        '''
        self.stage1 = Stage1()
        '''
        full part
        '''
        self.fc_in_real_low = nn.Sequential(GroupedLinear(128+32, 64,groups=1))

        self.fc_in_real_high = nn.Sequential(GroupedLinear(128+32, 32,groups=2))


        self.gru_real_low = GroupedGRU(64,64,num_layers= 2,batch_first=False,groups=2,shuffle=True,add_outputs=True)

        self.gru_real_high = GroupedGRU(32,32,num_layers= 1 ,batch_first=False,groups=2,shuffle=True,add_outputs=True)

        self.relu = nn.PReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc_out_real_low = nn.Sequential(GroupedLinear(64, 128,groups=1),nn.PReLU())
        self.fc_out_imag_low = nn.Sequential(GroupedLinear(64, 128, groups=1), nn.PReLU())
        self.fc_out_real_high = nn.Sequential(GroupedLinear(32, 128,groups=2),nn.PReLU())
        self.fc_out_imag_high = nn.Sequential(GroupedLinear(32, 128, groups=2), nn.PReLU())
    def forward(self,hp,main_input,specs,lps,device):  ## torch F X T X 2
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        log_power_input_low = lps[:, :, :hp.freq_cut]  # 4 252 257
        log_power_input_high = lps[:, :, hp.freq_cut:256]

        band_mask, full_mask, specs_out = self.stage1(hp,main_input,specs,lps,device)

        dense_low_r2r = self.fc_in_real_low(torch.cat((log_power_input_low,band_mask),dim=-1)) # B T F
        dense_low_r2r = dense_low_r2r.permute(1,0,2)
        r2r_gru_low, _= self.gru_real_low(dense_low_r2r)
        r2r_gru_low = r2r_gru_low.permute(1,0,2)
        out_r2r_low =  self.fc_out_real_low(r2r_gru_low)
        out_r2i_low =  self.fc_out_imag_low(r2r_gru_low) ##B T F

        dense_high_r2r = self.fc_in_real_high(torch.cat((log_power_input_high, band_mask), dim=-1))
        dense_high_r2r = dense_high_r2r.permute(1,0,2)
        r2r_gru_high, _= self.gru_real_high(dense_high_r2r)
        r2r_gru_high= r2r_gru_high.permute(1, 0, 2)
        out_r2r_high =  self.fc_out_real_high(r2r_gru_high)
        out_r2i_high = self.fc_out_imag_high(r2r_gru_high)  ##B T F

        real_out = torch.cat((out_r2r_low,out_r2r_high),dim=-1)
        imag_out = torch.cat((out_r2i_low, out_r2i_high), dim=-1) #[4, 252, 256]
        real_out = real_out.permute(0,2,1)
        imag_out = imag_out.permute(0, 2, 1) #[4, 256, 252]  full_mask 4 257 252

        if(hp.mask_mode == "C"):
            mask_mags = (real_out ** 2 + imag_out ** 2) ** 0.5
            real_phase = real_out/ (mask_mags + 1e-8)
            imag_phase = imag_out / (mask_mags + 1e-8)
            enhance_phase = torch.atan2(
                imag_phase,
                real_phase
            )   #[1, 256, 252]
            mask = torch.tanh(mask_mags)  ### 0-1
            mask_final = torch.cat((mask,full_mask[:,256:257]),dim=1)
            # mask_final = torch.unsqueeze(mask_final,dim=-1)     ### 0-1
            real = specs_out[...,0]
            imag = specs_out[...,1]
            spec_complex = torch.complex(real, imag)
            enhance_spec_mag = mask_final * torch.abs(spec_complex)
            enhance_spec_phase =  enhance_phase + torch.angle(spec_complex[:,:256])
            enhance_spec_phase = torch.cat((enhance_spec_phase,torch.angle(spec_complex[:,256:257])),dim=1)
            enhance_spec_real =  enhance_spec_mag * torch.cos(enhance_spec_phase)
            enhance_spec_imag =  enhance_spec_mag * torch.sin(enhance_spec_phase)
            enhance_spec_complex = torch.complex(enhance_spec_real, enhance_spec_imag)
            spec_out = torch.view_as_real(enhance_spec_complex)
        return full_mask, mask_final,enhance_spec_mag, spec_out

if __name__=='__main__':
    hp = hp
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model1 = Stage1()
    model2 = lps_real()
    model3 = lps_complex()
    model4= spec_complex()


    fbank = torch.randn(hp.batch_size,1,252,32)      #FBANK
    spec = torch.randn(hp.batch_size,257,252,2)
    lps= torch.randn(hp.batch_size,252,257)  # COMPLEX

    # flop, params = profile(model1, inputs=(hp,fbank ,spec,lps))
    # macs, params = clever_format([flop, params], "%.3f")
    # print("lps_complex macs = %s parasms = %s" % (macs, params))

    # flop, params = profile(model2, inputs=(hp,fbank,spec,lps))
    # macs, params = clever_format([flop, params], "%.3f")
    # print("group_gru_linear macs = %s parasms = %s" % (macs, params))
    # #

    # flop, params = profile(model3, inputs=(hp,fbank,spec,lps))
    # macs, params = clever_format([flop, params], "%.3f")
    # print("complex macs = %s parasms = %s" % (macs, params))

    flop, params = profile(model4, inputs=(hp,fbank,spec,lps))
    macs, params = clever_format([flop, params], "%.3f")
    print("complex macs = %s parasms = %s" % (macs, params))

    # total_params = sum(p.numel() for p in model1.parameters())
    # print("encode1 total params:",total_params)
    #
    # total_params = sum(p.numel() for p in model3.parameters())
    # print("group_gru_linear_DF total params:",total_params)
    # total_params = sum(p.numel() for p in model4.parameters())
    # print("complex total params:",total_params)
    # total_params = sum(p.numel() for p in model5.parameters())
    # print("LPS_complex total params:",total_params)













