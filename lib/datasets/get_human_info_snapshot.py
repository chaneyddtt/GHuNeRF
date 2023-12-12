import os
import sys
from lib.config import cfg

def get_human_info(split):

    data_root = cfg.virt_data_root
    data_name = data_root.split('/')[-1]

    if split == 'train':
        if cfg.use_all_frames:
            human_info = {'female-1-casual': {'begin_i': 0, 'i_intv': 1, 'ni': 757},
                          'female-3-sport': {'begin_i': 0, 'i_intv': 1, 'ni': 511},
                          'female-4-casual': {'begin_i': 0, 'i_intv': 1, 'ni': 524},
                          'female-4-sport': {'begin_i': 0, 'i_intv': 1, 'ni': 422},
                          'female-6-plaza': {'begin_i': 0, 'i_intv': 1, 'ni': 587},
                          'female-7-plaza': {'begin_i': 0, 'i_intv': 1, 'ni': 499},
                          'female-8-plaza': {'begin_i': 0, 'i_intv': 1, 'ni': 544},

                          'male-1-plaza': {'begin_i': 0, 'i_intv': 1, 'ni': 578},
                          'male-1-sport': {'begin_i': 0, 'i_intv': 1, 'ni': 575},
                          'male-2-casual': {'begin_i': 0, 'i_intv': 1, 'ni': 473},
                          'male-2-outdoor': {'begin_i': 0, 'i_intv': 1, 'ni': 416},
                          'male-2-plaza': {'begin_i': 0, 'i_intv': 1, 'ni': 500},
                          'male-2-sport': {'begin_i': 0, 'i_intv': 1, 'ni': 467},

                          'male-3-outdoor': {'begin_i': 0, 'i_intv': 1, 'ni': 666},
                          'male-3-plaza': {'begin_i': 0, 'i_intv': 1, 'ni': 735},
                          'male-3-sport': {'begin_i': 0, 'i_intv': 1, 'ni': 610},
                          'male-4-casual': {'begin_i': 0, 'i_intv': 1, 'ni': 873},
                          'male-4-sport':  {'begin_i': 0, 'i_intv': 1, 'ni': 723},
                          'male-5-outdoor': {'begin_i': 0, 'i_intv': 1, 'ni': 863},
                          'male-5-sport': {'begin_i': 0, 'i_intv': 1, 'ni': 1429},
                          'male-9-plaza': {'begin_i': 0, 'i_intv': 1, 'ni': 983}
                          }
        else:
            human_info = {'female-1-casual': {'begin_i': 0, 'i_intv': 1, 'ni': 258},
                          'female-3-sport': {'begin_i': 0, 'i_intv': 1, 'ni': 166},
                          'female-4-casual': {'begin_i': 0, 'i_intv': 1, 'ni': 218},
                          'female-4-sport': {'begin_i': 0, 'i_intv': 1, 'ni': 145},
                          'female-6-plaza': {'begin_i': 0, 'i_intv': 1, 'ni': 233},
                          'female-7-plaza': {'begin_i': 0, 'i_intv': 1, 'ni': 185},
                          'female-8-plaza': {'begin_i': 0, 'i_intv': 1, 'ni': 200},

                          'male-1-plaza': {'begin_i': 0, 'i_intv': 1, 'ni': 210},
                          'male-1-sport': {'begin_i': 0, 'i_intv': 1, 'ni': 230},
                          'male-2-casual': {'begin_i': 0, 'i_intv': 1, 'ni': 180},
                          'male-2-outdoor': {'begin_i': 0, 'i_intv': 1, 'ni': 150},
                          'male-2-plaza': {'begin_i': 0, 'i_intv': 1, 'ni': 190},
                          'male-2-sport': {'begin_i': 0, 'i_intv': 1, 'ni': 170},

                          'male-3-outdoor': {'begin_i': 0, 'i_intv': 1, 'ni': 260},
                          'male-3-plaza': {'begin_i': 0, 'i_intv': 1, 'ni': 245},
                          'male-3-sport': {'begin_i': 0, 'i_intv': 1, 'ni': 220},
                          'male-4-casual': {'begin_i': 0, 'i_intv': 1, 'ni': 340},
                          'male-4-sport': {'begin_i': 0, 'i_intv': 1, 'ni': 250},
                          'male-5-outdoor': {'begin_i': 0, 'i_intv': 1, 'ni': 310},
                          'male-5-sport': {'begin_i': 0, 'i_intv': 1, 'ni': 390},
                          'male-9-plaza': {'begin_i': 0, 'i_intv': 1, 'ni': 410}
                          }
    elif split == 'test':
        if cfg.test_mode == 'model_o_motion_o':
            human_info = {'female-3-casual': {'begin_i': 0, 'i_intv': 1, 'ni': 648},
                          }

        elif cfg.test_mode == 'model_x_motion_x':
            if cfg.use_all_frames:
                human_info = {
                    'female-3-casual': {'begin_i': 0, 'i_intv': 1, 'ni': 648},
                    'male-1-casual': {'begin_i': 0, 'i_intv': 1, 'ni': 834},
                    'male-3-casual': {'begin_i': 0, 'i_intv': 1, 'ni': 689}}
            else:
                human_info = {
                    'female-3-casual': {'begin_i': 0, 'i_intv': 1, 'ni': 215},
                    'male-1-casual': {'begin_i': 0, 'i_intv': 1, 'ni': 400},
                    'male-3-casual': {'begin_i': 0, 'i_intv': 1, 'ni': 250}}


    return human_info



