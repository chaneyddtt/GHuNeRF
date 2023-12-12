import os
import sys
from lib.config import cfg

def get_human_info(split):

    data_root = cfg.virt_data_root
    data_name = data_root.split('/')[-1]

    if split == 'train':

        human_info = {'CoreView_313': {'begin_i': 0, 'i_intv': 1, 'ni': 1060},
                      'CoreView_315': {'begin_i': 0, 'i_intv': 1,
                                       'ni': 1400},
                      'CoreView_377': {'begin_i': 0, 'i_intv': 1,
                                       'ni': 617},
                      'CoreView_386': {'begin_i': 0, 'i_intv': 1,
                                       'ni': 646},
                      'CoreView_390': {'begin_i': 0, 'i_intv': 1,
                                       'ni': 999},
                      'CoreView_392': {'begin_i': 0, 'i_intv': 1,
                                       'ni': 556}
                      }

    elif split == 'test':
        if cfg.test_mode == 'model_o_motion_o':
            human_info = {'CoreView_313': {'begin_i': 0, 'i_intv': 1, 'ni': 1060},
                          'CoreView_315': {'begin_i': 0, 'i_intv': 1,
                                           'ni': 1400},
                          'CoreView_377': {'begin_i': 0, 'i_intv': 1,
                                           'ni': 617},
                          'CoreView_386': {'begin_i': 0, 'i_intv': 1,
                                           'ni': 646},
                          'CoreView_390': {'begin_i': 0, 'i_intv': 1,
                                           'ni': 999},
                          'CoreView_392': {'begin_i': 0, 'i_intv': 1,
                                           'ni': 556}
                          }

        elif cfg.test_mode == 'model_o_motion_x':
            human_info = {
                'CoreView_313': {'begin_i': 60, 'i_intv': 1, 'ni': 1000},
                'CoreView_315': {'begin_i': 400, 'i_intv': 1, 'ni': 1000},
                'CoreView_377': {'begin_i': 300, 'i_intv': 1, 'ni': 317},
                'CoreView_386': {'begin_i': 300, 'i_intv': 1, 'ni': 346},
                'CoreView_390': {'begin_i': 0, 'i_intv': 1, 'ni': 700},
                'CoreView_392': {'begin_i': 300, 'i_intv': 1, 'ni': 256}}

        elif cfg.test_mode == 'model_x_motion_x':

            human_info = {
                'CoreView_387': {'begin_i': 0, 'i_intv': 1, 'ni': 654},
                'CoreView_393': {'begin_i': 0, 'i_intv': 1, 'ni': 658},
                'CoreView_394': {'begin_i': 0, 'i_intv': 1, 'ni': 859}}


    return human_info



