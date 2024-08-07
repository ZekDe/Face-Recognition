# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 19:56:48 2024

@author: Duatepe
"""

def TIME_OVER(target, time, preset_time):
    return (time - target) > preset_time

class Ton:
    def __init__(self):
        self.aux = False
        self.since = 0

    def TON(self, in_signal, now, preset_time):
        ret_val = False

        if in_signal:
            if not self.aux:
                self.since = now
                self.aux = True
            elif TIME_OVER(self.since, now, preset_time):
                ret_val = True
        else:
            self.aux = False
    
        return ret_val
        
        
    