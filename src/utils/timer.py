# -*- coding: utf-8 -*-      
# --------------------------------------------
# Author: chen hao
# Date: 2021/8/29 11:56 
# Description:  
# --------------------------------------------
def stats_time(start, end, step, total_step):
    t = end -start
    return '{:.3f}'.format((t / step * (total_step - step) / 3600))