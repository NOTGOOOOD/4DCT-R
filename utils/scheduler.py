import math
import numpy as np

from torch.optim.lr_scheduler import LambdaLR


class StopCriterion(object):
    def __init__(self, patient_len=80, min_epoch=20):
        """
        Parameters
        ----------
        patient_len patient list to stop
        """
        self.patient_len = patient_len
        self.ori_len = patient_len
        self.min_epoch = min_epoch
        self.ncc_loss_list = []
        self.jac_loss_list = []
        self.total_loss_list = []
        self.train_loss_list = []
        self.loss_min = 30.

    def add(self, ncc_loss, jac_loss=None, total_loss=None,train_loss=None):
        self.ncc_loss_list.append(ncc_loss)
        self.total_loss_list.append(total_loss)
        if jac_loss is not None:
            self.jac_loss_list.append(jac_loss)
        if train_loss is not None:
            self.train_loss_list.append(train_loss)

        if total_loss <= self.loss_min:
            self.loss_min = total_loss
            self.loss_min_i = len(self.total_loss_list)
            self.patient_len = self.ori_len

        else:
            self.patient_len = self.patient_len - 1

    def stop(self):
        # return True if the stop creteria are met
        # query_ncc_list = self.ncc_loss_list[-7:]
        # query_mse_lisst = self.mse_loss_list[-7:]
        # query_total_lisst = self.total_loss_list[-7:]
        # std_ncc = np.std(query_ncc_list)
        # std_mse = np.std(query_mse_lisst)
        # std_total = np.std(query_total_lisst)
        # length of patient <=0,and current epoch have no improve
        if len(self.total_loss_list) > self.min_epoch:
            if (self.patient_len <= 0 and len(self.ncc_loss_list) > self.loss_min_i):
                print('early stop by patient_len!')
                return True
            # elif (std_ncc < 0.0001 and std_mse < 0.0001 and len(self.ncc_loss_list) > self.loss_min_i):
            #     print('early stop by std!')
            #     return True
            else:
                return False

        else:
            return False


class ConstantLRSchedule(LambdaLR):
    """ Constant learning rate schedule.
    """

    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLRSchedule, self).__init__(optimizer, lambda _: 1.0, last_epoch=last_epoch)


class WarmupConstantSchedule(LambdaLR):
    """ Linear warmup and then constant.
        Linearly increases learning rate schedule from 0 to 1 over `warmup_steps` training steps.
        Keeps learning rate schedule equal to 1. after warmup_steps.
    """

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        super(WarmupConstantSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        return 1.


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """

    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class WarmupCosineSchedule(LambdaLR):
    """ Linear warmup and then cosine decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps following a cosine curve.
        If `cycles` (default=0.5) is different from default, learning rate follows cosine function after warmup.
    """

    def __init__(self, optimizer, warmup_steps, t_total, cycles=.5, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        self.cycles = cycles
        super(WarmupCosineSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1.0, self.warmup_steps))
        # progress after warmup
        progress = float(step - self.warmup_steps) / float(max(1, self.t_total - self.warmup_steps))
        return max(0.0, 0.5 * (1. + math.cos(math.pi * float(self.cycles) * 2.0 * progress)))
