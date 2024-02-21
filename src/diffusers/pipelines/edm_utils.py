from contextlib import contextmanager
import torch

class EDMScaling:
    def __init__(self, sigma_data=0.5):
        self.sigma_data = sigma_data

    def __call__(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        c_noise = 0.25 * sigma.log()
        return c_skip, c_out, c_in, c_noise

edm_scaling = EDMScaling()


def new_monkeypatched_scheduler_for_edm(scheduler, sigma_min=0.002, sigma_max=80.0):
    """
    Creates a monkeypatched scheduler that supports EDM-style inference.
    """
    from copy import deepcopy

    scheduler = deepcopy(scheduler)

    scheduler.use_karras_sigmas = True
    scheduler.config.use_karras_sigmas = True
    scheduler.config.sigma_min = 0.002
    scheduler.config.sigma_max = 80.0
    scheduler.config.prediction_type = "sample"
    scheduler.config.use_edm = True

    def edm_sigma_to_t(self, sigma, log_sigmas):
        """Do not discretize from sigma to t, just return sigma"""
        return sigma

    def edm_sigma_to_alpha_sigma_t(self, sigma):
        """
        For DPMSolverMultistepScheduler:
        Inputs are pre-scaled before going into unet, so alpha_t = 1
        """
        alpha_t = torch.tensor(1)
        sigma_t = sigma
        return alpha_t, sigma_t

    from types import MethodType

    scheduler._sigma_to_t = MethodType(edm_sigma_to_t, scheduler)
    scheduler._sigma_to_alpha_sigma_t = MethodType(edm_sigma_to_alpha_sigma_t, scheduler)

    return scheduler


@contextmanager
def swap_scheduler(self, new_scheduler):
    """
    A context manager to temporarily swap the scheduler of the pipeline with a new one.
    """
    original_scheduler, self.scheduler = self.scheduler, new_scheduler
    try:
        yield
    finally:
        self.scheduler = original_scheduler


def edm_init_noise_sigma(scheduler):
    max_sigma = max(scheduler.sigmas) if isinstance(scheduler.sigmas, list) else scheduler.sigmas.max()
    try:
        if scheduler.config.timestep_spacing in ["linspace", "trailing"]:
            return max_sigma
    except AttributeError:
        pass

    return (max_sigma**2 + 1) ** 0.5