from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import ipdb

class NoisePredictionNet(nn.Module, ABC):

    @abstractmethod
    def forward(self, sample, timestep, global_cond):
        raise NotImplementedError


class DiffusionPolicy(nn.Module):
    def __init__(
        self,
        action_len,
        action_dim,
        noise_pred_net,
        num_train_steps=100,
        num_inference_steps=10,
        num_train_noise_samples=1,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        mixture=False,
    ):
        super().__init__()
        self.action_len = action_len
        self.action_dim = action_dim
        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        self.num_train_noise_samples = num_train_noise_samples

        #FIXME:是否要混合损失
        self.mixture = mixture

        # Noise prediction net
        assert isinstance(noise_pred_net, NoisePredictionNet)
        self.noise_pred_net = noise_pred_net

        # Noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_steps,
            beta_schedule=beta_schedule,
            clip_sample=clip_sample,
        )

    @torch.no_grad()
    def sample(self, obs):
        # Initialize sample
        action = torch.randn(
            (obs.shape[0], self.action_len, self.action_dim), device=obs.device
        )

        # Initialize scheduler
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        # Reverse diffusion process
        for t in self.noise_scheduler.timesteps:
            # Predict noise
            noise_pred,_ = self.noise_pred_net(action, t, global_cond=obs)

            # Diffusion step
            action = self.noise_scheduler.step(noise_pred, t, action).prev_sample

        return action

    def forward(self, obs, action, gt_motion):
        # Repeat observations and actions for multiple noise samples
        if self.num_train_noise_samples > 1:
            obs = obs.repeat_interleave(self.num_train_noise_samples, dim=0)
            action = action.repeat_interleave(self.num_train_noise_samples, dim=0)
            if gt_motion is not None:
                gt_motion = gt_motion.repeat_interleave(
                    self.num_train_noise_samples, dim=0
                )

        # Sample random noise
        noise = torch.randn_like(action)

        # Sample a random timestep
        t = torch.randint(
            low=0,
            high=self.num_train_steps,
            size=(action.shape[0],),
            device=action.device,
        ).long()

        # Forward diffusion step
        noisy_action = self.noise_scheduler.add_noise(action, noise, t)

        # Diffusion loss
        #NOTE:返回的pred_motion_feats已经映射到了一样的维度
        noise_pred, pred_motion_feats, pred_quantized_of_logits = self.noise_pred_net(
            noisy_action, t, global_cond=obs
        )

        action_loss = F.mse_loss(noise_pred, noise)

        if self.mixture != 0:
            action_mask = (torch.rand(action.shape[0], device=action.device) > self.mixture).float()
            if action_mask.sum() > 0:
                action_loss = (action_loss * action_mask).sum() / action_mask.sum()
            else:
                # 防止这一 batch 随机出来全是 0 (概率极低)
                action_loss = action_loss.mean() * 0.0

        motion_loss = torch.tensor(0.0, device=action.device)

        if pred_quantized_of_logits is not None:
            if gt_motion is None:
                raise ValueError(
                    "use_quantized_of=True but gt_motion is None. "
                    "Please provide quantized optical-flow token ids with shape [B, N, M]."
                )
            target_tokens = gt_motion.long()
            if target_tokens.ndim == 3:
                target_tokens = target_tokens.reshape(target_tokens.shape[0], -1)
            if target_tokens.ndim != 2:
                raise ValueError(
                    "Quantized optical-flow targets must have shape [B, N, M] or [B, N*M]. "
                    f"Got shape={gt_motion.shape}."
                )
            if target_tokens.shape != pred_quantized_of_logits.shape[:2]:
                raise ValueError(
                    "Quantized optical-flow target shape mismatch: "
                    f"pred={pred_quantized_of_logits.shape[:2]}, gt={target_tokens.shape}."
                )
            motion_loss = F.cross_entropy(
                pred_quantized_of_logits.reshape(-1, pred_quantized_of_logits.shape[-1]),
                target_tokens.reshape(-1),
            )
        elif pred_motion_feats is not None and gt_motion is not None:
            if pred_motion_feats.shape != gt_motion.shape:
                raise ValueError(
                    f"Motion target shape mismatch: pred={pred_motion_feats.shape}, "
                    f"gt={gt_motion.shape}. Please set motion latent shape config to match dataset."
                )
            motion_loss = F.mse_loss(pred_motion_feats, gt_motion)

        total_loss = action_loss + motion_loss
        return {
            "loss": total_loss,
            "action_loss": action_loss,
            "motion_loss": motion_loss,
        }


class FlowPolicy(nn.Module):
    def __init__(
        self,
        action_len,
        action_dim,
        noise_pred_net,
        num_train_steps=100,
        num_inference_steps=10,
        timeshift=1.0,
    ):
        super().__init__()
        self.action_len = action_len
        self.action_dim = action_dim

        # Noise prediction net
        assert isinstance(noise_pred_net, NoisePredictionNet)
        self.noise_pred_net = noise_pred_net

        self.num_train_steps = num_train_steps
        self.num_inference_steps = num_inference_steps
        timesteps = torch.linspace(1, 0, self.num_inference_steps + 1)
        self.timesteps = (timeshift * timesteps) / (1 + (timeshift - 1) * timesteps)

    @torch.no_grad()
    def sample(self, obs):
        # Initialize sample
        action = torch.randn(
            (obs.shape[0], self.action_len, self.action_dim), device=obs.device
        )
        ipdb.set_trace()
        for tcont, tcont_next in zip(self.timesteps[:-1], self.timesteps[1:]):
            # Predict noise
            t = (tcont * self.num_train_steps).long()
            noise_pred = self.noise_pred_net(action, t, global_cond=obs)

            # Flow step
            action = action + (tcont_next - tcont) * noise_pred

        return action

    def forward(self, obs, action):
        # Sample random noise
        noise = torch.randn_like(action)

        # Sample random timestep
        tcont = torch.rand((action.shape[0],), device=action.device)

        # Forward flow step
        direction = noise - action
        noisy_action = (
            action + tcont.view(-1, *[1 for _ in range(action.dim() - 1)]) * direction
        )

        # Flow matching loss
        t = (tcont * self.num_train_steps).long()
        noise_pred = self.noise_pred_net(noisy_action, t, global_cond=obs)
        loss = F.mse_loss(noise_pred, direction)
        return loss
