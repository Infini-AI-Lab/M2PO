# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: if True, normalize advantage by std within group

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores

def compute_direct_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
):
    """
    Compute advantage where advantage = reward directly.
    Keeps same inputs/outputs as compute_rloo_outcome_advantage.

    Args:
        token_level_rewards: (torch.Tensor) shape (bs, response_length)
        response_mask:       (torch.Tensor) shape (bs, response_length)
        index:               (np.ndarray)   shape (bs,), prompt ids (unused here)
        epsilon:             float, unused in this simple version

    Returns:
        advantages: (torch.Tensor) shape (bs, response_length)
        returns:    (torch.Tensor) shape (bs, response_length)
    """
    # 只累计有效 token 的奖励得到每个样本的总 reward
    scores = token_level_rewards.sum(dim=-1)  # (bs,)

    # 广播到 token 维，并再次用 mask 清零 padding 位置
    rewards_broadcast = scores.unsqueeze(-1).expand_as(token_level_rewards)  # (bs, L)
    rewards_broadcast[rewards_broadcast == 0] = -1 # set negative reward to -1
    returns = rewards_broadcast * response_mask  # (bs, L)

    # 这里定义 advantage = reward
    advantages = returns

    return advantages, returns

def compute_opo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.tensor(id2score[idx])
                len_tensor = torch.tensor(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss

@torch.no_grad()
def get_ratio_stats(ratio: torch.Tensor,
                    advantages: torch.Tensor,
                    response_mask: torch.Tensor,
                    log_prob: torch.Tensor,
                    old_log_prob: torch.Tensor,
                    bins=(0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0),
                    eps: float = 1e-12,
                    tol: float = 1e-6):
    """
    Summarize ratio distribution for three advantage conditions (pos, neg, nonzero).
    Keeps the (0.8, 1.0) bin AND adds an explicit eq_1.0 bin.

    Final bin order (len=9):
        (-inf, 0.2], (0.2, 0.5], (0.5, 0.8], (0.8, 1.0), eq_1.0,
        (1.0, 1.2], (1.2, 1.5], (1.5, 2.0], (2.0, +inf)

    Returns a dict with keys like:
        ratio_pos/inf_0.2, ..., ratio_pos/gt_2.0  (fractions in [0,1])
        ratio_pos/avg (mean of ratio over masked & condition tokens)
    """
    mask = response_mask.bool()
    finite = torch.isfinite(ratio)
    mask = mask & finite

    edges = torch.tensor(bins, device=ratio.device, dtype=ratio.dtype)
    # bucketize indices for 8 original bins:
    # 0:(-inf,0.2], 1:(0.2,0.5], 2:(0.5,0.8], 3:(0.8,1.0], 4:(1.0,1.2], 5:(1.2,1.5], 6:(1.5,2.0], 7:(2.0,+inf)
    bin_idx = torch.bucketize(ratio, edges, right=True)

    def compute_for(cond: torch.Tensor):
        m = mask & cond
        # 9 bins now (insert eq_1.0 at index 4)
        counts = torch.zeros(len(bins) + 2, device=ratio.device, dtype=torch.float32)

        if m.any():
            eq1_mask = (torch.abs(ratio - 1.0) <= tol) & m
            not_eq1_mask = m & (~eq1_mask)

            if not_eq1_mask.any():
                idx = bin_idx[not_eq1_mask].reshape(-1).long()
                # shift indices >= 4 (i.e., > 1.0 side) by +1 to make room for eq_1.0 at index 4
                shift = (idx >= 4).long()
                idx = idx + shift
                counts.scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.float32))

            # put exact-1.0 counts at index 4
            counts[4] = eq1_mask.sum()

        total = counts.sum()
        frac = counts / (total + eps)

        # average ratio under this condition (masked)
        if m.any():
            avg = ratio[m].sum() / (m.sum() + eps)
        else:
            avg = torch.tensor(0.0, device=ratio.device, dtype=torch.float32)

        return frac, avg

    results = {}
    conditions = {
        "pos": advantages > 0,
        "neg": advantages < 0,
        "nonzero": advantages != 0
    }

    bin_names = [
        f"inf_{bins[0]}", f"{bins[0]}_{bins[1]}", f"{bins[1]}_{bins[2]}", f"{bins[2]}_{bins[3]}",
        "eq_1.0",
        f"{bins[3]}_{bins[4]}", f"{bins[4]}_{bins[5]}", f"{bins[5]}_{bins[6]}", f"gt_{bins[-1]}"
    ]

    for cond_name, cond_mask in conditions.items():
        frac, avg = compute_for(cond_mask)
        for i, bn in enumerate(bin_names):
            results[f"ratio_{cond_name}/{bn}"] = frac[i].item()
        results[f"ratio_{cond_name}/avg"] = float(avg.item())

    # ---- append: conditional KL means ----
    negative_approx_kl = log_prob - old_log_prob    # = log(ratio)
    approx_kl = -negative_approx_kl                 # PPO-style approx KL ≥ 0

    base_mask = response_mask.bool() & torch.isfinite(ratio) \
                & torch.isfinite(log_prob) & torch.isfinite(old_log_prob)

    m_neg_r_lt_1 = base_mask & (advantages < 0) & (ratio < (1.0 - tol))
    m_pos_r_gt_1 = base_mask & (advantages > 0) & (ratio > (1.0 + tol))

    def _mean_where(x: torch.Tensor, m: torch.Tensor):
        n = m.sum()
        if n.item() == 0:
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)
        return x[m].sum() / (n + eps)

    results["kl_neg_r_lt_1/mean"] = float(_mean_where(approx_kl, m_neg_r_lt_1).item())
    results["kl_pos_r_gt_1/mean"] = float(_mean_where(approx_kl, m_pos_r_gt_1).item())

    # optional diagnostics
    total_tokens = int(mask.sum().item())
    results["kl_neg_r_lt_1/count"] = int(m_neg_r_lt_1.sum().item())
    results["kl_pos_r_gt_1/count"] = int(m_pos_r_gt_1.sum().item())
    results["kl_neg_r_lt_1/frac_tokens"] = float((m_neg_r_lt_1.sum() / (mask.sum() + eps)).item()) if total_tokens > 0 else 0.0
    results["kl_pos_r_gt_1/frac_tokens"] = float((m_pos_r_gt_1.sum() / (mask.sum() + eps)).item()) if total_tokens > 0 else 0.0

    # ---- append: KL stats (flat with kl_stats/ prefix) ----

    conds = {
        "pos": advantages > 0,
        "neg": advantages < 0,
        "nonzero": advantages != 0,
    }

    def _mean_where(x: torch.Tensor, m: torch.Tensor):
        n = m.sum()
        if n.item() == 0:
            return torch.tensor(0.0, device=x.device, dtype=torch.float32)
        return x[m].sum() / (n + eps)

    for name, cmask in conds.items():
        m = base_mask & cmask
        results[f"kl_stats/{name}_abs_mean"]    = float(_mean_where(negative_approx_kl.abs(), m).item())
        results[f"kl_stats/{name}_sq_mean"]     = float(_mean_where(negative_approx_kl.pow(2), m).item())
        results[f"kl_stats/{name}_signed_mean"] = float(_mean_where(-negative_approx_kl, m).item())
        # results[f"kl_stats/{name}_approx_mean"] = float(_mean_where(approx_kl, m).item())

    return results


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=100.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    ratio_stats = get_ratio_stats(ratio, advantages, response_mask, log_prob, old_log_prob)
    # import pdb; pdb.set_trace()

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, ratio_stats


# we need it
def _solve_tau_from_sorted_delta2(sorted_delta2: torch.Tensor, target_sum: float) -> float:
    """
    Given sorted ascending values v_i = Δ_i^2 (i=0..n-1) and a target sum S,
    find τ^2 such that sum_i min(v_i, τ^2) = S.
    This uses a single pass over breakpoints without binary search.

    Returns:
        tau (float): sqrt(τ^2). If S >= sum(v_i), returns +inf (no clipping needed).
                     If S <= 0, returns 0.0 (clip everything to 0).
    """

    if sorted_delta2.numel() == 0:
        return 100000

    total = float(sorted_delta2.sum().item())
    if target_sum >= total - 1e-12: # no clipping needed
        return 100000
    if target_sum <= 1e-12: # clip everything to 0
        return 0.0

    csum = torch.cumsum(sorted_delta2, dim=0)  # prefix sums
    n = sorted_delta2.numel()

    for k in range(0,n):
        left_sum = float(csum[k].item())
        rest = n - k - 1
        m2 = sorted_delta2[k].item() - 1e-12
        if m2 * rest + left_sum >= target_sum - 1e-12:
            # print(f"================")
            # print(f"n: {n}, k: {k}, left_sum: {left_sum}, target_sum: {target_sum}")
            # print(f"sorted_delta2[k]: {sorted_delta2[k].item()}")
            # print(f"{list(zip(sorted_delta2[k-5:k+5].tolist(), csum[k-5:k+5].tolist()))}")
            # print((sorted_delta2 == 0).float().mean())
            # print(f"{target_sum}")
            if k == 0:
                return 0.0, csum[-1].item() / n
            else:
                M2_after = (sorted_delta2[k-1].item() * (rest + 1) + float(csum[k-1].item())) / n
                return float(sorted_delta2[k-1].item() - 1e-12) ** 0.5, M2_after

    return 100000

# we need it
def _get_trust_region_tokens_delta_sq(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
):
    mask = response_mask.bool()
    adv_example = advantages[:,0]
    pos_adv_mask = adv_example > 1e-12
    neg_adv_mask = adv_example < -1e-12

    delta = (old_log_prob - log_prob)             # Δ = log p_old - log p_new
    ratio = torch.exp(-delta)                     # r = exp(log_new - log_old)

    pos_adv_response_mask = mask[pos_adv_mask]
    neg_adv_response_mask = mask[neg_adv_mask]

    pos_adv_ratio = ratio[pos_adv_mask]
    neg_adv_ratio = ratio[neg_adv_mask]

    pos_adv_r_gt_1_mask = pos_adv_ratio > 1.0 + 1e-12
    neg_adv_r_lt_1_mask = neg_adv_ratio < 1.0 - 1e-12

    delta_sq = delta.pow(2)
    pos_adv_harm_tokens_delta_sq = delta_sq[pos_adv_mask][pos_adv_r_gt_1_mask & pos_adv_response_mask]
    neg_adv_harm_tokens_delta_sq = delta_sq[neg_adv_mask][neg_adv_r_lt_1_mask & neg_adv_response_mask]

    tr_tokens_delta_sq = torch.cat([pos_adv_harm_tokens_delta_sq, neg_adv_harm_tokens_delta_sq])

    return tr_tokens_delta_sq

# we need it
def kpo_clip_harmful_tokens(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    KL2_budget: float = None
):
    """
    Decide global clip scalars (clip_low, clip_high) under an M2 budget.

    Policy:
      - Consider only harmful tokens: (A>0 & r>1) or (A<0 & r<1), where r = exp(log_new - log_old).
      - Sort harmful tokens by delta^2 = (log p_old - log p_new)^2 ascending.
      - Find a single threshold τ so that capping |delta| at τ across harmful tokens
        yields overall M2 <= KL2_budget.
      - Map τ to two global ratio bounds:
            clip_low  = exp(-τ)  (applies to adv<0 & r<1)
            clip_high = exp(+τ)  (applies to adv>0 & r>1)
      - Non-harmful quadrants are not constrained by these bounds.

    Returns:
      clip_low  (float): lower clamp for tokens with (adv<0 & r<1)
      clip_high (float): upper clamp for tokens with (adv>0 & r>1)
    """
    assert KL2_budget is not None, "KL2_budget must be set."

    tr_tokens_delta_sq = _get_trust_region_tokens_delta_sq(old_log_prob, log_prob, advantages, response_mask)
    token_num = tr_tokens_delta_sq.numel()

    if token_num == 0: # no clipping needed
        return 0.0, 100000, 0.0, 0.0

    target_total = KL2_budget * float(token_num)
    M2_now = float(tr_tokens_delta_sq.sum().detach().item() / token_num)

    if M2_now <= KL2_budget + 1e-12:
        # No clipping needed -> effectively no constraint
        return 0.0, 100000, M2_now, M2_now

    print(f"tr-M2_now: {M2_now}")
    print(f"KL2_budget: {KL2_budget}")

    # import pdb; pdb.set_trace()

    sorted_delta2, _ = torch.sort(tr_tokens_delta_sq)  # ascending
    tau, M2_after = _solve_tau_from_sorted_delta2(sorted_delta2, target_total)

    # Map |Δ|<=τ to ratio bounds per quadrant
    clip_low = float(torch.exp(torch.tensor(-tau)).item())   # applies to (adv<0, r<1)
    clip_high = float(torch.exp(torch.tensor(+tau)).item())  # applies to (adv>0, r>1)

    return clip_low, clip_high, M2_now, M2_after


def compute_m2po_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    M2_budget: float = None,
    miniclip_low: float = 0.3,
    miniclip_high: float = 0.5,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute policy loss under an M2 (KL^2) budget using per-token clipping bounds.

    Steps:
      1) Get per-token (clip_low, clip_high) from kpo_clip.
      2) Compute ratio and apply element-wise clamp.
      3) Compute surrogate loss -A * ratio_clipped and aggregate.

    Returns:
      pg_loss:       aggregated policy loss
      stats:         dict with basic diagnostics (M2 before/after, fractions)
      clip_low/high: the per-token bounds actually used
    """

    clip_low, clip_high, M2_data, M2_after = kpo_clip_harmful_tokens(old_log_prob, log_prob, advantages, response_mask, M2_budget)

    clip_low = 1 - clip_low
    clip_high = clip_high - 1
    print(f"clip_low: {clip_low}, clip_high: {clip_high}")
    if miniclip_low is not None and clip_low < miniclip_low:
        clip_low = miniclip_low
    if miniclip_high is not None and clip_high < miniclip_high:
        clip_high = miniclip_high

    # ratio = exp(log_new - log_old)
    ratio = torch.exp(log_prob - old_log_prob)
    ppo_kl = verl_F.masked_mean(-(log_prob - old_log_prob), response_mask)

    ratio_stats = get_ratio_stats(ratio, advantages, response_mask, log_prob, old_log_prob)

    ##### clip
    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - clip_low, 1 + clip_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_loss = agg_loss(loss_mat=clip_pg_losses1, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)


    ratio_stats["m2po/clip_low"] = clip_low
    ratio_stats["m2po/clip_high"] = clip_high
    ratio_stats["m2po/M2"] = M2_data
    ratio_stats["m2po/M2_after"] = M2_after
    ratio_stats["m2po/M2_budget"] = M2_budget

    return pg_loss, pg_clipfrac, ppo_kl, (ppo_kl - ppo_kl), ratio_stats


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
