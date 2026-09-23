import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class GatedMlp(nn.Module):
    """Gated MLP in the SwiGLU style."""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.gate_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.up_proj = nn.Linear(in_features, hidden_features, bias=False)
        self.down_proj = nn.Linear(hidden_features, out_features, bias=False)
        self.act_fn = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        x = self.drop(x)
        x = self.down_proj(x)
        return x

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        """
        num_experts: int
        gates: B*L x Experts
        """
        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self._nonzero_gates)
        # Match the stitched output dtype.
        zeros = torch.zeros(self._gates.size(0), expert_out[-1].size(1), requires_grad=True,
                            device=stitched.device, dtype=stitched.dtype)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched)
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    routed_num_experts: an integer - number of experts
    shared_num_experts: an integer - number of shared experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, output_size, hidden_size,
                 routed_num_experts, shared_num_experts=1,
                 noisy_gating=True, k=2):
        super(MoE, self).__init__()
        self.noisy_gating = noisy_gating
        self.routed_num_experts = routed_num_experts
        self.shared_num_experts = shared_num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.k = k
        # instantiate experts
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.shared_experts = GatedMlp(in_features=input_size, hidden_features=self.shared_num_experts*hidden_size,
                                       out_features=output_size, act_layer=approx_gelu, drop=0)
        self.routed_experts = nn.ModuleList([GatedMlp(in_features=input_size, hidden_features=hidden_size, act_layer=approx_gelu, drop=0) for _ in range(self.routed_num_experts)])
        self.w_gate = nn.Parameter(torch.zeros(input_size, self.routed_num_experts), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(input_size, self.routed_num_experts), requires_grad=True)

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

        assert(self.k <= self.routed_num_experts)

    def cv_squared(self, x):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.
        Args:
        x: a `Tensor`.
        Returns:
        a `Scalar`.
        """
        eps = 1e-10
        # if only routed_num_experts = 1

        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def _gates_to_load(self, gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
        gates: a `Tensor` of shape [batch_size, n]
        Returns:
        a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.
        This gives us a way of backpropagating from a loss that balances the number
        of times each expert is in the top k experts per example.
        In the case of no noise, pass in None for noise_stddev, and the result will
        not be differentiable.
        Args:
        clean_values: a `Tensor` of shape [batch, n].
        noisy_values: a `Tensor` of shape [batch, n].  Equal to clean values plus
          normally distributed noise with standard deviation noise_stddev.
        noise_stddev: a `Tensor` of shape [batch, n], or None
        noisy_top_values: a `Tensor` of shape [batch, m].
           "values" Output of tf.top_k(noisy_top_values, m).  m >= k+1
        Returns:
        a `Tensor` of shape [batch, n].
        """
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)
        # is each value currently in the top k.
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """Noisy top-k gating.
          See paper: https://arxiv.org/abs/1701.06538.
          Args:
            x: input Tensor with shape [batch_size, input_size]
            train: a boolean - we only add noise at training time.
            noise_epsilon: a float
          Returns:
            gates: a Tensor with shape [batch_size, routed_num_experts]
            load: a Tensor with shape [routed_num_experts]
        """
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.routed_num_experts), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates.to(zeros.dtype))

        if self.noisy_gating and self.k < self.routed_num_experts and train:
            load = (self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        """Args:
        x: B x L x D
        loss_coef: a scalar - multiplier on load-balancing losses

        Returns:
        y: a tensor with shape [batch_size, output_size].
        """
        shape = x.shape
        x = x.reshape(-1, self.input_size)
        # shared experts
        z = self.shared_experts(x)

        # routed experts
        gates, load = self.noisy_top_k_gating(x, self.training)

        # calculate importance loss
        importance = gates.sum(0)
        self.aux_loss = self.cv_squared(importance) + self.cv_squared(load)
        self.aux_loss *= loss_coef

        dispatcher = SparseDispatcher(self.routed_num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        gates = dispatcher.expert_to_gates()
        expert_outputs = [self.routed_experts[i](expert_inputs[i]) for i in range(self.routed_num_experts)]
        y = dispatcher.combine(expert_outputs)
        return (y + z).reshape(*shape)


class RegionalPriorMoE(nn.Module):
    """MoE with hard routing by region-map class.

    Each token is processed by the expert of its region-map class,
    plus a shared expert that handles cross-region and global processes.
    No learned gating — routing is purely structural.

    Args:
        input_size: feature dimension (D)
        hidden_size: expert hidden dimension
        output_size: output dimension (usually == input_size)
        num_regions: number of region-map classes
    """
    def __init__(self, input_size, output_size, hidden_size, num_regions):
        super().__init__()
        self.num_regions = num_regions
        approx_gelu = lambda: nn.GELU(approximate="tanh")

        self.region_experts = nn.ModuleList([
            GatedMlp(in_features=input_size, hidden_features=hidden_size,
                     out_features=output_size, act_layer=approx_gelu, drop=0)
            for _ in range(num_regions)
        ])
        self.shared_expert = GatedMlp(
            in_features=input_size, hidden_features=hidden_size,
            out_features=output_size, act_layer=approx_gelu, drop=0)

    def forward(self, x, region_labels):
        """
        Args:
            x: (B, L, D) token features
            region_labels: (B, L) region-map class id in [0, num_regions)
        Returns:
            output: (B, L, D) routed + shared expert output
        """
        B, L, D = x.shape
        output = torch.zeros_like(x)

        # Hard routing: each token goes to its region's expert
        for r in range(self.num_regions):
            mask = (region_labels == r)
            if not mask.any():
                continue
            output[mask] = self.region_experts[r](x[mask]).to(output.dtype)

        # Shared expert handles all tokens
        shared_out = self.shared_expert(x.reshape(-1, D)).reshape(B, L, D)
        return output + shared_out


class SoftRegionalPriorMoE(nn.Module):
    """MoE with learned top-k gating and a region-to-expert prior.

    Gating combines two signals:
      1. token → expert (w_gate): data-driven routing
      2. region → expert (region_bias): a fixed region-map prior refined by
         training.

    Final logits = x @ w_gate + exp(bias_logit_scale) × region_bias[region_labels]

    bias_logit_scale is a learnable scalar controlling the strength of the
    region prior. Initialised so exp(scale) = prior_scale_init (default 1.0).
    Training will adjust it up or down — >1 means the prior is amplified,
    <1 means it's attenuated, ~0 means it's ignored.

    Args:
        input_size: feature dimension (D)
        output_size: output dimension (usually == input_size)
        hidden_size: expert hidden dimension
        routed_num_experts: number of routed experts
        num_regions: number of region-map classes
        shared_num_experts: shared expert width multiplier
        prior_scale_init: initial value of exp(bias_logit_scale) (default 1.0)
        noisy_gating: add noise for load-balancing gradient
        k: number of top experts per token
        use_mlp_gate: replace Linear gate with MLP (Linear→GELU→Linear)
        use_weight_separation: separate expert selection (prior+data) from weighting (data only)
        use_gating_conv: add depthwise conv 3×3 residual before routing
        img_size: (H, W) patch grid, required when use_gating_conv=True
    """
    def __init__(self, input_size, output_size, hidden_size,
                 routed_num_experts, num_regions,
                 shared_num_experts=1,
                 prior_scale_init=1.0,
                 noisy_gating=True, k=2,
                 use_mlp_gate=False,
                 use_weight_separation=False,
                 use_gating_conv=False,
                 img_size=None,
                 region_prior_init=None,
                 use_region_prior=True):
        super().__init__()
        self.input_size = input_size
        self.use_region_prior = use_region_prior
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.routed_num_experts = routed_num_experts
        self.num_regions = num_regions
        self.noisy_gating = noisy_gating
        self.k = k
        self.use_mlp_gate = use_mlp_gate
        self.use_weight_separation = use_weight_separation
        self.use_gating_conv = use_gating_conv
        self.img_size = img_size
        self.region_prior_init = region_prior_init

        approx_gelu = lambda: nn.GELU(approximate="tanh")

        # Shared expert (processes all tokens)
        self.shared_experts = GatedMlp(
            in_features=input_size,
            hidden_features=shared_num_experts * hidden_size,
            out_features=output_size, act_layer=approx_gelu, drop=0)

        # Routed experts
        self.routed_experts = nn.ModuleList([
            GatedMlp(in_features=input_size, hidden_features=hidden_size,
                     out_features=output_size, act_layer=approx_gelu, drop=0)
            for _ in range(routed_num_experts)
        ])

        # --- Gating parameters ---
        # Data-driven: token features → expert logits
        self.w_gate = nn.Parameter(torch.zeros(input_size, routed_num_experts))
        self.w_noise = nn.Parameter(torch.zeros(input_size, routed_num_experts))

        # Optional: MLP gate (replaces x @ w_gate)
        if self.use_mlp_gate:
            self.gate_mlp = nn.Sequential(
                nn.Linear(input_size, input_size // 2),
                nn.GELU(),
                nn.Linear(input_size // 2, routed_num_experts, bias=False))
            nn.init.zeros_(self.gate_mlp[-1].weight)

        # Optional: Depthwise conv for spatial context
        if self.use_gating_conv:
            assert self.img_size is not None, \
                "img_size=(H, W) required when use_gating_conv=True"
            self.gate_conv = nn.Conv2d(
                input_size, input_size, 3, padding=1, groups=input_size)

        # Region prior: region-map class → expert bias (learned)
        self.region_bias = nn.Parameter(torch.zeros(num_regions, routed_num_experts))
        # Learnable scale: how much the prior contributes to routing logits
        # Keep this parameter float32 for FSDP flattening.
        self.bias_logit_scale = nn.Parameter(
            torch.tensor(np.log(max(prior_scale_init, 1e-6))).float())

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.routed_num_experts

        if region_prior_init is not None:
            # Initialize from the supplied region-to-expert prior.
            with torch.no_grad():
                init_t = torch.as_tensor(region_prior_init, dtype=self.region_bias.dtype)
                if init_t.shape != self.region_bias.shape:
                    raise ValueError(
                        f"region_prior_init shape {tuple(init_t.shape)} != "
                        f"region_bias shape {tuple(self.region_bias.shape)}")
                self.region_bias.copy_(init_t)
        else:
            # Fall back to latitude-band initialization.
            self._init_region_bias()

    def _init_region_bias(self):
        """Initialise the region prior using a latitudinal-band structure.

        The bundled nine-region map is ordered from north to south.

        We use a band-diagonal matrix: each region prefers its own expert
        and shares partial preference with adjacent regions, encoding the
        spatial structure of the region map.
        """
        with torch.no_grad():
            self.region_bias.zero_()
            for r in range(min(self.num_regions, self.routed_num_experts)):
                self.region_bias[r, r] = 1.0           # own expert
                if r > 0:
                    self.region_bias[r, r-1] = 0.5     # north neighbour
                if r < self.routed_num_experts - 1:
                    self.region_bias[r, r+1] = 0.5     # south neighbour

    def forward(self, x, region_labels):
        """Forward pass.

        Args:
            x: (B, L, D) token features
            region_labels: (B, L) region-map class id in [0, num_regions)

        Returns:
            output: (B, L, D) routed + shared expert output
        """
        shape = x.shape
        x_flat = x.reshape(-1, self.input_size)  # (B*L, D)

        # Shared expert
        z = self.shared_experts(x_flat)

        # === Optional: Spatial context (depthwise conv residual) ===
        if self.use_gating_conv:
            B, L, D = shape
            H, W = self.img_size
            x_img = x.reshape(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
            x_conv = self.gate_conv(x_img)                      # (B, D, H, W)
            x_conv = x_conv.permute(0, 2, 3, 1).reshape(B, L, D)
            x = x + x_conv
            x_flat = x.reshape(-1, self.input_size)

        # === Routing logits ===
        # Data-driven term
        if self.use_mlp_gate:
            data_logits = self.gate_mlp(x_flat)  # (B*L, num_experts)
        else:
            data_logits = x_flat @ self.w_gate   # original: clean_logits

        # Region-prior routing term.
        logits = data_logits
        if self.use_region_prior:
            region_labels_flat = region_labels.reshape(-1)
            region_logits = self.region_bias[region_labels_flat]  # (B*L, num_experts)
            bias_scale = torch.exp(self.bias_logit_scale)
            logits = data_logits + bias_scale * region_logits     # combined for selection

        # Noisy gating (training only)
        if self.noisy_gating and self.training:
            raw_noise_stddev = x_flat @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + 1e-2
            logits = logits + torch.randn_like(logits) * noise_stddev

        # === Top-k (optional: weight separation) ===
        topk_val = min(self.k + 1, self.routed_num_experts)
        top_logits, top_indices = logits.topk(topk_val, dim=1)

        if self.use_weight_separation:
            # Weight separation: prior+data selects experts, only data sets weights
            top_k_indices = top_indices[:, :self.k]
            weights = self.softmax(data_logits.gather(1, top_k_indices))
        else:
            # Original: combined logits for both selection and weighting
            top_k_logits = top_logits[:, :self.k]
            top_k_indices = top_indices[:, :self.k]
            weights = self.softmax(top_k_logits)

        zeros = torch.zeros_like(data_logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, weights.to(zeros.dtype))

        # Dispatch to experts
        dispatcher = SparseDispatcher(self.routed_num_experts, gates)
        expert_inputs = dispatcher.dispatch(x_flat)
        expert_gates = dispatcher.expert_to_gates()
        expert_outputs = [
            self.routed_experts[i](expert_inputs[i])
            for i in range(self.routed_num_experts)
        ]
        y = dispatcher.combine(expert_outputs)

        return (y + z).reshape(*shape)
