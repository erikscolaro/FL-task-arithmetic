import torch
from torch.optim import Optimizer

class SparseSGDM(Optimizer):
    r"""
    Implements Stochastic Gradient Descent with Momentum (SGDM)
    where the gradient mask is passed directly into the constructor.
    """

    def __init__(self, params, masks=None, lr=1e-3, momentum=0.0, dampening=0.0,
                 weight_decay=0.0, nesterov=False):
        """
        Args:
            params (iterable): iterable of parameters to optimize (usually model.parameters())
            masks (iterable, optional): iterable of (mask, param) tuples.
                                        (in case we use the TaLoS masked_parameters(model))
            lr (float): learning rate
            momentum (float, optional): momentum factor (default: 0)
            weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
            dampening (float, optional): dampening for momentum (default: 0)
            nesterov (bool, optional): enables Nesterov momentum (default: False)
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        # 1. Initialize the Base Optimizer with the parameters
        super(SparseSGDM, self).__init__(params, defaults)

        # 2. Automatically register masks if provided in constructor
        if masks is not None:
            self._register_masks(masks)

    def _register_masks(self, masked_parameters):
        """
        Internal helper to populate self.state with masks.
        """
        for mask, param in masked_parameters:
            # Initialize state if it implies lazy initialization
            if param not in self.state:
                self.state[param] = {}

            # Store mask in the specific parameter's state dictionary
            # We ensure device consistency (move mask to CPU/GPU of the param)
            self.state[param]['mask'] = mask.to(param.device)

    @torch.no_grad()
    def step(self, closure=None): #type: ignore
        """Performs a single optimization step with Masking."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                # the name d_p resembles "delta p" of the gradient
                d_p = p.grad

                # Retrieve mask from state
                state = self.state[p]
                mask = state.get('mask')

                # 1. Apply Weight Decay (Before masking)
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                # 2. MASKING GRADIENT: Prevent update from current batch
                if mask is not None:
                    d_p.mul_(mask)

                # 3. Momentum Logic
                if momentum != 0:
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                    # 4. MASKING MOMENTUM: Prevent drift from history
                    if mask is not None:
                        buf.mul_(mask)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                # 5. Update Weight
                p.add_(d_p, alpha=-group['lr'])

        return loss