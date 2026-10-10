"""Metric depth DR with persistent calibration, coherent holes and side occlusion.

Ranges are project-specific proposals inspired by Parkour in the Wild, not a
reproduction of its camera parameters. Perfect reference frames never enter DR.
"""
import torch
from torch.nn import functional as F


# scale, bias(m), local sigma(m), edge probability, hole min/max,
# side probability/max columns, blur sigma min/max. Level 0 is exact clean.
PROFILES = (
    (0., 0., 0., 0., 0., 0., 0., 0., 0., 0.),
    (.005, .005, .005, .10, 0., .03, .30, 2., .30, .50),
    (.010, .010, .010, .25, .03, .10, .50, 4., .40, .70),
    (.015, .015, .015, .40, .10, .20, .70, 6., .50, 1.),
    # Control-only midpoint of light and medium; preserve auxiliary level IDs.
    (.0075, .0075, .0075, .175, .015, .065, .40, 3., .35, .60),
)


class StructuredDepthDR:
    def __init__(self, count, height, width, device, *, near=0., far=3.):
        self.height, self.width, self.near, self.far = height, width, near, far
        self.profiles = torch.tensor(PROFILES, device=device)
        self.level = torch.full((count,), -1, dtype=torch.long, device=device)
        self.gain = torch.zeros(count, device=device)
        self.bias = torch.zeros_like(self.gain)
        self.phase = torch.zeros_like(self.gain)
        self.duration = torch.ones_like(self.gain)
        self.last_time = torch.zeros_like(self.gain)
        # Separate fields for local measurement bias, holes and side boundaries.
        self.field = torch.zeros(count, 3, 5, 7, device=device)
        self.target = torch.zeros_like(self.field)
        self.sides = torch.zeros(count, 2, device=device)
        self.side_target = torch.zeros_like(self.sides)
        self.holes = torch.zeros_like(self.gain)
        self.hole_target = torch.zeros_like(self.gain)
        self.blur_sigma = torch.ones_like(self.gain)

    def reset_envs(self, mask):
        self.level[mask] = -1  # Calibration is resampled on the next frame.

    def _sides(self, params):
        active = torch.rand(len(params), 2, device=params.device) < params[:, 6:7]
        widths = 1+torch.floor(torch.rand_like(active.float())*params[:, 7:8])
        return active*widths

    @torch.no_grad()
    def apply(self, image, ids, levels, time):
        """One fresh frame per selected env; held frames are never re-noised."""
        levels = levels.to(device=image.device, dtype=torch.long)
        params = self.profiles[levels]
        changed = self.level[ids] != levels
        fresh_ids, fresh_params = ids[changed], params[changed]
        if len(fresh_ids):
            self.level[fresh_ids] = levels[changed]
            self.gain[fresh_ids] = (2*torch.rand_like(fresh_ids.float())-1)*fresh_params[:, 0]
            self.bias[fresh_ids] = (2*torch.rand_like(fresh_ids.float())-1)*fresh_params[:, 1]
            self.field[fresh_ids] = torch.randn_like(self.field[fresh_ids])
            self.target[fresh_ids] = torch.randn_like(self.target[fresh_ids])
            self.sides[fresh_ids] = self._sides(fresh_params)
            self.side_target[fresh_ids] = self._sides(fresh_params)
            self.holes[fresh_ids] = fresh_params[:, 4]+(fresh_params[:, 5]-fresh_params[:, 4])*torch.rand_like(fresh_ids.float())
            self.hole_target[fresh_ids] = fresh_params[:, 4]+(fresh_params[:, 5]-fresh_params[:, 4])*torch.rand_like(fresh_ids.float())
            self.blur_sigma[fresh_ids] = fresh_params[:, 8]+(fresh_params[:, 9]-fresh_params[:, 8])*torch.rand_like(fresh_ids.float())
            self.phase[fresh_ids] = 0
            self.duration[fresh_ids] = .5+1.5*torch.rand_like(fresh_ids.float())
            self.last_time[fresh_ids] = time
        dt = (time-self.last_time[ids]).clamp_min(0)
        self.last_time[ids] = time
        self.phase[ids] += dt/self.duration[ids]
        evolved = self.phase[ids] >= 1
        evolved_ids = ids[evolved]
        if len(evolved_ids):
            self.field[evolved_ids] = self.target[evolved_ids]
            self.target[evolved_ids] = torch.randn_like(self.target[evolved_ids])
            self.sides[evolved_ids] = self.side_target[evolved_ids]
            self.side_target[evolved_ids] = self._sides(params[evolved])
            self.holes[evolved_ids] = self.hole_target[evolved_ids]
            self.hole_target[evolved_ids] = params[evolved, 4]+(params[evolved, 5]-params[evolved, 4])*torch.rand_like(evolved_ids.float())
            self.phase[evolved_ids] = 0
            self.duration[evolved_ids] = .5+1.5*torch.rand_like(evolved_ids.float())
        phase = self.phase[ids].clamp(0, 1)
        blend = phase.square()*(3-2*phase)
        fields = self.field[ids].lerp(self.target[ids], blend[:, None, None, None])
        fields = F.interpolate(fields, (self.height, self.width), mode='bilinear', align_corners=False)
        local = fields[:, 0]
        local = (local-local.mean((1, 2), keepdim=True))/local.std((1, 2), keepdim=True).clamp_min(1e-6)
        depth = (image[..., 0]+.5)*(self.far-self.near)+self.near
        # A physical near-field blind zone, independent of the random missing
        # pixel budget. Clean views remain perfect, including near surfaces.
        valid = (image[..., 0] < .5) & ((depth >= .15) | (levels == 0)[:, None, None])
        multiplier = image.new_tensor((0., .5, 1., 1.5, .75))[levels]
        sigma = multiplier[:, None, None]*(.003+.002*depth.square())
        corrupted = depth*(1+self.gain[ids, None, None])+self.bias[ids, None, None]
        corrupted += local*params[:, 2, None, None]+torch.randn_like(depth)*sigma

        # Detect edges on the perfect metric image, not on noisy estimates.
        jump = torch.zeros_like(depth)
        dx = (depth[:, :, 1:]-depth[:, :, :-1]).abs()
        dy = (depth[:, 1:]-depth[:, :-1]).abs()
        vx = valid[:, :, 1:] & valid[:, :, :-1]
        vy = valid[:, 1:] & valid[:, :-1]
        jump[:, :, 1:] = torch.maximum(jump[:, :, 1:], dx*vx)
        jump[:, :, :-1] = torch.maximum(jump[:, :, :-1], dx*vx)
        jump[:, 1:] = torch.maximum(jump[:, 1:], dy*vy)
        jump[:, :-1] = torch.maximum(jump[:, :-1], dy*vy)
        edge = jump > torch.maximum(depth*.03, depth.new_tensor(.05))
        edge = F.max_pool2d(edge[:, None].float(), 3, 1, 1)[:, 0].bool()
        broken = edge & (torch.rand_like(depth) < params[:, 3, None, None])
        remove = broken & (torch.rand_like(depth) < .5)
        y = torch.arange(self.height, device=image.device)[None, :, None]
        x = torch.arange(self.width, device=image.device)[None, None, :]
        ny = (y+torch.randint(-1, 2, depth.shape, device=image.device)).clamp(0, self.height-1)
        nx = (x+torch.randint(-1, 2, depth.shape, device=image.device)).clamp(0, self.width-1)
        neighbor = (ny*self.width+nx).flatten(1)
        neighbor_valid = valid.flatten(1).gather(1, neighbor).reshape_as(valid)
        neighbor_depth = corrupted.flatten(1).gather(1, neighbor).reshape_as(depth)
        corrupted = torch.where(broken & ~remove & neighbor_valid, neighbor_depth, corrupted)
        remove |= broken & ~neighbor_valid

        fraction = self.holes[ids].lerp(self.hole_target[ids], blend)
        # Rank the smooth field for bounded coverage. No independent rectangular
        # masks; the underlying spatial pattern persists and evolves slowly.
        field_rank = fields[:, 1].flatten(1).argsort(1).argsort(1)
        holes = (field_rank < (fraction*self.height*self.width)[:, None]).reshape_as(valid)
        widths = self.sides[ids].lerp(self.side_target[ids], blend[:, None])
        jagged = .4*fields[:, 2].mean(-1, keepdim=True).tanh()
        maximum = params[:, 7, None, None]
        left_width = torch.minimum((widths[:, 0, None, None]+jagged).clamp_min(0), maximum)
        right_width = torch.minimum((widths[:, 1, None, None]-jagged).clamp_min(0), maximum)
        left = (widths[:, 0, None, None] > 0) & (x+.5 < left_width)
        right = (widths[:, 1, None, None] > 0) & (x+.5 >= self.width-right_width)
        missing = (remove | holes | left | right) & valid
        # Bound additional random misses to 35% of measurable valid pixels.
        score = fields[:, 2].sigmoid()+missing*2
        rank = score.flatten(1).argsort(1, descending=True).argsort(1)
        budget = torch.floor(.35*valid.sum((1, 2))).long()
        missing &= (rank < budget[:, None]).reshape_as(valid)
        valid_out = valid & ~missing

        # Validity-aware Gaussian blur: missing measurements cannot become a
        # fictitious near/far surface through interpolation.
        blur_sigma = self.blur_sigma[ids]
        axis = torch.arange(-2, 3, device=image.device, dtype=image.dtype)
        kernel = torch.exp(-axis[None].square()/(2*blur_sigma.clamp_min(.1)[:, None].square()))
        kernel /= kernel.sum(1, keepdim=True)
        horizontal = kernel[:, None, None, :]
        vertical = kernel[:, None, :, None]
        # Separable grouped filters avoid a 25x image-size unfold allocation.
        numerator = (corrupted*valid_out)[None]
        denominator = valid_out.float()[None]
        numerator = F.conv2d(numerator, horizontal, padding=(0, 2), groups=len(ids))
        denominator = F.conv2d(denominator, horizontal, padding=(0, 2), groups=len(ids))
        numerator = F.conv2d(numerator, vertical, padding=(2, 0), groups=len(ids))[0]
        denominator = F.conv2d(denominator, vertical, padding=(2, 0), groups=len(ids))[0]
        blurred = numerator/denominator.clamp_min(1e-6)
        result = (blurred.clamp(self.near, self.far)-self.near)/(self.far-self.near)-.5
        result = torch.where(valid_out, result, .5)[..., None]
        return torch.where((levels == 0)[:, None, None, None], image, result)
