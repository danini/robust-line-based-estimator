import os
import numpy as np
from sklearn.cluster import DBSCAN
import torch
from torch import nn
import torch.utils.checkpoint
from copy import deepcopy
from omegaconf import OmegaConf


from .superglue_endpoints import SuperPoint, sample_descriptors


### Wireframe pre-processing

def lines_to_wireframe(lines, line_scores, all_descs, conf):
    """ Given a set of lines, their score and dense descriptors,
        merge close-by endpoints and compute a wireframe defined by
        its junctions and connectivity.
    Returns:
        junctions: list of [num_junc, 2] tensors listing all wireframe junctions
        junc_scores: list of [num_junc] tensors with the junction score
        junc_descs: list of [dim, num_junc] tensors with the junction descriptors
        connectivity: list of [num_junc, num_junc] bool arrays with True when 2 junctions are connected
        new_lines: the new set of [b_size, num_lines, 2, 2] lines
        lines_junc_idx: a [b_size, num_lines, 2] tensor with the indices of the junctions of each endpoint
        num_true_junctions: a list of the number of valid junctions for each image in the batch,
                            i.e. before filling with random ones
    """
    b_size, _, h, w = all_descs.shape
    device = lines.device
    h, w = h * 8, w * 8
    endpoints = lines.reshape(b_size, -1, 2)

    (junctions, junc_scores, junc_descs, connectivity, new_lines,
     lines_junc_idx, num_true_junctions) = [], [], [], [], [], [], []
    for bs in range(b_size):
        # Cluster the junctions that are close-by
        db = DBSCAN(eps=conf.nms_radius, min_samples=1).fit(
            endpoints[bs].cpu().numpy())
        clusters = db.labels_
        n_clusters = len(set(clusters))
        num_true_junctions.append(n_clusters)

        # Compute the average junction and score for each cluster
        clusters = torch.tensor(clusters, dtype=torch.long,
                                device=device)
        new_junc = torch.zeros(n_clusters, 2, dtype=torch.float,
                               device=device)
        new_junc.scatter_reduce_(0, clusters[:, None].repeat(1, 2),
                                 endpoints[bs], reduce='mean',
                                 include_self=False)
        junctions.append(new_junc)
        new_scores = torch.zeros(n_clusters, dtype=torch.float, device=device)
        new_scores.scatter_reduce_(
            0, clusters, torch.repeat_interleave(line_scores[bs], 2),
            reduce='mean', include_self=False)
        junc_scores.append(new_scores)

        # Compute the new lines
        new_lines.append(junctions[-1][clusters].reshape(-1, 2, 2))
        lines_junc_idx.append(clusters.reshape(-1, 2))

        if conf.force_num_junctions:
            # Add random junctions (with no connectivity)
            missing = conf.max_n_junctions - len(junctions[-1])
            junctions[-1] = torch.cat(
                [junctions[-1], torch.rand(missing, 2).to(lines)
                 * lines.new_tensor([[w - 1, h - 1]])], dim=0)
            junc_scores[-1] = torch.cat([
                junc_scores[-1], torch.zeros(missing).to(lines)], dim=0)

            junc_connect = torch.eye(
                conf.max_n_junctions, dtype=torch.bool, device=device)
            pairs = clusters.reshape(-1, 2)  # these pairs are connected by a line
            junc_connect[pairs[:, 0], pairs[:, 1]] = True
            junc_connect[pairs[:, 1], pairs[:, 0]] = True
            connectivity.append(junc_connect)
        else:
            # Compute the junction connectivity
            junc_connect = torch.eye(n_clusters, dtype=torch.bool,
                                     device=device)
            pairs = clusters.reshape(-1, 2)  # these pairs are connected by a line
            junc_connect[pairs[:, 0], pairs[:, 1]] = True
            junc_connect[pairs[:, 1], pairs[:, 0]] = True
            connectivity.append(junc_connect)

        # Interpolate the new junction descriptors
        junc_descs.append(sample_descriptors(
            junctions[-1][None], all_descs[bs:(bs+1)], 8)[0])

    new_lines = torch.stack(new_lines, dim=0)
    lines_junc_idx = torch.stack(lines_junc_idx, dim=0)
    return (junctions, junc_scores, junc_descs, connectivity,
            new_lines, lines_junc_idx, num_true_junctions)


### GlueStick backbone

class StitchedWireframe(nn.Module):
    default_conf = {
        'descriptor_dim': 256,
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'num_line_iterations': 1,
        'output_normalization': 'double_softmax',
        'num_sinkhorn_iterations': 50,
        'filter_threshold': 0.2,
    }
    required_data_keys = [
        'keypoints0', 'keypoints1',
        'descriptors0', 'descriptors1',
        'keypoint_scores0', 'keypoint_scores1',
        'lines_junc_idx0', 'lines_junc_idx1',
        'lines0', 'lines1', 'line_scores0', 'line_scores1']

    DEFAULT_LOSS_CONF = {'nll_weight': 1., 'nll_balancing': 0.5, 'reward_weight': 0., 'bottleneck_l2_weight': 0.}

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.create({**self.default_conf, **conf})

        self.kenc = KeypointEncoder(self.conf.descriptor_dim,
                                    self.conf.keypoint_encoder)
        self.lenc = EndPtEncoder(self.conf.descriptor_dim, self.conf.keypoint_encoder)
        self.gnn = AttentionalGNN(self.conf.descriptor_dim, self.conf.GNN_layers)
        self.final_proj = nn.Conv1d(self.conf.descriptor_dim, self.conf.descriptor_dim,
                                    kernel_size=1)
        nn.init.constant_(self.final_proj.bias, 0.0)
        nn.init.orthogonal_(self.final_proj.weight, gain=1)
        self.final_line_proj = nn.Conv1d(
            self.conf.descriptor_dim, self.conf.descriptor_dim, kernel_size=1)
        nn.init.constant_(self.final_line_proj.bias, 0.0)
        nn.init.orthogonal_(self.final_line_proj.weight, gain=1)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        line_bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('line_bin_score', line_bin_score)

    def forward(self, data):
        device = data['lines0'].device
        b_size = len(data['lines0'])

        pred = {}
        desc0, desc1 = data['descriptors0'], data['descriptors1']
        kpts0, kpts1 = data['keypoints0'], data['keypoints1']

        n_kpts0, n_kpts1 = kpts0.shape[1], kpts1.shape[1]
        n_lines0, n_lines1 = data['lines0'].shape[1], data['lines1'].shape[1]
        if n_kpts0 == 0 or n_kpts1 == 0:
            # No detected keypoints nor lines
            pred['log_assignment'] = torch.zeros(
                b_size, n_kpts0, n_kpts1, dtype=torch.float, device=device)
            pred['matches0'] = torch.full(
                (b_size, n_kpts0), -1, device=device, dtype=torch.int64)
            pred['matches1'] = torch.full(
                (b_size, n_kpts1), -1, device=device, dtype=torch.int64)
            pred['match_scores0'] = torch.zeros(
                (b_size, n_kpts0), device=device, dtype=torch.float32)
            pred['match_scores1'] = torch.zeros(
                (b_size, n_kpts1), device=device, dtype=torch.float32)
            pred['line_log_assignment'] = torch.zeros(b_size, n_lines0, n_lines1,
                                        dtype=torch.float, device=device)
            pred['line_matches0'] = torch.full((b_size, n_lines0), -1,
                                    device=device, dtype=torch.int64)
            pred['line_matches1'] = torch.full((b_size, n_lines1), -1,
                                    device=device, dtype=torch.int64)
            pred['line_match_scores0'] = torch.zeros(
                (b_size, n_lines0), device=device, dtype=torch.float32)
            pred['line_match_scores1'] = torch.zeros(
                (b_size, n_kpts1), device=device, dtype=torch.float32)
            return pred
        
        lines0 = data['lines0'].flatten(1, 2)
        lines1 = data['lines1'].flatten(1, 2)
        lines_junc_idx0 = data['lines_junc_idx0'].flatten(1, 2)  # [b_size, num_lines * 2]
        lines_junc_idx1 = data['lines_junc_idx1'].flatten(1, 2)

        kpts0 = normalize_keypoints(kpts0, data['image_shape0'])
        kpts1 = normalize_keypoints(kpts1, data['image_shape1'])

        assert torch.all(kpts0 >= -1) and torch.all(kpts0 <= 1)
        assert torch.all(kpts1 >= -1) and torch.all(kpts1 <= 1)
        desc0 = desc0 + self.kenc(kpts0, data['keypoint_scores0'])
        desc1 = desc1 + self.kenc(kpts1, data['keypoint_scores1'])
        
        if n_lines0 != 0 and n_lines1 != 0:
            # Pre-compute the line encodings
            lines0 = normalize_keypoints(
                lines0, data['image_shape0']).reshape(b_size, n_lines0, 2, 2)
            lines1 = normalize_keypoints(
                lines1, data['image_shape1']).reshape(b_size, n_lines1, 2, 2)
            line_enc0 = self.lenc(lines0, data['line_scores0'])
            line_enc1 = self.lenc(lines1, data['line_scores1'])
        else:
            line_enc0 = torch.zeros(
                b_size, self.conf.descriptor_dim, n_lines0 * 2,
                dtype=torch.float, device=device)
            line_enc1 = torch.zeros(
                b_size, self.conf.descriptor_dim, n_lines1 * 2,
                dtype=torch.float, device=device)

        desc0, desc1 = self.gnn(desc0, desc1, line_enc0, line_enc1,
                                lines_junc_idx0, lines_junc_idx1)

        # Match all points (KP and line junctions)
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)

        kp_scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        kp_scores = kp_scores / self.conf.descriptor_dim ** .5
        kp_scores = log_double_softmax(kp_scores, self.bin_score)
        m0, m1, mscores0, mscores1 = self._get_matches(kp_scores)
        pred['log_assignment'] = kp_scores
        pred['matches0'] = m0
        pred['matches1'] = m1
        pred['match_scores0'] = mscores0
        pred['match_scores1'] = mscores1

        # Match the lines
        if n_lines0 > 0 and n_lines1 > 0:
            (line_scores, m0_lines, m1_lines, mscores0_lines,
             mscores1_lines, raw_line_scores) = self._get_line_matches(
                desc0[:, :, :2*n_lines0], desc1[:, :, :2*n_lines1],
                lines_junc_idx0, lines_junc_idx1, self.final_line_proj)
        else:
            line_scores = torch.zeros(b_size, n_lines0, n_lines1,
                                      dtype=torch.float, device=device)
            m0_lines = torch.full((b_size, n_lines0), -1,
                                    device=device, dtype=torch.int64)
            m1_lines = torch.full((b_size, n_lines1), -1,
                                    device=device, dtype=torch.int64)
            mscores0_lines = torch.zeros(
                (b_size, n_lines0), device=device, dtype=torch.float32)
            mscores1_lines = torch.zeros(
                (b_size, n_lines1), device=device, dtype=torch.float32)
            raw_line_scores = torch.zeros(b_size, n_lines0, n_lines1,
                                          dtype=torch.float, device=device)
        pred['line_log_assignment'] = line_scores
        pred['line_matches0'] = m0_lines
        pred['line_matches1'] = m1_lines
        pred['line_match_scores0'] = mscores0_lines
        pred['line_match_scores1'] = mscores1_lines
        pred['raw_line_scores'] = raw_line_scores

        return pred

    def _get_matches(self, scores_mat):
        max0 = scores_mat[:, :-1, :-1].max(2)
        max1 = scores_mat[:, :-1, :-1].max(1)
        m0, m1 = max0.indices, max1.indices
        mutual0 = arange_like(m0, 1)[None] == m1.gather(1, m0)
        mutual1 = arange_like(m1, 1)[None] == m0.gather(1, m1)
        zero = scores_mat.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
        valid0 = mutual0 & (mscores0 > self.conf.filter_threshold)
        valid1 = mutual1 & valid0.gather(1, m1)
        m0 = torch.where(valid0, m0, m0.new_tensor(-1))
        m1 = torch.where(valid1, m1, m1.new_tensor(-1))
        return m0, m1, mscores0, mscores1
    
    def _get_line_matches(self, ldesc0, ldesc1, lines_junc_idx0,
                          lines_junc_idx1, final_proj):
        mldesc0 = final_proj(ldesc0)
        mldesc1 = final_proj(ldesc1)

        line_scores = torch.einsum('bdn,bdm->bnm', mldesc0, mldesc1)
        line_scores = line_scores / self.conf.descriptor_dim ** .5

        # Get the line representation from the junction descriptors
        n2_lines0 = lines_junc_idx0.shape[1]
        n2_lines1 = lines_junc_idx1.shape[1]
        line_scores = torch.gather(
            line_scores, dim=2,
            index=lines_junc_idx1[:, None, :].repeat(1, line_scores.shape[1], 1))
        line_scores = torch.gather(
            line_scores, dim=1,
            index=lines_junc_idx0[:, :, None].repeat(1, 1, n2_lines1))
        line_scores = line_scores.reshape((-1, n2_lines0 // 2, 2,
                                           n2_lines1 // 2, 2))

        # Match either in one direction or the other
        raw_line_scores = 0.5 * torch.maximum(
            line_scores[:, :, 0, :, 0] + line_scores[:, :, 1, :, 1],
            line_scores[:, :, 0, :, 1] + line_scores[:, :, 1, :, 0])
        line_scores = log_double_softmax(raw_line_scores, self.line_bin_score)
        m0_lines, m1_lines, mscores0_lines, mscores1_lines = self._get_matches(
            line_scores)
        return (line_scores, m0_lines, m1_lines, mscores0_lines,
                mscores1_lines, raw_line_scores)


def MLP(channels, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, shape_or_size):
    if isinstance(shape_or_size, (tuple, list)):
        # it's a shape
        h, w = shape_or_size[-2:]
        size = kpts.new_tensor([[w, h]])
    else:
        # it's a size
        assert isinstance(shape_or_size, torch.Tensor)
        size = shape_or_size.to(kpts)
    c = size / 2
    f = size.max(1, keepdim=True).values * 0.7  # somehow we used 0.7 for SG
    return (kpts - c[:, None, :]) / f[:, None, :]


class KeypointEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + list(layers) + [feature_dim], do_bn=True)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


class EndPtEncoder(nn.Module):
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([5] + list(layers) + [feature_dim], do_bn=True)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, endpoints, scores):
        # endpoints should be [B, N, 2, 2]
        # output is [B, feature_dim, N * 2]
        b_size, n_pts, _, _ = endpoints.shape
        assert tuple(endpoints.shape[-2:]) == (2, 2)
        endpt_offset = (endpoints[:, :, 1] - endpoints[:, :, 0]).unsqueeze(2)
        endpt_offset = torch.cat([endpt_offset, -endpt_offset], dim=2)
        endpt_offset = endpt_offset.reshape(b_size, 2 * n_pts, 2).transpose(1, 2)
        inputs = [endpoints.flatten(1, 2).transpose(1, 2),
                  endpt_offset, scores.repeat(1, 2).unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        self.dim = d_model // h
        self.h = h
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        # self.prob = []

    def forward(self, query, key, value):
        b = query.size(0)
        query, key, value = [l(x).view(b, self.dim, self.h, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        # self.prob.append(prob.mean(dim=1))
        return self.merge(x.contiguous().view(b, self.dim * self.h, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, num_dim, num_heads):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, num_dim)
        self.mlp = MLP([num_dim * 2, num_dim * 2, num_dim], do_bn=True)
        nn.init.constant_(self.mlp[-1].bias, 0.0)
        self.scaling = 1.

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1)) * self.scaling


class GNNLayer(nn.Module):
    def __init__(self, feature_dim, layer_type):
        super().__init__()
        assert layer_type in ['cross', 'self']
        self.type = layer_type
        self.update = AttentionalPropagation(feature_dim, 4)

    def forward(self, desc0, desc1):
        if self.type == 'cross':
            src0, src1 = desc1, desc0
        elif self.type == 'self':
            src0, src1 = desc0, desc1
        else:
            raise ValueError("Unknown layer type: " + self.type)
        # self.update.attn.prob = []
        delta0, delta1 = self.update(desc0, src0), self.update(desc1, src1)
        desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


class LineLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.dim = feature_dim
        self.mlp = MLP([self.dim * 3, self.dim * 2, self.dim], do_bn=True)

    def get_endpoint_update(self, ldesc, line_enc, lines_junc_idx):
        # ldesc is [bs, D, n_junc], line_enc [bs, D, n_lines * 2]
        # and lines_junc_idx [bs, n_lines * 2]
        # Create one message per line endpoint
        b_size = lines_junc_idx.shape[0]
        line_desc = torch.gather(
            ldesc, 2,
            lines_junc_idx[:, None].repeat(1, self.dim, 1))
        message = torch.cat([
            line_desc,
            line_desc.reshape(b_size, self.dim, -1, 2).flip([-1]).flatten(2, 3).clone(),
            line_enc], dim=1)
        return self.mlp(message)  # [b_size, D, n_lines * 2]

    def forward(self, ldesc0, ldesc1, line_enc0, line_enc1, lines_junc_idx0,
                lines_junc_idx1):
        # Gather the endpoint updates
        lupdate0 = self.get_endpoint_update(ldesc0, line_enc0, lines_junc_idx0)
        lupdate1 = self.get_endpoint_update(ldesc1, line_enc1, lines_junc_idx1)

        # Average the updates for each junction (requires torch > 1.12)
        update0, update1 = torch.zeros_like(ldesc0), torch.zeros_like(ldesc1)
        dim = ldesc0.shape[1]
        update0 = update0.scatter_reduce_(
            dim=2, index=lines_junc_idx0[:, None].repeat(1, dim, 1),
            src=lupdate0, reduce='mean', include_self=False)
        update1 = update1.scatter_reduce_(
            dim=2, index=lines_junc_idx1[:, None].repeat(1, dim, 1),
            src=lupdate1, reduce='mean', include_self=False)

        # Update
        ldesc0 = ldesc0 + update0
        ldesc1 = ldesc1 + update1

        return ldesc0, ldesc1


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim, layer_types):
        super().__init__()
        self.inter_layers = {}
        self.layers = nn.ModuleList([
            GNNLayer(feature_dim, layer_type)
            for layer_type in layer_types])
        self.line_layers = nn.ModuleList(
            [LineLayer(feature_dim) for _ in range(len(layer_types) // 2)])

    def forward(self, desc0, desc1, line_enc0, line_enc1,
                lines_junc_idx0, lines_junc_idx1):
        for i, layer in enumerate(self.layers):
            desc0, desc1 = layer(desc0, desc1)
            if (layer.type == 'self' and lines_junc_idx0.shape[1] > 0
                and lines_junc_idx1.shape[1] > 0):
                # Add line self attention layers after every self layer
                desc0, desc1 = self.line_layers[i // 2](
                    desc0, desc1, line_enc0, line_enc1,
                    lines_junc_idx0, lines_junc_idx1)
        return desc0, desc1


def log_double_softmax(scores, bin_score):
    b, m, n = scores.shape
    bin_ = bin_score[None, None, None]
    scores0 = torch.cat([scores, bin_.expand(b, m, 1)], 2)
    scores1 = torch.cat([scores, bin_.expand(b, 1, n)], 1)
    scores0 = torch.nn.functional.log_softmax(scores0, 2)
    scores1 = torch.nn.functional.log_softmax(scores1, 1)
    scores = scores.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = (scores0[:, :, :n] + scores1[:, :m, :]) / 2
    scores[:, :-1, -1] = scores0[:, :, -1]
    scores[:, -1, :-1] = scores1[:, -1, :]
    return scores


def arange_like(x, dim):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


### Line matching through GlueStick

class GlueStick():
    """ Class extracting SuperPoint descriptors at line endpoints,
        then matching them with GlueStick.
    """
    default_config = {
        'sp_params': {
            'descriptor_dim': 256,
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': -1,
            'remove_borders': 4,
        },
        'wireframe_params': {
            'nms_radius': 3,
            'max_n_junctions': 1000,
            'force_num_junctions': False
        },
        'gs_params': {
            'descriptor_dim': 256,
            'weights': 'outdoor',
            'keypoint_encoder': [32, 64, 128, 256],
            'GNN_layers': ['self', 'cross'] * 9,
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
        },
        'device': 'cuda',
    }

    def __init__(self, conf):
        self.conf = OmegaConf.create({**self.default_config, **conf})
        self.device = conf.device

        # SuperPoint backbone
        self.sp = SuperPoint(self.conf.sp_params).to(self.device).eval()

        # GlueStick backbone
        self.gs = StitchedWireframe(self.conf.gs_params).to(self.device).eval()
        ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'weights/gluestick.tar')
        ckpt = torch.load(ckpt, map_location='cpu')['model']
        ckpt = {k[8:]: v for (k, v) in ckpt.items() if k.startswith('matcher')}
        self.gs.load_state_dict(ckpt, strict=True)

    def describe_lines(self, image, segs):
        """ Return a [D, N, 2] torch tensor of line endpoints descriptors. """
        if ((not len(image.shape) == 2)
            or (not isinstance(image, np.ndarray))):
            raise ValueError(
                "The input image should be a 2D numpy array")

        if len(segs) == 0:
            return {'image_shape': image.shape, 'lines': np.empty((0, 2, 2)),
                    'line_scores': np.empty((0,)), 'junctions': np.empty((0, 2)),
                    'junc_scores': np.empty((0,)), 'junc_desc': np.empty((256, 0)),
                    'lines_junc_idx': np.empty((0, 2))}
        lines = segs[:, :, [1, 0]].reshape(-1, 2)
        line_scores = np.sqrt(np.linalg.norm(segs[:, 0] - segs[:, 1], axis=1))
        line_scores /= np.amax(line_scores) + 1e-8
        torch_img = {'image': torch.tensor(image.astype(np.float32) / 255,
                                           dtype=torch.float,
                                           device=self.device)[None, None]}
        with torch.no_grad():
            kp, scores, dense_desc = self.sp.compute_dense_descriptor(
                torch_img)
            kp, scores = torch.stack(kp), torch.stack(scores)
            kp_desc = [sample_descriptors(k[None], d[None], 8)[0]
                       for k, d in zip(kp, dense_desc)]

        # Remove keypoints that are too close to line endpoints
        line_endpts = torch.tensor(lines.reshape(1, -1, 2), dtype=torch.float,
                                   device=self.device)
        torch_line_scores = torch.tensor(line_scores[None], dtype=torch.float,
                                         device=self.device)
        dist_pt_lines = torch.norm(
            kp[:, :, None] - line_endpts[:, None], dim=-1)
        # For each keypoint, mark it as valid or to remove
        pts_to_remove = torch.any(
            dist_pt_lines < self.conf.wireframe_params.nms_radius, dim=2)
        kp = kp[0][~pts_to_remove[0]][None]
        scores = scores[0][~pts_to_remove[0]][None]
        kp_desc = kp_desc[0].T[~pts_to_remove[0]].T[None]

        # Connect the lines together to form a wireframe
        # Merge first close-by endpoints to connect lines
        (line_points, line_pts_scores, line_descs, _,
         lines, lines_junc_idx, _) = lines_to_wireframe(
             line_endpts, torch_line_scores, dense_desc,
             conf=self.conf.wireframe_params)

        # Add the keypoints to the junctions
        all_points = torch.cat([torch.stack(line_points), kp], dim=1)
        all_scores = torch.cat([torch.stack(line_pts_scores), scores], dim=1)
        all_descs = torch.cat([torch.stack(line_descs), kp_desc], dim=2)

        return {'image_shape': image.shape, 'lines': lines.reshape(1, -1, 2, 2),
                'line_scores': torch_line_scores, 'junctions': all_points,
                'junc_scores': all_scores, 'junc_desc': all_descs,
                'lines_junc_idx': lines_junc_idx}

    def match_lines(self, desc0, desc1):
        """ Match the line endpoints with GlueStick. """
        # Setup the inputs for GlueStick
        inputs = {
            'image_shape0': tuple(desc0['image_shape']),
            'image_shape1': tuple(desc1['image_shape']),
            'keypoints0': desc0['junctions'],
            'keypoints1': desc1['junctions'],
            'keypoint_scores0': desc0['junc_scores'],
            'keypoint_scores1': desc1['junc_scores'],
            'descriptors0': desc0['junc_desc'],
            'descriptors1': desc1['junc_desc'],
            'lines0': desc0['lines'],
            'lines1': desc1['lines'],
            'line_scores0': desc0['line_scores'],
            'line_scores1': desc1['line_scores'],
            'lines_junc_idx0': desc0['lines_junc_idx'],
            'lines_junc_idx1': desc1['lines_junc_idx'],
        }

        with torch.no_grad():
            # Run the point matching
            out = self.gs(inputs)
            matches = out['line_matches0'].cpu().numpy()[0]

        return matches
