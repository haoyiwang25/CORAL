import torch
import torch.nn as nn
import torch.nn.functional as F

class SupConLoss(nn.Module):
    def __init__(self, threshold=0.1, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, contrastive_method='simclr'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
        self.threshold = threshold
        self.contrastive_method = contrastive_method

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, N, C)
        # v shape: (N, N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, features, labels=None, mask=None):
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            if self.contrastive_method == 'gcl':
                mask = torch.eq(labels, labels.T).float().to(device)
            elif self.contrastive_method == 'pcl':
                mask = (torch.abs(labels.T.repeat(batch_size,1) - labels.repeat(1,batch_size)) < self.threshold).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        logits = torch.div(
            self._cosine_simililarity(anchor_feature, contrast_feature),
            self.temperature)
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


class CORALLoss(nn.Module):
    def __init__(self, feature_dim, temperature=0.07, alpha=1.0, contrast_mode='all',
                 apply_soft_weights=True, sigmoid_scale=1.0):
        super(CORALLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.contrast_mode = contrast_mode
        self.apply_soft_weights = apply_soft_weights
        self.sigmoid_scale = sigmoid_scale
        
        self.v_prog = nn.Parameter(torch.randn(feature_dim))
        nn.init.normal_(self.v_prog, mean=0.0, std=1.0)
        
    def _normalize_vector(self, v):
        return v / (torch.norm(v, p=2) + 1e-8)
    
    def _sigmoid_weighting(self, distances):
        return torch.sigmoid(self.sigmoid_scale * distances)
    
    def _compute_ral_loss(self, features, positions):
        device = features.device
        n = features.shape[0]

        positions = positions.to(device)
        
        features = F.normalize(features, p=2, dim=1)
        
        sim_matrix = torch.matmul(features, features.T)
        sim_matrix_scaled = sim_matrix / self.temperature

        pos_dist = torch.abs(positions - positions.T)
        pos_dist_ik = pos_dist.unsqueeze(1).expand(n, n, n) 
        pos_dist_ij = pos_dist.unsqueeze(2).expand(n, n, n)  
        mask_neg = (pos_dist_ik >= pos_dist_ij).float()
        
        mask_not_i = 1.0 - torch.eye(n, device=device).unsqueeze(1).expand(n, n, n)
        mask_neg = mask_neg * mask_not_i
        
        exp_sim = torch.exp(sim_matrix_scaled)
        exp_sim_expanded = exp_sim.unsqueeze(1).expand(n, n, n)
        
        denominator = torch.sum(exp_sim_expanded * mask_neg, dim=2)
        numerator = torch.exp(sim_matrix_scaled)
        
        epsilon = 1e-8
        loss_matrix = -torch.log(numerator / (denominator + epsilon))
        
        mask_self = 1.0 - torch.eye(n, device=device)
        valid_mask = mask_self * (denominator > 0).float()
        
        if self.apply_soft_weights:
            soft_weights = self._sigmoid_weighting(pos_dist)
            weighted_loss_matrix = loss_matrix * soft_weights
        else:
            weighted_loss_matrix = loss_matrix
        
        total_loss = (weighted_loss_matrix * valid_mask).sum()
        count = valid_mask.sum()
        
        if count > 0:
            ral_loss = total_loss / count
        else:
            ral_loss = torch.tensor(0.0, device=device)
        
        return ral_loss
    
    def _compute_oal_loss(self, features, positions):
        device = features.device
        n = features.shape[0]

        positions = positions.to(device)
        
        v_prog_normalized = self._normalize_vector(self.v_prog)
        v_prog_normalized = v_prog_normalized.to(device)

        pos_matrix = positions.expand(n, n) 
        pos_matrix_T = positions.T.expand(n, n)  

        mask = (pos_matrix < pos_matrix_T).float()
        
        features_i = features.unsqueeze(1).expand(n, n, -1) 
        features_j = features.unsqueeze(0).expand(n, n, -1) 
        
        direction_vecs = features_j - features_i
        direction_vecs_normalized = F.normalize(direction_vecs, p=2, dim=2)
        cos_sim = torch.matmul(direction_vecs_normalized, v_prog_normalized)
        
        cos_sim_masked = cos_sim * mask
        total_cos_sim = cos_sim_masked.sum()
        count = mask.sum()
        
        if count > 0:
            oal_loss = -total_cos_sim / count
        else:
            oal_loss = torch.tensor(0.0, device=device)
        
        return oal_loss
    
    def forward(self, features, labels):
        
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, feat_dim], '
                           'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)
        
        batch_size = features.shape[0]
        
        if labels is None:
            raise ValueError('Labels (positions) are required for CORAL loss')

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_positions = labels
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_positions = labels.repeat(contrast_count, 1)
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        
        ral_loss = self._compute_ral_loss(anchor_feature, anchor_positions)
        
        oal_loss = self._compute_oal_loss(anchor_feature, anchor_positions)

        total_loss = ral_loss + self.alpha * oal_loss
        
        return {
            'loss': total_loss,
            'ral_loss': ral_loss.item() if torch.is_tensor(ral_loss) else ral_loss,
            'oal_loss': oal_loss.item() if torch.is_tensor(oal_loss) else oal_loss
        }
