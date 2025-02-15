"""

adapted from GeoTransformer  https://github.com/qinzheng93/GeoTransformer
experiments/*/loss.py

"""
import torch

from ..modules.ops import apply_transform
from ..modules.registration.metrics import isotropic_transform_error


class Evaluator(torch.nn.Module):
    def __init__(self, cfg):
        super(Evaluator, self).__init__()
        self.acceptance_overlap = cfg.eval.acceptance_overlap
        self.acceptance_radius = cfg.eval.acceptance_radius
        self.acceptance_rmse = cfg.eval.rmse_threshold
        self.acceptance_rre = cfg.eval.rre_threshold
        self.acceptance_rte = cfg.eval.rte_threshold
        # logger = logging.getLogger(__name__)
        # logger.info('TRUE RMSE CALCULATION')

    @torch.no_grad()
    def evaluate_coarse(self, output_dict):
        ref_length_c = output_dict['ref_points_c'].shape[0]
        src_length_c = output_dict['src_points_c'].shape[0]
        gt_node_corr_overlaps = output_dict['gt_node_corr_overlaps']
        gt_node_corr_indices = output_dict['gt_node_corr_indices']
        masks = torch.gt(gt_node_corr_overlaps, self.acceptance_overlap)
        gt_node_corr_indices = gt_node_corr_indices[masks]
        gt_ref_node_corr_indices = gt_node_corr_indices[:, 0]
        gt_src_node_corr_indices = gt_node_corr_indices[:, 1]
        gt_node_corr_map = torch.zeros(ref_length_c, src_length_c).cuda()
        gt_node_corr_map[gt_ref_node_corr_indices, gt_src_node_corr_indices] = 1.0

        ref_node_corr_indices = output_dict['ref_node_corr_indices']
        src_node_corr_indices = output_dict['src_node_corr_indices']

        precision = gt_node_corr_map[ref_node_corr_indices, src_node_corr_indices].mean()

        return precision

    @torch.no_grad()
    def evaluate_fine(self, output_dict, data_dict):
        transform = data_dict['transform'].squeeze()
        ref_corr_points = output_dict['ref_corr_points'][:, :3].to(transform.dtype)
        src_corr_points = output_dict['src_corr_points'][:, :3].to(transform.dtype)
        src_corr_points = apply_transform(src_corr_points, transform)
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        precision = torch.lt(corr_distances, self.acceptance_radius).float().mean()
        return torch.tensor(0.) if precision.isnan() else precision

    @torch.no_grad()
    def evaluate_registration(self, output_dict, data_dict):
        est_transform = output_dict['estimated_transform']
        transform = data_dict['transform'].squeeze().to(est_transform.dtype)
        src_points = output_dict['src_points'][:, :3].to(est_transform.dtype)

        rre, rte = isotropic_transform_error(transform, est_transform)
        recall_t = torch.logical_and(torch.lt(rre, self.acceptance_rre), torch.lt(rte, self.acceptance_rte)).float()

        realignment_transform = torch.matmul(torch.inverse(transform), est_transform)
        realigned_src_points_f = apply_transform(src_points, realignment_transform)
        rmse = torch.linalg.norm(realigned_src_points_f - src_points, dim=1).mean()
        # diff = realigned_src_points_f - src_points
        # rmse = torch.sqrt(torch.einsum('nd,nd->n', diff, diff).mean())
        recall_rmse = torch.lt(rmse, self.acceptance_rmse).float()

        return rre, rte, recall_t, rmse, recall_rmse

    def forward(self, output_dict, data_dict):
        # c_precision = self.evaluate_coarse(output_dict)
        f_precision = self.evaluate_fine(output_dict, data_dict)
        rre, rte, recall_t, rmse, recall = self.evaluate_registration(output_dict, data_dict)

        return {
            # 'PIR': c_precision,
            'IR': f_precision,
            'RRE': rre,
            'RTE': rte,
            'RR_T': recall_t,
            'RMSE': rmse,
            'RR_RMSE': recall,
        }
