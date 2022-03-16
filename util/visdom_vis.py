import copy
import logging

import matplotlib.patches as mpatches
import numpy as np
import torch
import torchvision.transforms as T
from matplotlib import colors
from matplotlib import pyplot as plt
from visdom import Visdom

# from .util.plot_utils import fig_to_numpy

logging.getLogger('visdom').setLevel(logging.CRITICAL)


class BaseVis(object):

    def __init__(self, viz_opts, update_mode='append', env=None, win=None,
                 resume=False, port=8097, server='http://localhost'):
        self.viz_opts = viz_opts
        self.update_mode = update_mode
        self.win = win
        if env is None:
            env = 'main'
        self.viz = Visdom(env=env, port=port, server=server)
        # if resume first plot should not update with replace
        self.removed = not resume

    def win_exists(self):
        return self.viz.win_exists(self.win)

    def close(self):
        if self.win is not None:
            self.viz.close(win=self.win)
            self.win = None

    def register_event_handler(self, handler):
        self.viz.register_event_handler(handler, self.win)


class LineVis(BaseVis):
    """Visdom Line Visualization Helper Class."""

    def plot(self, y_data, x_label):
        """Plot given data.

        Appends new data to exisiting line visualization.
        """
        update = self.update_mode
        # update mode must be None the first time or after plot data was removed
        if self.removed:
            update = None
            self.removed = False

        if isinstance(x_label, list):
            Y = torch.Tensor(y_data)
            X = torch.Tensor(x_label)
        else:
            y_data = [d.cpu() if torch.is_tensor(d)
                      else torch.tensor(d)
                      for d in y_data]

            Y = torch.Tensor(y_data).unsqueeze(dim=0)
            X = torch.Tensor([x_label])

        win = self.viz.line(X=X, Y=Y, opts=self.viz_opts, win=self.win, update=update)

        if self.win is None:
            self.win = win
        self.viz.save([self.viz.env])

    def reset(self):
        #TODO: currently reset does not empty directly only on the next plot.
        # update='remove' is not working as expected.
        if self.win is not None:
            # self.viz.line(X=None, Y=None, win=self.win, update='remove')
            self.removed = True


class ImgVis(BaseVis):
    """Visdom Image Visualization Helper Class."""

    def plot(self, images):
        """Plot given images."""

        # images = [img.data if isinstance(img, torch.autograd.Variable)
        #           else img for img in images]
        # images = [img.squeeze(dim=0) if len(img.size()) == 4
        #           else img for img in images]

        self.win = self.viz.images(
            images,
            nrow=1,
            opts=self.viz_opts,
            win=self.win, )
        self.viz.save([self.viz.env])


# def vis_results(visualizer, img, result, target, tracking, vis_th_score, category_map):
#     inv_normalize = T.Normalize(
#         mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
#         std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
#     )
#
#     imgs = [inv_normalize(img).cpu()]
#     img_ids = [target['image_id'].item()]
#     for key in ['prev_image', 'random_image']:
#         if key in target:
#             imgs.append(inv_normalize(target[key]).cpu())
#             img_ids.append(target[f'{key}_id'].item())
#
#     # img.shape=[3, H, W]
#     dpi = 96
#     figure, axarr = plt.subplots(len(imgs))
#     figure.tight_layout()
#     figure.set_dpi(dpi)
#     figure.set_size_inches(
#         imgs[0].shape[2] / dpi,
#         imgs[0].shape[1] * len(imgs) / dpi)
#
#     if len(imgs) == 1:
#         axarr = [axarr]
#
#     for ax, img, img_id in zip(axarr, imgs, img_ids):
#         ax.set_axis_off()
#         ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
#
#         ax.text(
#             0, 0, f'IMG_ID={img_id}',
#             fontsize=20, bbox=dict(facecolor='white', alpha=0.5))
#
#     num_track_queries = num_track_queries_with_id = 0
#     if tracking:
#         num_track_queries = len(target['track_query_boxes'])
#         num_track_queries_with_id = len(target['track_query_match_ids'])
#         if "track_query_match_ids_only_positive" in target:
#             track_ids = target['track_ids'][target['track_query_match_ids_only_positive']]
#         else:
#             track_ids = target['track_ids'][target['track_query_match_ids']]
#
#     keep = result['scores'].cpu() > vis_th_score
#     det_categories = None
#     if category_map is not None:
#         det_categories = [category_map[label.item()+1] for label in result['labels'].cpu()]
#
#     cmap = plt.cm.get_cmap('hsv', len(keep))
#
#     prop_i = 0
#     for box_id in range(len(keep)):
#         mask_value = 0
#         if tracking:
#             if "track_queries_match_mask_only_positive" in target:
#                 mask_value = target['track_queries_match_mask_only_positive'][box_id].item()
#             else:
#                 mask_value = target['track_queries_match_mask'][box_id].item()
#
#         rect_color = 'green'
#         offset = 0
#         text = f"{result['scores'][box_id]:0.2f}"
#         if det_categories is not None:
#             text = (f"{text}\n"
#                     f"{det_categories[box_id]}")
#         # Check if result comes from track
#         if mask_value == 1:
#             if keep[box_id]:
#                 offset = 50
#                 rect_color = 'blue'
#                 text = (
#                     f"{track_ids[prop_i]}\n"
#                     f"{text}\n"
#                     f"{result['track_queries_with_id_iou'][prop_i]:0.2f}")
#             else:
#                 offset = 50
#                 rect_color = 'grey'
#                 text = (
#                     f"{track_ids[prop_i]}\n"
#                     f"{text}"
#                     )
#             prop_i += 1
#
#         # Check if result comes from track but other condition
#         elif mask_value == -1:
#             rect_color = 'red'
#
#         if not keep[box_id] and not rect_color == 'grey':
#             continue
#
#
#         # Plot current frame detection
#         x1, y1, x2, y2 = result['boxes'][box_id]
#
#         axarr[0].add_patch(plt.Rectangle(
#             (x1, y1), x2 - x1, y2 - y1,
#             fill=False, color=rect_color, linewidth=2))
#
#         axarr[0].text(
#             x1, y1 + offset, text,
#             fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
#
#         if 'masks' in result:
#             mask = result['masks'][box_id][0].numpy()
#             mask = np.ma.masked_where(mask == 0.0, mask)
#
#             axarr[0].imshow(
#                 mask, alpha=0.5, cmap=colors.ListedColormap([cmap(box_id)]))
#
#     query_keep = keep
#     if "track_queries_match_mask_only_positive" in target:
#
#         query_keep = keep[target['track_queries_match_mask_only_positive'] == 0]
#         num_track_queries_with_id_only_postivive =  (target['track_queries_match_mask_only_positive'] == 1).sum()
#         num_track_queries_only_positive = (target['track_queries_match_mask_only_positive'] != 0).sum()
#         num_track_queries_only_positive_filtered_score = (~keep[target['track_queries_match_mask_only_positive'] == 1]).sum()
#
#         legend_handles = [mpatches.Patch(
#             color="black",
#             label=f"Tot query positv: {num_track_queries_with_id}/{num_track_queries}")]
#
#         # Current frame detections that should have been detected by new embeddings has they don't belong to any track
#         legend_handles.append(mpatches.Patch(
#             color='green',
#             label=f"object queries ({query_keep.sum()}/{len(target['boxes']) - num_track_queries_with_id_only_postivive})\n- cls_score"))
#         # Current frame detections that should have been detected from the track_embedding from last frame
#         if num_track_queries_with_id_only_postivive:
#             legend_handles.append(mpatches.Patch(
#                 color='blue',
#                 label=f"track queries ({keep[target['track_queries_match_mask_only_positive'] == 1].sum()}/{num_track_queries_with_id_only_postivive})\n- track_id\n- cls_score\n- iou"))
#
#         # Added false postive examples from last frame that SHOULD NOT PRODUCE ANY detection on current frame
#         if num_track_queries_with_id_only_postivive != num_track_queries_only_positive:
#             legend_handles.append(mpatches.Patch(
#                 color='red',
#                 label=f"false track queries ({keep[target['track_queries_match_mask_only_positive'] == -1].sum()}/{num_track_queries_only_positive - num_track_queries_with_id_only_postivive})"))
#
#         # Track queries without false positive that have been filtered by top_k score AND SHOULD NOT HAD HAPPENED
#         if num_track_queries_only_positive_filtered_score > 0:
#             legend_handles.append(mpatches.Patch(
#                 color='gray',
#                 label=f"track queries filter {num_track_queries_only_positive_filtered_score}/{num_track_queries_with_id_only_postivive}\n- cls_score"))
#
#
#     else:
#         if tracking:
#             # TODO: Results should probably distinct between track and detections with only positive
#             query_keep = keep[target['track_queries_match_mask'] == 0]
#
#         legend_handles = [mpatches.Patch(
#             color='green',
#             label=f"object queries ({query_keep.sum()}/{len(target['boxes']) - num_track_queries_with_id})\n- cls_score")]
#
#         if num_track_queries:
#             legend_handles.append(mpatches.Patch(
#                 color='blue',
#                 label=f"track queries ({keep[target['track_queries_match_mask'] == 1].sum()}/{num_track_queries_with_id})\n- track_id\n- cls_score\n- iou"))
#         if num_track_queries_with_id != num_track_queries:
#             legend_handles.append(mpatches.Patch(
#                 color='red',
#                 label=f"false track queries ({keep[target['track_queries_match_mask'] == -1].sum()}/{num_track_queries - num_track_queries_with_id})"))
#
#     axarr[0].legend(handles=legend_handles)
#
#     i = 1
#     for frame_prefix in ['prev', 'random']:
#         if f'{frame_prefix}_image_id' not in target or f'{frame_prefix}_boxes' not in target:
#             continue
#
#         cmap = plt.cm.get_cmap('hsv', len(target[f'{frame_prefix}_track_ids']))
#
#         for j, track_id in enumerate(target[f'{frame_prefix}_track_ids']):
#             x1, y1, x2, y2 = target[f'{frame_prefix}_boxes'][j]
#             axarr[i].text(
#                 x1, y1, f"track_id={track_id}",
#                 fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
#             axarr[i].add_patch(plt.Rectangle(
#                 (x1, y1), x2 - x1, y2 - y1,
#                 fill=False, color='green', linewidth=2))
#
#             if f'{frame_prefix}_masks' in target:
#                 mask = target[f'{frame_prefix}_masks'][j].cpu().numpy()
#                 mask = np.ma.masked_where(mask == 0.0, mask)
#
#                 axarr[i].imshow(
#                     mask, alpha=0.5, cmap=colors.ListedColormap([cmap(j)]))
#         i += 1
#
#     plt.subplots_adjust(wspace=0.01, hspace=0.01)
#     plt.axis('off')
#
#     img = fig_to_numpy(figure).transpose(2, 0, 1)
#     plt.close()
#
#     visualizer.plot(img)


def build_visualizers(args):
    visualizers = {}
    visualizers['train'] = {}
    visualizers['val'] = {}

    if args.no_vis:
        return visualizers

    env_name = str(args.output_dir).split('/')[-1]

    vis_kwargs = {
        'env': env_name,
        'resume': False,
        'port': args.vis_port,
        'server': args.vis_server}

    #
    # METRICS
    #
    legend = [
        'class_error',
        'loss',
        'loss_bbox',
        'loss_ce',
        'loss_giou',
        'loss_mask',
        'loss_dice',
        'loss_centroids',
        'cardinality_error_unscaled',
        'loss_bbox_unscaled',
        'loss_ce_unscaled',
        'loss_giou_unscaled',
        'loss_mask_unscaled',
        'loss_dice_unscaled',
        'loss_centroids_unscaled',
        'lr_finetuned_params',
        'lr_new_params',
        'lr_backbone',
        'lr_curr_sampling',
        'lr_temporal_sampling',
        'iter_time'
    ]

    if not args.new_segm_module == "final":
        legend.remove('loss_centroids')
        legend.remove('loss_centroids_unscaled')



    opts = dict(
        title="TRAIN METRICS ITERS",
        xlabel='ITERS',
        ylabel='METRICS',
        width=1000,
        height=500,
        legend=legend)

    # TRAIN
    visualizers['train']['iter_metrics'] = LineVis(opts, **vis_kwargs)

    legend = [
        'TRACK mAP IoU=0.50:0.95',
        'TRACK mAR IoU=0.50:0.95',
    ]

    opts = dict(
        title='TRAIN EVAL EPOCHS',
        xlabel='EPOCHS',
        ylabel='METRICS',
        width=1000,
        height=500,
        legend=legend)

    # TRAIN
    visualizers['train']['epoch_eval'] = LineVis(opts, **vis_kwargs)

    # # VAL
    opts = copy.deepcopy(opts)
    opts['title'] = 'VAL EVAL EPOCHS'
    visualizers['val']['epoch_eval'] = LineVis(opts, **vis_kwargs)

    return visualizers

def get_vis_win_names(vis_dict):
    vis_win_names = {
        outer_k: {
            inner_k: inner_v.win
            for inner_k, inner_v in outer_v.items()
        }
        for outer_k, outer_v in vis_dict.items()
    }
    return vis_win_names
