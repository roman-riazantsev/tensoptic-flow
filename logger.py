import tensorflow as tf

from utils import flow_vis


class Logger(object):
    def __init__(self, config):
        logs_dir_path = config['logs_dir_path']
        self.loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.final_loss_metric = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        self.summary_writer = tf.summary.create_file_writer(logs_dir_path)

        self.get_loss_value = self.loss_metric.result
        self.get_final_loss = self.final_loss_metric.result

    def log_loss(self, step):
        with self.summary_writer.as_default():
            tf.summary.scalar('loss', self.get_loss_value(), step=step)
            tf.summary.scalar('final loss', self.get_final_loss(), step=step)

    def log_flows(self, scaled_flows_gt_batch, pred_flows_batch, step):
        scaled_flows_gt = [learning_example[0] for learning_example in scaled_flows_gt_batch]
        pred_flows = [learning_example[0] for learning_example in pred_flows_batch]
        log_pairs = []

        for idx, (true_flow, pred_flow) in enumerate(zip(scaled_flows_gt, pred_flows)):
            flow_color_true = flow_vis.flow_to_color(true_flow, convert_to_bgr=False)
            flow_color_pred = flow_vis.flow_to_color(pred_flow, convert_to_bgr=False)

            flow_color_true = [flow_color_true/255]
            flow_color_pred = [flow_color_pred/255]

            log_pairs.append(("Flow_GT_{}".format(idx), flow_color_true))
            log_pairs.append(("Flow_Pred_{}".format(idx), flow_color_pred))

        for flow_id, img in log_pairs:
            with self.summary_writer.as_default():
                tf.summary.image(flow_id, img, step=step)

    def reset_loss_states(self):
        self.loss_metric.reset_states()
