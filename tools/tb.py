from torch.utils.tensorboard import SummaryWriter


class AutoStepWriter:
    def __init__(self, logdir=None):
        self.writer = SummaryWriter(logdir)
        self.step_dict = {}

    def add_scalar(self, tag, value, step=None):
        if step is None:
            step = self.step_dict.get(tag, 0)
            self.step_dict[tag] = step + 1
        self.writer.add_scalar(tag, value, step)

    def add_scalars(self, main_tag, tag_scalar_dict, step=None):
        if step is None:
            step = self.step_dict.get(main_tag, 0)
            self.step_dict[main_tag] = step + 1
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def add_histogram(self, tag, values, step=None, bins='tensorflow'):
        if step is None:
            step = self.step_dict.get(tag, 0)
            self.step_dict[tag] = step + 1
        self.writer.add_histogram(tag, values, step, bins=bins)

    def add_image(self, tag, img_tensor, step=None, dataformats='CHW'):
        if step is None:
            step = self.step_dict.get(tag, 0)
            self.step_dict[tag] = step + 1
        self.writer.add_image(tag, img_tensor, step, dataformats=dataformats)

    def add_images(self, tag, img_tensor, step=None, dataformats='NCHW'):
        if step is None:
            step = self.step_dict.get(tag, 0)
            self.step_dict[tag] = step + 1
        self.writer.add_images(tag, img_tensor, step, dataformats=dataformats)

    def add_figure(self, tag, figure, step=None):
        if step is None:
            step = self.step_dict.get(tag, 0)
            self.step_dict[tag] = step + 1
        self.writer.add_figure(tag, figure, step)

    def add_text(self, tag, text_string, step=None):
        if step is None:
            step = self.step_dict.get(tag, 0)
            self.step_dict[tag] = step + 1
        self.writer.add_text(tag, text_string, step)

    def add_embedding(self, mat, metadata=None, label_img=None, step=None, tag='default'):
        if step is None:
            step = self.step_dict.get(tag, 0)
            self.step_dict[tag] = step + 1
        self.writer.add_embedding(mat, metadata=metadata, label_img=label_img, global_step=step, tag=tag)

    def add_pr_curve(self, tag, labels, predictions, step=None, num_thresholds=127):
        if step is None:
            step = self.step_dict.get(tag, 0)
            self.step_dict[tag] = step + 1
        self.writer.add_pr_curve(tag, labels, predictions, step, num_thresholds=num_thresholds)

    def add_graph(self, model, input_to_model):
        self.writer.add_graph(model, input_to_model)

    def add_hparams(self, hparam_dict, metric_dict):
        self.writer.add_hparams(hparam_dict, metric_dict)

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False