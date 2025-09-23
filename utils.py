from einops import rearrange
from torchvision.utils import draw_bounding_boxes
import torch
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import EarlyStopping


class CustomProgressBar(TQDMProgressBar):

    def __init__(self, rm_metrics=None, add_metrics=None):
        super().__init__()
        self.rm_metrics = rm_metrics
        self.add_metrics = add_metrics

    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        if self.rm_metrics:
            for metric in self.rm_metrics:
                items.pop(metric, None)
        if self.add_metrics:
            items.update(self.add_metrics)
        return items

class CustomEarlyStopping(EarlyStopping):
    def __init__(self, min_epoch=50, *args, **kwargs):
        """
        Early stops the training if the monitored metric doesn't improve after a given patience.

        Args:
            min_epoch (int): Earliest epoch possible for stopping.
            *args: Variable length argument list for the base EarlyStopping class.
            **kwargs: Arbitrary keyword arguments for the base EarlyStopping class.
        """
        super().__init__(*args, **kwargs)
        self.min_epoch = min_epoch

    def _run_early_stopping_check(self, trainer):
        """
        Override the original method to incorporate min_epoch functionality.
        Performs the check to stop training early, but only enforces stopping after min_epoch.
        """
        logs = trainer.callback_metrics

        # Disable early_stopping with fast_dev_run or if metric is not present
        if trainer.fast_dev_run or not self._validate_condition_metric(logs):
            return

        current = logs[self.monitor].squeeze()
        # Even if we are before the min_epoch, we still track the best score and wait count
        should_stop, reason = self._evaluate_stopping_criteria(current)

        # Enforce the early stopping only if the current epoch is greater than min_epoch
        if trainer.current_epoch >= self.min_epoch:
            should_stop = trainer.strategy.reduce_boolean_decision(should_stop, all=False)
            trainer.should_stop = trainer.should_stop or should_stop
            if should_stop:
                self.stopped_epoch = trainer.current_epoch
            if reason and self.verbose:
                self._log_info(trainer, reason, self.log_rank_zero_only)
        else:
            # Here you could add logic to reset should_stop if you want to ignore
            # the stopping criteria before min_epoch completely.
            # For now, it tracks performance without enforcing stop.
            if self.verbose:
                trainer.logger.info(f"CustomEarlyStopping skipped enforcement due to current epoch \
                        {trainer.current_epoch + 1} < min_epoch {self.min_epoch}. Best score so far: {self.best_score}")


def visualize_bag(bag:dict, positive_classes=[]):
    bag_imgs = bag['bag']
    n,c,h,w = bag_imgs.shape
    slide = rearrange(bag_imgs, 'n c h w -> c h (n w)')
    if bag['label'] > 0 and positive_classes:
        # draw red bounding boxes on positive patches
        for i, patch_label in enumerate(bag['patch_labels']):
            if patch_label in positive_classes:
                x1 = i * w
                x2 = (i+1) * w
                y1 = 0
                y2 = h
                slide = draw_bounding_boxes(slide, torch.tensor([[x1, y1, x2, y2]]), width=2)
    return slide

def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1


def overwrite_grad(params, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1

if __name__ == '__main__':
    from torchvision.datasets import MNIST
    from datasets.dataset import SimpleBagWapper, RotatedMNIST
    import matplotlib.pyplot as plt
    import torchvision

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    mnist = MNIST('../data', train=True, transform=transform, download=True)
    bagged_mnist = SimpleBagWapper(mnist, bag_length=10, positive_classes=[7])
    for i in range(5):
        bag = bagged_mnist[i]
        slide = visualize_bag(bag, positive_classes=[7])
        plt.imshow(slide.permute(1,2,0))
        plt.show()