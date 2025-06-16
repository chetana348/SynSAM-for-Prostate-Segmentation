import os
import torch
import pickle
from tqdm import tqdm
import monai
from monai.inferers import SliceInferer
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch, SmartCacheDataset
#from monai.metrics import DiceMetric
from trainer.losses import *


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def train2D(model, dataloader, test_dataloader, loss, optim, max_epochs, model_dir, device, name, test_interval=1, cl_dir=None, lambda_ewc=1.7):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []

    # Load EWC info if available
    fisher, prev_params = {}, {}
    use_ewc = False
    if cl_dir and os.path.exists(os.path.join(cl_dir, "fisher.pkl")):
        fisher = load_pickle(os.path.join(cl_dir, "fisher.pkl"))
        prev_params = load_pickle(os.path.join(cl_dir, "params.pkl"))
        ewc_loss_fn = EWC(fisher_information=fisher, saved_params=prev_params, lambda_ewc=lambda_ewc)
        use_ewc = True
        print(f"[INFO] Loaded EWC prior with {len(fisher)} parameters.")

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train = 0

        with tqdm(dataloader, unit="batch") as tepoch:
            for image, label, _, _ in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                train_step += 1

                image = image.to(device)                         # [B, 1, H, W]
                label = label.to(device)                 # [B, H, W]

                optim.zero_grad()
                outputs = model(image)                          # [B, C, H, W]
                train_loss = loss(outputs, label)              # DiceFocalLoss: target = [B, H, W]

                if use_ewc:
                    train_loss += ewc_loss_fn.compute(model.named_parameters())

                train_loss.backward()
                optim.step()

                train_epoch_loss += train_loss.item()

                # Metric computation
                labels_list = decollate_batch(label)
                labels_convert = [post_label(x) for x in labels_list]

                output_list = decollate_batch(outputs)
                output_convert = [post_pred(x) for x in output_list]

                dice_metric(y_pred=output_convert, y=labels_convert)
                iou_metric(y_pred=output_convert, y=labels_convert)
                dice_vals = dice_metric.aggregate()
                mean_dice = dice_vals.mean().item()  # Now this is scalar-safe
                tepoch.set_postfix(loss=train_loss.item(), dice_score=mean_dice)

        train_epoch_loss /= train_step
        epoch_metric_train = dice_metric.aggregate().mean().item()
        iou_metric_train = iou_metric.aggregate().mean().item()

        dice_metric.reset()
        iou_metric.reset()

        print(f"Train Loss: {train_epoch_loss:.4f}, Dice: {epoch_metric_train:.4f}, IoU: {iou_metric_train:.4f}")
        save_loss_train.append(train_epoch_loss)
        save_metric_train.append(epoch_metric_train)

        # === Eval
        if (epoch + 1) % test_interval == 0:
            model.eval()
            test_epoch_loss = 0
            test_step = 0

            with torch.no_grad():
                for image, label, _, _ in test_dataloader:
                    test_step += 1

                    image = image.to(device)
                    label = label.to(device)

                    outputs = model(image)

                    test_loss = loss(outputs, label)
                    test_epoch_loss += test_loss.item()

                    labels_list = decollate_batch(label)
                    labels_convert = [post_label(x) for x in labels_list]

                    output_list = decollate_batch(outputs)
                    output_convert = [post_pred(x) for x in output_list]

                    dice_metric(y_pred=output_convert, y=labels_convert)
                    iou_metric(y_pred=output_convert, y=labels_convert)

            test_epoch_loss /= test_step
            epoch_metric_test = dice_metric.aggregate().mean().item()
            iou_metric_test = iou_metric.aggregate().mean().item()

            dice_metric.reset()
            iou_metric.reset()

            print(f"[TEST] Loss: {test_epoch_loss:.4f}, Dice: {epoch_metric_test:.4f}, IoU: {iou_metric_test:.4f}")
            save_loss_test.append(test_epoch_loss)
            save_metric_test.append(epoch_metric_test)

            if epoch_metric_test > best_metric:
                best_metric = epoch_metric_test
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(model_dir, name + "_best_metric_model.pth"))
                print(f"[INFO] Best model updated and saved at epoch {best_metric_epoch}")

                print("[INFO] Saving EWC data...")
                fisher = {}
                prev_params = {}
                model.eval()
                for image, label, _, _ in dataloader:
                    image = image.to(device)
                    label = label.to(device)
            
                    optim.zero_grad()
                    outputs = model(image)
                    ewc_loss = loss(outputs, label)
                    ewc_loss.backward()
            
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            fisher[name] = param.grad.detach().clone().pow(2)
                            prev_params[name] = param.detach().clone()
                    break  # Estimate Fisher from one batch
            
                save_pickle(fisher, os.path.join(model_dir, "fisher.pkl"))
                save_pickle(prev_params, os.path.join(model_dir, "params.pkl"))
                print(f"[INFO] Saved Fisher and Param snapshots to {model_dir}")



loss_function = monai.losses.DiceFocalLoss(to_onehot_y=True, sigmoid=False,softmax=True,include_background=True)

torch.backends.cudnn.benchmark = True
#optimizer = monai.optimizers.Novograd(model.parameters(), lr=0.001, weight_decay=0.01)

post_label = monai.transforms.AsDiscrete(to_onehot=3)
post_pred = monai.transforms.Compose([
        monai.transforms.AsDiscrete(argmax=True, to_onehot=3, num_classes=3),
        monai.transforms.KeepLargestConnectedComponent(),]
    )
dice_metric = monai.metrics.DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False,ignore_empty=True)
iou_metric=monai.metrics.MeanIoU(include_background=False,reduction="mean_batch",get_not_nans=False,ignore_empty=True)
