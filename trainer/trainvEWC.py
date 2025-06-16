import os
import torch
import pickle
from tqdm import tqdm
import monai
from monai.data import decollate_batch
from trainer.losses import *
from trainer.metrics import *
from trainer.losses import *  # Import vEWC


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def trainv2D(model, dataloader, test_dataloader, loss, optim, max_epochs, model_dir, device, name, test_interval=1, cl_dir=None, lambda_ewc=0.4):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = []
    save_metric_test = []

    fisher, prev_params, scores = {}, {}, {}
    use_vewc = False
    if cl_dir and all(os.path.exists(os.path.join(cl_dir, f + ".pkl")) for f in ["fisher", "params", "scores"]):
        fisher = load_pickle(os.path.join(cl_dir, "fisher.pkl"))
        prev_params = load_pickle(os.path.join(cl_dir, "params.pkl"))
        scores = load_pickle(os.path.join(cl_dir, "scores.pkl"))
        vewc_loss_fn = vEWC(reg_strength=lambda_ewc, fisher_scores=fisher, prev_params=prev_params, importance_scores=scores)
        use_vewc = True
        print(f"[INFO] Loaded vEWC with tasks: {list(fisher.keys())[:-1]}")

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        train_epoch_loss = 0
        train_step = 0

        with tqdm(dataloader, unit="batch") as tepoch:
            for image, label, _, _ in tepoch:
                tepoch.set_description(f"Epoch {epoch + 1}")
                train_step += 1

                image = image.to(device)
                label = label.to(device)

                optim.zero_grad()
                outputs = model(image)
                train_loss = loss(outputs, label)

                if use_vewc:
                    train_loss += vewc_loss_fn.compute(model.named_parameters())

                train_loss.backward()
                optim.step()

                train_epoch_loss += train_loss.item()

                labels_list = decollate_batch(label)
                labels_convert = [post_label(x) for x in labels_list]
                output_list = decollate_batch(outputs)
                output_convert = [post_pred(x) for x in output_list]
                dice_metric(y_pred=output_convert, y=labels_convert)
                iou_metric(y_pred=output_convert, y=labels_convert)
                dice_vals = dice_metric.aggregate()
                mean_dice = dice_vals.mean().item()
                tepoch.set_postfix(loss=train_loss.item(), dice_score=mean_dice)

        train_epoch_loss /= train_step
        epoch_metric_train = dice_metric.aggregate().mean().item()
        iou_metric_train = iou_metric.aggregate().mean().item()
        dice_metric.reset()
        iou_metric.reset()
        print(f"Train Loss: {train_epoch_loss:.4f}, Dice: {epoch_metric_train:.4f}, IoU: {iou_metric_train:.4f}")
        save_loss_train.append(train_epoch_loss)
        save_metric_train.append(epoch_metric_train)

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

                #print("[INFO] Saving vEWC data...")
                #fisher[name] = {}
                #prev_params[name] = {}
                #scores[name] = {}
                #model.eval()
                #for image, label, _, _ in dataloader:
                #    image = image.to(device)
                #    label = label.to(device)
                #    optim.zero_grad()
                #    outputs = model(image)
                #    ewc_loss = loss(outputs, label)
                #    ewc_loss.backward()
                #    for n, p in model.named_parameters():
                #        if p.grad is not None:
                #            fisher[name][n] = p.grad.detach().clone().pow(2)
                #            prev_params[name][n] = p.detach().clone()
                #            scores[name][n] = torch.abs(p.grad.detach() * (prev_params[name][n] - p.detach()))
                #    break

                #save_pickle(fisher, os.path.join(model_dir, "fisher.pkl"))
                #save_pickle(prev_params, os.path.join(model_dir, "params.pkl"))
                #save_pickle(scores, os.path.join(model_dir, "scores.pkl"))
                #print(f"[INFO] Saved vEWC data to {model_dir}")

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