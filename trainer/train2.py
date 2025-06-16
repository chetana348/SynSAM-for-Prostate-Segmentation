from trainer.losses import *
from trainer.metrics import *

network = network.to('cuda')
criterion = DiceLoss2D(num_classes=3)
accuracy_metric = DiceScore2D(num_classes=3)
iou_metric = IoU2D(num_classes=3)

optimizer = torch.optim.Adam(network.parameters(), lr=0.0001, betas=(0.9,0.999))

# Define the directory to save models and TensorBoard logs
save_dir = './wp/tl/cat_uab_on_xnn'
os.makedirs(save_dir, exist_ok=True)


# Training loop
num_epochs = 150
best_accuracy = 0.0

for epoch in range(num_epochs):
    total_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0  # Initialize IoU accumulation
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
    for batch_idx, (images, labels, im_idx, lb_idx) in progress_bar:
        # Move images and labels to GPU
        images = images.cuda()
        labels = labels.cuda()

        # Forward pass
        optimizer.zero_grad()
        outputs = network(images)
        if isinstance(outputs, tuple):
            outputs = outputs[0]
        #outputs = F.interpolate(outputs, size=(128, 128), mode='bilinear', align_corners=True)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        # Calculate accuracy (Dice score)
        accuracy = accuracy_metric(outputs, labels)
        total_accuracy += accuracy.item()

        # Calculate IoU
        iou = iou_metric(outputs, labels)
        total_iou += iou.item()

        progress_bar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), iou=iou.item())  # Update progress bar with current loss, accuracy, and IoU

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    avg_iou = total_iou / len(dataloader)  # Average IoU for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}, IoU: {avg_iou:.4f}')

    # Validation loop to calculate Dice coefficient and IoU
    network.eval()
    total_dice = 0.0
    total_val_loss = 0.0
    total_val_iou = 0.0  # Initialize IoU accumulation for validation
    progress_bar_val = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc="Validation", leave=False)
    with torch.no_grad():
        for batch_idx, (images_test, labels_test, im_idx, lb_idx) in progress_bar_val:
            images_test = images_test.cuda()
            labels_test = labels_test.cuda()
            outputs_test = network(images_test)
            #print(outputs_tests.shape)
            #outputs_test= F.interpolate(outputs_test, size=(128, 128), mode='bilinear', align_corners=True)
            val_loss = criterion(outputs_test, labels_test)
            dice = accuracy_metric(outputs_test, labels_test)
            iou = iou_metric(outputs_test, labels_test)
            total_dice += dice.item()
            total_val_loss += val_loss.item()
            total_val_iou += iou.item()
            progress_bar_val.set_postfix(loss=val_loss.item(), dice=dice.item(), iou=iou.item())  # Update progress bar with current loss, dice, and IoU

    avg_dice = total_dice / len(test_dataloader)
    avg_val_loss = total_val_loss / len(test_dataloader)
    avg_val_iou = total_val_iou / len(test_dataloader)  # Average IoU for validation
    print(f'Average Dice coefficient: {avg_dice:.4f}')
    print(f'Validation Loss: {avg_val_loss:.4f}')
    print(f'Validation IoU: {avg_val_iou:.4f}')

    # Check if current accuracy is the best so far
    if avg_dice > best_accuracy:
        best_accuracy = avg_dice
        best_model_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(network.state_dict(), best_model_path)
        print(f"New best model saved with accuracy {avg_dice:.4f}")

    network.train()

writer.close()
print("Training finished.")
