import torch
import torch.nn as nn
from utils import *

# Check for the appropriate device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Mapping from ImageNet-1000 classes to custom 200 classes
thousand_k_to_200 = {
    0: -1, 1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: 0, 7: -1, 8: -1, 9: -1,
    # ... [truncated for brevity] ...
    987: 198, 988: 199, 989: -1, 990: -1, 991: -1, 992: -1, 993: -1, 994: -1, 995: -1, 996: -1, 997: -1, 998: -1, 999: -1
}

# Extract the indices of the relevant classes
indices_in_1k = [k for k in thousand_k_to_200 if thousand_k_to_200[k] != -1]

def test_all_dataset(model, data_loader, cost_function):
    """
    Evaluate the model on the entire dataset.
    
    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the dataset.
        cost_function (callable): The loss function.
        
    Returns:
        float: Average loss per sample.
        float: Accuracy percentage.
    """
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    model.eval()  # Set the model to evaluation mode

    # Disable gradient computation
    with torch.no_grad():
        # Iterate over the test set
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs)
            outputs = outputs[:, indices_in_1k]

            # Loss computation
            loss = cost_function(outputs, targets)

            # Update counters
            samples += inputs.shape[0]
            cumulative_loss += loss.item()
            _, predicted = outputs.max(1)
            cumulative_accuracy += predicted.eq(targets).sum().item()

    return cumulative_loss / samples, cumulative_accuracy / samples * 100

def tune_model(image, model, mask_generator, optimizer, cost_function, num_aug, flag_memo_plus=False, centroid=None):
    """
    Fine-tune the model using MEMO and MEMO_PLUS techniques.
    
    Args:
        image (numpy array): The input image.
        model (torch.nn.Module): The model to fine-tune.
        mask_generator (callable): Function to generate masks.
        optimizer (torch.optim.Optimizer): The optimizer.
        cost_function (callable): The loss function.
        num_aug (int): Number of augmentations.
        flag_memo_plus (bool): Flag to indicate whether to use MEMO_PLUS.
        centroid (tuple, optional): The centroid for augmentation.
        
    Returns:
        list: List of augmented images.
        list: List of augmentation names.
    """
    model.eval()
    aug, names = apply_augmentations(image, num_aug, centroid)

    if flag_memo_plus:
        aug = segmentImage(aug, mask_generator, (100, 100))

    segmented_aug = transform_images(aug)

    input_tensor = torch.stack(segmented_aug).to(device)
    optimizer.zero_grad()
    output = model(input_tensor)
    output = torch.mean(output, dim=0).unsqueeze(0)
    output = output[:, indices_in_1k]
    loss, _ = cost_function(output)

    loss.backward()
    optimizer.step()
    
    return aug, names

def test_model(image, target, model):
    """
    Test the model on a single image.
    
    Args:
        image (numpy array): The input image.
        target (int): The target class.
        model (torch.nn.Module): The model to test.
        
    Returns:
        tuple: Correctness (1 if correct, 0 otherwise) and confidence.
    """
    model.eval()
    inputs = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(inputs)
        outputs = outputs[:, indices_in_1k]
        _, predicted = outputs.max(1)
        confidence = nn.functional.softmax(outputs, dim=1).squeeze()[predicted].item()
    
    correctness = 1 if predicted.item() == target else 0
    return correctness, confidence

def marginal_entropy(outputs):
    """
    Compute the marginal entropy loss.
    
    Args:
        outputs (torch.Tensor): The model outputs.
        
    Returns:
        tuple: Marginal entropy loss and averaged logits.
    """
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1), avg_logits
