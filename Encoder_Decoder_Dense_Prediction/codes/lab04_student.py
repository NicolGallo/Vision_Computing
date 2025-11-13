"""
Lab 4: Encoder-Decoder Architectures for Dense Prediction
Student Version - Complete the TODOs

Author: [Your Name]
Date: [Current Date]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import OxfordIIITPet
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from lab04_utils import *
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ================== Part 1: U-Net Architecture ==================

class DoubleConv(nn.Module):
    """Two consecutive convolution layers with BatchNorm and ReLU"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Task 1.1: Implement double convolution block
        # Hint: Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU
        # Use kernel_size=3, padding=1 to maintain spatial dimensions
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    """Encoder block with double convolution and pooling"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Task 1.1: Initialize layers
        # You need: DoubleConv and MaxPool2d
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Return both: features before pooling (for skip connection) and after pooling
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class DecoderBlock(nn.Module):
    """Decoder block with upsampling, skip connection, and double convolution"""
    
    def __init__(self, in_channels, skip_channels, out_channels, upsampling='transpose'):
        super().__init__()
        self.upsampling = upsampling
        
        # Task 1.2: Initialize upsampling layer
        if upsampling == 'transpose':
            self.up = nn.ConvTranspose2d(in_channels,out_channels//2, kernel_size=2, stride=2)
        else:  # bilinear
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up_conv2d = nn.Conv2d(in_channels, out_channels//2, kernel_size=1)  #Use Upsample + Conv2d
            
        # Task 1.2: Initialize double convolution
        # Note: Input will be concatenated features (in_channels + skip_channels)
        self.conv = DoubleConv(in_channels=out_channels//2 + skip_channels, out_channels=out_channels)
        
    def forward(self, x, skip_features):
        # Task 1.2: Implement forward pass
        # 1. Upsample x
        # 2. Handle dimension mismatch if necessary (crop or pad)
        # 3. Concatenate with skip_features
        # 4. Apply double convolution
        
        if self.upsampling == 'transpose':
            x = self.up(x)
        else:
            x = self.upsample(x)
            x = self.up_conv2d(x)
        
        # Handle dimension mismatch
        if x.size() != skip_features.size():
            x = F.pad(x, [0, skip_features.size(3) - x.size(3), 0, skip_features.size(2) - x.size(2)])
        
        x = torch.cat((skip_features, x), dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    """Complete U-Net architecture"""
    
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()
        
        # Task 1.3: Build encoder path
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        # Create encoder blocks
        # Hint: First block takes in_channels, others take features[i-1]
        for feature in features:
            if len(self.encoders) == 0:
                self.encoders.append(EncoderBlock(in_channels, feature))
            else:
                self.encoders.append(EncoderBlock(features[len(self.encoders)-1], feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        
        # Task 1.3: Build decoder path
        self.decoders = nn.ModuleList()
        
        # Create decoder blocks (in reverse order)
        for i in range(len(features)-1, -1, -1):
            if i == len(features)-1:
                self.decoders.append(DecoderBlock(features[-1]*2, features[i], features[i]))
            else:
                self.decoders.append(DecoderBlock(features[i+1], features[i], features[i]))
        
        # Task 1.4: Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        # Task 1.4: Connect everything together
        skip_connections = []
        
        # Encoder path
        # Process through encoders, save skip connections
        for encoder in self.encoders:
            features, x = encoder(x)
            skip_connections.append(features)
        
        # Bottleneck
        # Process through bottleneck
        x = self.bottleneck(x)
        
        # Decoder path
        # Process through decoders with skip connections
        skip_connections = skip_connections[::-1]  # Reverse for decoding
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
        
        # Final layer
        # Apply final convolution
        x = self.final_conv(x)
        
        return x


# ================== Part 2: Skip Connection Strategies ==================

class AttentionGate(nn.Module):
    """Attention gate for skip connections"""
    
    def __init__(self, gate_channels, skip_channels):
        super().__init__()
        # Task 2.3: Implement attention gate
        # You need: Conv2d layers for gating signal, skip features, and psi

        # self.W_g: Conv11(Y, Cd → Cg) -> Procesar la señal de Gating (Y). Se reduce los canales del decoder de Cd​ a un número intermedio Cg​. La capa g actúa como la señal guía.
        self.W_g = nn.Conv2d(in_channels=gate_channels, out_channels=gate_channels//2, kernel_size=1)  # Conv2d for gating signal

        # self.W_x: Conv11(F, Cs → Cg) -> Procesar las características Skip (F) del encoder. Se reduce los canales del encoder de Cs​ a Cg, emparejando la dimensión de canales con g.​
        # La capa x actúa como la señal de entrada.
        self.W_x = nn.Conv2d(in_channels=skip_channels, out_channels=gate_channels//2, kernel_size=1)  # Conv2d for skip features

        # self.psi: Conv11(q, Cg → 1) -> Generar el mapa de atención final (α). Una convolución final 1×1 reduce los canales de q (de Cg​) a un solo canal (1) por cada píxel.
        self.psi = nn.Conv2d(in_channels=gate_channels//2, out_channels=1, kernel_size=1)  # Conv2d for final attention coefficients

        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, gate, skip):
        # Task 2.3: Implement attention mechanism
        # 1. Process gate and skip through respective convolutions
        g = self.W_g(gate)
        x = self.W_x(skip)
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        # 2. Add and apply ReLU
        q = self.relu(g + x) 

        # 3. Apply psi convolution and sigmoid
        #La función Sigmoid mapea estos valores al rango [0,1], creando un mapa de pesos de atención (α) donde 1 significa "muy relevante" y 0 significa "irrelevante".
        alpha = torch.sigmoid(self.psi(q))  # Attention coefficients of the attention map.
        
        # 4. Multiply with skip features
        # Fatt = alpha * F (Modulación por Broadcasting)
        # Este es el paso de modulación. Los pesos α se multiplican elemento a elemento con las características del encoder (F).
        # La señal F se atenúa o suprime en las regiones donde α es bajo, y se preserva donde α es alto, filtrando el ruido irrelevante.
        Fatt = skip * alpha
        
        return Fatt


class FlexibleSkipConnection(nn.Module):
    """Flexible skip connection with different strategies"""
    
    def __init__(self, decoder_channels, skip_channels, mode='concat'):
        super().__init__()
        self.mode = mode
        
        if mode == 'concat':
            # Task 2.1: Setup for concatenation
            # Output conv to handle concatenated channels
            self.conv = nn.Conv2d(decoder_channels + skip_channels, decoder_channels, kernel_size=3, padding=1)
            
        elif mode == 'add':
            # Task 2.2: Setup for addition
            # May need 1x1 conv to match channels
            if skip_channels != decoder_channels:
                self.proj = nn.Conv2d(skip_channels, decoder_channels, kernel_size=1)  # Handle channel mismatch
            else:
                self.proj = nn.Identity()

        elif mode == 'attention':
            # Task 2.3: Setup attention gate
            self.attention_gate = AttentionGate(decoder_channels, skip_channels)
            self.conv = nn.Conv2d(decoder_channels + skip_channels, decoder_channels, kernel_size=3, padding=1)  # Output conv after attention
            
    def forward(self, decoder_features, skip_features):
        # Implement forward pass based on mode
        if self.mode == 'concat':
            # Task 2.1
            return self.conv(torch.cat((decoder_features, skip_features), dim=1))
        
        elif self.mode == 'add':
            # Task 2.2
            return decoder_features + self.proj(skip_features)
        
        elif self.mode == 'attention':
            # Task 2.3
            gated = self.attention_gate(decoder_features, skip_features)
            return self.conv(torch.cat((decoder_features, gated), dim=1))


# ================== Part 3: Loss Functions ==================

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Task 3.1: Implement Dice loss
        # 1. Apply sigmoid to predictions to get probabilities
        P = torch.sigmoid(pred)

        # Obtener las dimensiones
        B, C, H, W = P.shape
        
        LDice_total = 0.0 # Inicializamos la pérdida total

        # 2. Flatten both pred and target
        # Iterar sobre las clases (canales) y el batch
        for b in range(B):
            for c in range(C):
                pred_flat = P[b, c, :, :]      # Aplanar las predicciones
                target_flat = P[b, c, :, :]  # Aplanar las etiquetas

                # 3. Compute intersection and union
                intersection = torch.sum(pred_flat * target_flat)  # Intersección
                union = torch.sum(pred_flat) + torch.sum(target_flat)  # Unión

                # 4. Calculate Dice coefficient
                dice = (2. * intersection + self.smooth) / (union + self.smooth)  # Coeficiente de Dice
                LDice_total += (1 - dice)  # Acumular la pérdida de Dice
        
        # 5. Return 1 - dice
        # LDice ← LDice / C (Promediamos sobre todas las muestras y clases)
        # Dividimos por el número total de combinaciones (Batch * Clases)
        LDice_avg = LDice_total / (B * C)

        return LDice_avg


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, pred, target):
        # Task 3.2: Implement Focal loss
        # 1. Apply sigmoid to get probabilities and compute BCE
        P = torch.sigmoid(pred)
        BCE = nn.CrossEntropyLoss(pred, target, reduction='none')

        # 2. Calculate p_t (probability of correct class)
        p_t = target * P + (1 - target) * (1 - P)
        # p_t = torch.exp(-BCE) # Alternatively, use exp(-BCE) to get p_t

        # 3. Apply focal term: (1-p_t)^gamma
        focal_term = (1 - p_t) ** self.gamma

        # 4. Apply alpha weighting
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        
        focal_loss = alpha_t * focal_term * BCE
        focal_loss = focal_loss.mean()  # Average over batch
        return focal_loss


class CombinedLoss(nn.Module):
    """Combined loss function"""
    
    def __init__(self, weights={'ce': 0.5, 'dice': 0.5, 'focal': 0.0}):
        super().__init__()
        # Task 3.3: Initialize component losses
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.weights = weights
        
    def forward(self, pred, target):
        # Task 3.3: Compute weighted combination of losses
        # Add each loss component with its weight
        
        ce = self.ce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)
        return self.weights['ce'] * ce + self.weights['dice'] * dice + self.weights['focal'] * focal


# ================== Training and Evaluation Functions ==================

def calculate_iou(pred, target, threshold=0.5):
    """Calculate Intersection over Union"""
    # Task 3.4: Implement IoU calculation
    # 1. Threshold predictions
    P = torch.sigmoid(pred)

    # Obtener las dimensiones
    B, C, H, W = P.shape
        
    total_iou = 0.0 # Inicializamos la pérdida iou total
    for b in range(B):
        for c in range(C):
            P_binary = (P[b, c, :, :] > threshold)
            target_binary = (target[b, c, :, :] > threshold)

            # 2. Calculate intersection and union
            true_positive = (P_binary & target_binary).float().sum()  # Intersection
            false_positive = (P_binary & ~target_binary).float().sum()
            false_negative = (~P_binary & target_binary).float().sum()

            intersection = true_positive
            union = true_positive + false_positive + false_negative

            iou = (intersection + 1e-6) / (union + 1e-6)  # IoU score with smoothing
            total_iou += iou
    
    # 3. Return IoU score
    
    iou_score = total_iou / (B * C)  # Promediamos sobre todas las muestras y clases
    return iou_score


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_iou = 0.0
    
    # Complete training loop
    for images, masks in tqdm(dataloader, desc='Training'):
        images, masks = images.to(device), masks.to(device)
        
        # Forward pass
        predictions = model(images)

        # Calculate loss
        loss = criterion(predictions, masks)
        loss_iou = calculate_iou(predictions, masks)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        #clip_grad_norm(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()

        # Calculate metrics
        total_loss = total_loss + loss.item()
        total_iou = total_iou + loss_iou.item()
    
    return total_loss / len(dataloader), total_iou / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        # Complete validation loop
        for images, masks in tqdm(dataloader, desc='Validation'):
            images, masks = images.to(device), masks.to(device)
            
            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, masks)
            loss_iou = calculate_iou(predictions, masks)

            # Calculate loss and metrics
            total_loss = total_loss + loss.item()
            total_iou = total_iou + loss_iou.item()

    
    return total_loss / len(dataloader), total_iou / len(dataloader)


def visualize_predictions(model, dataloader, device, num_samples=4):
    """Visualize model predictions"""
    model.eval()
    
    # Implement visualization
    # 1. Get a batch of images and masks
    # Obtener el primer batch del DataLoader
    # Usamos next(iter(dataloader)) para obtener el primer batch de datos.
    with torch.no_grad():
        images, masks = next(iter(dataloader))
        images, masks = images.to(device), masks.to(device)

    # 2. Generate predictions
        # Generamos las predicciones (logits)
        predictions = model(images)
        
        # Opcional: limitar el número de muestras para la visualización
        batch_size = images.size(0)
        num_to_plot = min(num_samples, batch_size)
        
        # Seleccionar solo las muestras que se van a visualizar
        images_to_plot = images[:num_to_plot]
        masks_to_plot = masks[:num_to_plot]
        predictions_to_plot = predictions[:num_to_plot]

    # 3. Create subplot showing: input, ground truth, prediction
    fig = plot_segmentation_results(images=images_to_plot, masks=masks_to_plot, predictions=predictions_to_plot, title=f"Segmentation Results (First {num_to_plot} Samples)")
    plt.show()

    return fig

def plot_training_curves(train_losses, val_losses, train_ious, val_ious, epochs, filename='training_curves_practice.png'):
    """
    Genera dos gráficos: Pérdida vs. Época y IoU vs. Época.
    """
    epochs_range = range(1, epochs + 1)
    
    # ----------------------------------
    # GRÁFICO 1: Pérdida (Loss)
    # ----------------------------------
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    
    plt.plot(epochs_range, train_losses, label='Training Loss', marker='o', linestyle='-')
    plt.plot(epochs_range, val_losses, label='Validation Loss', marker='o', linestyle='--')
    
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    # ----------------------------------
    # GRÁFICO 2: IoU (Métrica)
    # ----------------------------------
    plt.subplot(1, 2, 2)
    
    plt.plot(epochs_range, train_ious, label='Training IoU', marker='o', linestyle='-')
    plt.plot(epochs_range, val_ious, label='Validation IoU', marker='o', linestyle='--')
    
    plt.title('IoU vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle("Model Performance Over Training Epochs", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Ajustar para el suptitle

    # --- GUARDAR EL GRÁFICO ---
    try:
        plt.savefig(filename)
        print(f"✅ Training curves saved to {filename}")
    except Exception as e:
        print(f"Error saving figure: {e}")

    plt.show()


# ================== Main Training Script ==================

def main():
    # Hyperparameters
    config = {
        'batch_size': 16,
        'learning_rate': 0.001,
        'epochs': 50,
        'image_size': 128,
        'skip_mode': 'concat',  # Try: 'concat', 'add', 'attention'
    }
    
    # Setup data transforms
    image_transform = transforms.Compose([
        # Add necessary transforms
        # Resize, ToTensor, Normalize
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor()
    ])
    
    # Load dataset
    # Use OxfordIIITPet or a simple synthetic dataset for testing
    dataset = OxfordIIITPet(root='./data', split='trainval', target_types='segmentation', transform=image_transform, target_transform=mask_transform, download=True)
    val_size = int(0.2 * len(dataset))
    test_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size - test_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Initialize model
    model = UNet(in_channels=3, out_channels=1).to(device)
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = CombinedLoss()
    
    # Training loop
    train_losses = []
    val_losses = []
    train_ious = []
    val_ious = []
    best_iou = -1.0

    print("Starting training...")
    for epoch in range(config['epochs']):
        # Train and validate
        train_loss, train_iou = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou = validate(model, val_loader, criterion, device)
        
        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_ious.append(train_iou)
        val_ious.append(val_iou)

        # Print progress
        print(f"Epoch [{epoch+1}/{config['epochs']}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train IoU: {train_iou:.4f}, Val IoU: {val_iou:.4f}")
        
        # Save best model
        if val_iou > best_iou:
            best_iou = val_iou
            # Guardar los pesos del modelo.
            torch.save(model.state_dict(), 'best_unet_model.pth') 
            print(">>> Saved best model!")
    
    # Plot training curves
    print("Visualizing training curves about losses and ious scores...")
    plot_training_curves(train_losses, val_losses, train_ious, val_ious, config['epochs'])

    # Visualize final predictions
    print("Visualizing final predictions...")
    # Usamos el modelo y el cargador de datos de validación
    visualize_predictions(model, val_loader, device, num_samples=4) 

    print("Training complete!")


# ================== Analysis Functions ==================

def analyze_skip_connections():
    """Compare different skip connection strategies"""
    # TODO Task 2.4: Implement comparison
    # 1. Train models with different skip modes
    # 2. Compare gradient flow
    # 3. Compare memory usage
    # 4. Compare final performance
    
    results = {}
    
    # TODO: Run experiments for each mode
    for mode in ['concat', 'add', 'attention']:
        # TODO: Train model
        # TODO: Collect metrics
        pass
    
    # TODO: Create comparison table/plot
    
    return results


def ablation_study():
    """Perform ablation study on U-Net components"""
    # TODO: Implement ablation study
    # Test: no skip connections, no batch norm, different depths
    
    ablation_results = {}
    
    # TODO: Run different configurations
    
    return ablation_results


if __name__ == "__main__":
    # Run main training
    main()
    
    # Run analysis (optional)
    # analyze_skip_connections()
    # ablation_study()