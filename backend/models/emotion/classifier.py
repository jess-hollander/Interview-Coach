# cnn model for emotion classification
import torch
import torch.nn as nn
import torch.nn.functional as F

# image settings and emotion labels
IMG_SIZE = 48
NUM_CHANNELS = 1
CLASS_LABELS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


# convolutional neural network for grayscale face images
class ConvNet(nn.Module):

    # sets up the layers with dropout to prevent overfitting
    def __init__(
        self,
        num_classes: int = 10,
        dropout_features: float = 0.25,
        dropout_classifier: float = 0.5,
    ) -> None:
        super().__init__()
        # feature extraction layers - indices must match the saved checkpoint
        # checkpoint has conv at 0,4,7,11 and bn at 1,5,8,12
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   
            nn.BatchNorm2d(32),                            
            nn.ReLU(inplace=True),                         
            nn.MaxPool2d(kernel_size=2),                   
            nn.Conv2d(32, 64, kernel_size=3, padding=1),   
            nn.BatchNorm2d(64),                            
            nn.ReLU(inplace=True),                         
            nn.Conv2d(64, 64, kernel_size=3, padding=1),   
            nn.BatchNorm2d(64),                            
            nn.ReLU(inplace=True),                         
            nn.MaxPool2d(kernel_size=2),                   
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  
            nn.BatchNorm2d(128),                           
            nn.ReLU(inplace=True),                         
            nn.Dropout2d(dropout_features),                
        )
        # fully connected layers to output class predictions
        self.classifier = nn.Sequential(
            nn.Flatten(),                                  
            nn.Linear(128 * 12 * 12, 256),                 
            nn.BatchNorm1d(256),                           
            nn.ReLU(inplace=True),                         
            nn.Dropout(dropout_classifier),                
            nn.Linear(256, num_classes),                   
        )

        # initialize weights using kaiming method
        self.apply(self._init_weights)

    # initializes conv and linear layers with kaiming normal
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initial weights."""
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # runs the image through the network and returns logits
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        logits = self.classifier(x)
        return logits

    # counts total params in the model
    def num_parameters(self, trainable_only: bool = True) -> int:
        """Convenience helper to count model parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
