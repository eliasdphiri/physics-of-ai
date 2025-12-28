Chapter 5: Vision and Optics: Convolutional Neural Networks (CNNs)In This ChapterWhy standard networks are blind to shapesThe Physics of Images: Tensors and PhotonsThe Convolution Operation: Scanning with LensesMax Pooling: Reducing Entropy and NoiseLab: Building an Optical Bench in PyTorchIn Chapter 4, we built a Multi-Layer Perceptron (MLP). It was great at logic (XOR), but if you showed it a picture of a cat, it would fail miserably.Why? Because an MLP flattens the world. If you feed a $28 \times 28$ pixel image into an MLP, you have to crush it into a single line of 784 numbers. This destroys all spatial relationships. The network no longer knows that Pixel A is next to Pixel B. It loses the concept of "up," "down," and "shape."To fix this, we need a new architecture that respects the geometry of space. We need Convolutional Neural Networks (CNNs).The Physics of ImagesTo a computer, an image is not a picture; it is a Tensor. A tensor is just a multi-dimensional matrix.Height ($H$): Vertical spatial dimension.Width ($W$): Horizontal spatial dimension.Channels ($C$): The spectrum of light.Grayscale = 1 Channel (Intensity)RGB = 3 Channels (Red, Green, Blue)So, a standard color photo is a block of matter with dimensions $C \times H \times W$.The Convolution: The Sliding LensImagine you are searching for a specific pattern—say, a vertical edge—in a dark room using a flashlight. You don't look at the whole room at once. You sweep the flashlight from top-left to bottom-right.This sweeping motion is Convolution.The "flashlight" is called a Kernel or Filter. It is a small matrix (usually $3 \times 3$ or $5 \times 5$) containing weights.The Math of the Scan:Place the $3 \times 3$ Kernel over a $3 \times 3$ patch of the image.Multiply the pixels by the kernel weights (element-wise).Sum them up to get a single number.Slide the Kernel one step to the right (Stride) and repeat.THEORY LAB: Think of the Kernel as an Interference Pattern. If the image patch matches the pattern on the Kernel, the signal amplifies (Constructive Interference). If they don't match, the signal cancels out.A "Vertical Edge" kernel will light up only when it passes over a vertical line.A "Eye Detector" kernel will light up only when it passes over an eye.Pooling: Thermodynamics and EntropyAfter convolution, we have a map of where the features are. But this map is still big and noisy. We need to compress it. We need to reduce the Entropy (disorder) and keep only the most important signals.We use Max Pooling.We take a $2 \times 2$ window and look at the four values inside. We pick the maximum value and throw the other three away.Physics: We are discarding low-energy noise and preserving the high-energy signal "spikes."Result: The image shrinks by half in height and width ($H/2, W/2$), making the network faster and less prone to overfitting.Lab 5.1: The Shape InspectorConstructing a CNN requires careful management of dimensions. If you try to push a square peg into a round hole (mismatched tensor shapes), PyTorch will crash.Let's build a CNN that simulates a simple visual cortex.The Code:Pythonimport torch
import torch.nn as nn

# 1. The Optical Bench (The Model)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Layer 1: The Retina
        # We take 1 input channel (Grayscale) and output 8 Feature Maps.
        # Kernel size 3x3 acts as our lens.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.relu = nn.ReLU() # The activation (allows light to pass or blocks it)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # Downsample by half
        
        # Layer 2: Visual Cortex V1
        # Input is now 8 channels (from prev layer). We expand to 16 feature maps.
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        
        # Layer 3: Decision Making (Fully Connected)
        # We must FLATTEN the 3D tensor into a 1D vector to feed the final classifier.
        # We need to calculate the size: 16 channels * H * W.
        # If input is 28x28 -> pool -> 14x14 -> pool -> 7x7.
        self.fc = nn.Linear(16 * 7 * 7, 10) # 10 Output classes (e.g., digits 0-9)

    def forward(self, x):
        # Pass through Layer 1
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x) # Image size drops from 28->14
        
        # Pass through Layer 2
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x) # Image size drops from 14->7
        
        # Flatten: Reshape 3D block (16, 7, 7) to 1D line (784)
        x = x.view(x.size(0), -1) 
        
        # Classification
        x = self.fc(x)
        return x

# 2. Run a Photon Simulation
model = SimpleCNN()

# Create a dummy image: Batch Size 1, 1 Channel, 28x28 Pixels
dummy_image = torch.randn(1, 1, 28, 28)

# Pass light through the lens
output = model(dummy_image)

print(f"Input Shape: {dummy_image.shape}")
print(f"Output Shape: {output.shape}") 
# Output should be [1, 10] -> Probabilities for 10 classes
Analyzing the Optical Path:Input: A $28 \times 28$ grid of intensity values.Conv1: Scans the grid, creating 8 "maps" of features (edges, corners).Pool1: Shrinks the maps to $14 \times 14$ to save processing power.Conv2: Scans the simpler maps to find complex combinations (shapes, loops).Pool2: Shrinks to $7 \times 7$.FC: Looks at the combination of shapes and decides "That is the number 5."Translational InvarianceA major property of physics is symmetry. If I drop a ball in New York, it falls down. If I drop it in London, it falls down. Physics is invariant to location.CNNs share this property. Because the kernel slides across the whole image, it can detect a cat whether the cat is in the top-left corner or the bottom-right corner.Standard MLPs do not have this. If you train an MLP on centered cats, it will not recognize a cat in the corner.CNNs are robust to translation.ExercisesThe Math of Shapes: If you have an input image of $32 \times 32$ and you apply a kernel_size=5 with stride=1 and no padding, what is the output size?Formula: $Output = (Input - Kernel) + 1$Color Vision: Change the code above to accept color images. You need to change in_channels in conv1 from 1 to 3.Manual Filter: (Advanced) Create a $3 \times 3$ tensor in PyTorch that represents a "Vertical Edge Detector."Hint: The left column should be positive (1), the middle zero (0), and the right negative (-1).