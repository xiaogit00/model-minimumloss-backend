from services import db
import unittest
import asyncio
from pathlib import Path
class TestDB(unittest.TestCase):
    # def test_delete_models(self):
    #     # res1 = db.truncate_models()
    #     res = db.delete_models()
    #     print(res)
    def test_insert_model(self):
        modelCode = """
class ShallowNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # By default flattens everything from dim 1 onwards (retains batch size)
        self.fc1 = nn.Linear(784, 16)
        self.fc2 = nn.Linear(16, 10)

    def forward(self, x):
        x = self.flatten(x) # (64, 28, 28) -> (64, 784)
        Z1 = self.fc1(x) # (64, 784) -> (64, 16)
        A1 = torch.relu(Z1) #(64, 16)
        logits = self.fc2(A1) #(64, 10)
        return logits #(64, 10)
"""
        trainingCode = ""
        script_dir = Path(__file__).resolve().parent.parent
        file_path = Path(script_dir / 'models'/'mnist_fashion'/'training_code'/'mnist_fashion_2d_shallowNN.py')
        try:
            with open(file_path, "r") as f:
                trainingCode = f.read()
        except FileNotFoundError:
            print("Error: The file was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")
        print(trainingCode)

        modelData = {
            "slug": 'mnistfashion_shallowNN',
            "name": 'MNIST Fashion Classifier Shallow NN',
            "description": 'A simple shallow neural network with 1 hidden layers, and dimensions D of: \nD_i = 784\nD_1 = 16\nD_o = 10',
            "model_architecture": 'Shallow NN',
            "reflections_url": 'https://blog.minimumloss.xyz/posts/what-i-learned-implementing-a-shallow-neural-network-for-fashionmnist/',
            "dataset_description": 'MNIST Fashion dataset. Training set: 60k. Test set: 10k.',
            "dataset_url": 'https://github.com/zalandoresearch/fashion-mnist',
            "training_code": trainingCode,
            "model_code": modelCode,
            "tags": ['shallowNN', 'MNISTFashion']
        }
        res = db.insert_model(modelData)
        print(res)

if __name__ == '__main__':
    asyncio.run(unittest.main())

