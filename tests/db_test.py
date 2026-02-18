from services import db
import unittest
import asyncio

class TestDB(unittest.TestCase):
    # def test_delete_models(self):
    #     res = db.delete_models()
    #     print(res)
    def test_insert_model(self):
        modelData = {
            "slug": 'mnistfashion_shallowNN',
            "name": 'MNIST Fashion Classifier Shallow NN',
            "description": 'A model for classifing 10 categories of fashion images. Trained with MNIST Fashion dataset. Optimizer: SGD. Loss Fn: Cross Entropy loss.',
            "dataset_description": 'MNIST Fashion dataset. Training set: 60k. Test set: 10k.',
            "dataset_link": 'https://github.com/zalandoresearch/fashion-mnist',
            "training_code_link": 'https://github.com/xiaogit00/model-minimumloss-backend/blob/main/models/mnist_fashion/training_code/mnist_fashion_2d_shallowNN.py',
            "tags": ['shallowNN', 'MNISTFashion']
        }
        res = db.insert_model(modelData)
        print(res)

if __name__ == '__main__':
    asyncio.run(unittest.main())

