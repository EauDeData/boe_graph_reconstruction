from src.data.datasets import BOEDataset
import pytesseract

test_set = BOEDataset('test.txt').crop_dataset()
train_set = BOEDataset('train.txt').crop_dataset()
