
from torch.utils.data import Dataset, Subset

class iCaRLDataset(Dataset):
    """
    Dataset that combines new task data with stored exemplars.
    """
    def __init__(self, new_data, exemplars, transform=None):
        self.new_data = new_data # List of (image, label) tuples
        self.exemplars = exemplars # List of lists of images
        self.transform = transform

        # Flatten exemplars into a list of (img, label)
        self.exemplar_data = []
        for label, img_list in enumerate(exemplars):
            for img in img_list:
                self.exemplar_data.append((img, label))

        self.all_data = self.new_data + self.exemplar_data

    def __getitem__(self, index):
        img, label = self.all_data[index]
        # img is a Tensor here if coming from CIFAR100(ToTensor),
        # but iCaRL usually stores raw images.
        # For simplicity in this script, we assume img is already Tensor from prev loader
        # If transform is needed, apply here.
        return img, label

    def __len__(self):
        return len(self.all_data)

def get_data_for_classes(dataset, classes):
    """
    Extracts all samples belonging to specific classes.
    """
    indices = [i for i, label in enumerate(dataset.targets) if label in classes]
    return Subset(dataset, indices)

def extract_images_from_subset(subset):
    """
    Helper to pull images out of a Subset for exemplar storage.
    """
    images = []
    # This is slow for large sets, efficient implementation would use indices directly
    # But for a tutorial script, iterating is safe.
    for i in range(len(subset)):
        img, _ = subset[i]
        images.append(img)
    return images
