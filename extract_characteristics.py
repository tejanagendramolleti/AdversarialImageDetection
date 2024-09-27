print('Load modules...')
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from scipy.spatial.distance import cdist
from tqdm import tqdm
from collections import OrderedDict
from models.vgg_mnist import VGG  # Use the MNIST version of VGG
import argparse
import sklearn
import sklearn.covariance

# Processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--attack", default='fgsm', help="the attack method which created the adversarial examples you want to use. Either fgsm, bim, pgd, df or cw")
parser.add_argument("--detector", default='InputMFS', help="the detector you want to use, out of InputMFS, InputPFS, LayerMFS, LayerPFS, LID, Mahalanobis")
parser.add_argument("--net", default='mnist', help="the network used for the attack, either mnist")
args = parser.parse_args()

# Choose attack
attack_method = args.attack
detector = args.detector
net = args.net

# Load adversarials and their non-adversarial counterpart
print('Loading images and adversarial examples...')
images = torch.load('./data/' + net + '_adversarial_images/' + net + '_images_' + attack_method, map_location=torch.device('cpu'))
images_advs = torch.load('./data/' + net + '_adversarial_images/' + net + '_images_adv_' + attack_method, map_location=torch.device('cpu'))
number_images = len(images)

# Load model VGG16
print('Loading model...')
if net == 'mnist':
    model = VGG('VGG16')
    checkpoint = torch.load('./models/vgg_mnist.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['net'])
else:
    print('unknown model')
model = model.eval()
device = torch.device("cpu")
model = model.to(device)

# Get a list of all feature maps of all layers
model_features = model.features

def get_layer_feature_maps(X, layers):
    X_l = []
    for i in range(len(model_features)):
        X = model_features[i](X)
        if i in layers:
            Xc = torch.Tensor(X.cpu())
            X_l.append(Xc)
    return X_l

# Normalization for MNIST
def mnist_normalize(images):
    images = (images - 0.1307) / 0.3081
    return images

# Indices of activation layers
act_layers = [2, 5, 9, 12, 16, 19, 22, 26, 29, 32, 36, 39, 42]
fourier_act_layers = [9, 16, 22, 29, 36, 42]

################ Sections for each different detector

####### Fourier section

def calculate_fourier_spectrum(im, typ='MFS'):
    im = im.float()
    im = im.cpu()
    im = im.data.numpy() # Transform to numpy
    fft = np.fft.fft2(im)
    if typ == 'MFS':
        fourier_spectrum = np.abs(fft)
    elif typ == 'PFS':
        fourier_spectrum = np.abs(np.angle(fft))
    return fourier_spectrum

def calculate_spectra(images, typ='MFS'):
    fs = []
    for i in range(len(images)):
        image = images[i]
        fourier_image = calculate_fourier_spectrum(image, typ=typ)
        fs.append(fourier_image.flatten())
    return fs

### Fourier Input
print('Extracting ' + detector + ' characteristic...')
if detector == 'InputMFS':
    mfs = calculate_spectra(images)
    mfs_advs = calculate_spectra(images_advs)

    characteristics       = np.asarray(mfs, dtype=np.float32)
    characteristics_adv   = np.asarray(mfs_advs, dtype=np.float32)

elif detector == 'InputPFS':
    pfs = calculate_spectra(images, typ='PFS')
    pfs_advs = calculate_spectra(images_advs, typ='PFS')

    characteristics       = np.asarray(pfs, dtype=np.float32)
    characteristics_adv   = np.asarray(pfs_advs, dtype=np.float32)

### Fourier Layer
elif detector == 'LayerMFS':
    mfs = []
    mfs_advs = []
    layers = fourier_act_layers
    for i in tqdm(range(number_images)):
        image = images[i].unsqueeze_(0)
        adv = images_advs[i].unsqueeze_(0)
        image = mnist_normalize(image)
        adv = mnist_normalize(adv)
        image_feature_maps = get_layer_feature_maps(image, layers)
        adv_feature_maps = get_layer_feature_maps(adv, layers)
        fourier_maps = calculate_spectra(image_feature_maps)
        fourier_maps_adv = calculate_spectra(adv_feature_maps)
        mfs.append(np.hstack(fourier_maps))
        mfs_advs.append(np.hstack(fourier_maps_adv))

    characteristics       = np.asarray(mfs, dtype=np.float32)
    characteristics_adv   = np.asarray(mfs_advs, dtype=np.float32)

elif detector == 'LayerPFS':
    pfs = []
    pfs_advs = []
    layers = fourier_act_layers
    for i in tqdm(range(number_images)):
        image = images[i].unsqueeze_(0)
        adv = images_advs[i].unsqueeze_(0)
        image = mnist_normalize(image)
        adv = mnist_normalize(adv)
        image_feature_maps = get_layer_feature_maps(image, layers)
        adv_feature_maps = get_layer_feature_maps(adv, layers)
        fourier_maps = calculate_spectra(image_feature_maps, typ='PFS')
        fourier_maps_adv = calculate_spectra(adv_feature_maps, typ='PFS')
        pfs.append(np.hstack(fourier_maps))
        pfs_advs.append(np.hstack(fourier_maps_adv))

    characteristics       = np.asarray(pfs, dtype=np.float32)
    characteristics_adv   = np.asarray(pfs_advs, dtype=np.float32)

####### LID section
elif detector == 'LID':
    # Hyperparameters
    batch_size = 100
    k = 20  # Number of nearest neighbors for LID

    def mle_batch(data, batch, k):
        data = np.asarray(data, dtype=np.float32)
        batch = np.asarray(batch, dtype=np.float32)
        k = min(k, len(data) - 1)
        distances = cdist(batch, data, metric='euclidean')
        distances = np.sort(distances, axis=1)[:, 1:k+1]
        lid_score = -np.mean(np.log(distances[:, -1] / distances[:, :-1]), axis=1)
        return lid_score

    # Compute LID scores for original and adversarial images
    print('Computing LID scores...')
    lid_scores = []
    lid_scores_adv = []

    images = torch.stack(images)
    images_advs = torch.stack(images_advs)

    for i in tqdm(range(0, len(images), batch_size)):
        batch = images[i:i+batch_size]
        batch_adv = images_advs[i:i+batch_size]
        batch_lid = mle_batch(images, batch, k)
        batch_lid_adv = mle_batch(images_advs, batch_adv, k)
        lid_scores.extend(batch_lid)
        lid_scores_adv.extend(batch_lid_adv)

    characteristics = np.asarray(lid_scores, dtype=np.float32)
    characteristics_adv = np.asarray(lid_scores_adv, dtype=np.float32)

####### Mahalanobis section
elif detector == 'Mahalanobis':
    def mahalanobis_distance(sample, mean, cov_inv):
        diff = sample - mean
        dist = np.dot(np.dot(diff, cov_inv), diff.T)
        return dist

    def get_mahalanobis_score(images, model, layers):
        scores = []
        for i in tqdm(range(len(images))):
            image = images[i].unsqueeze_(0)
            image = mnist_normalize(image)
            feature_maps = get_layer_feature_maps(image, layers)
            layer_means = [torch.mean(fm, dim=0) for fm in feature_maps]
            layer_covs = [torch.cov(fm.reshape(fm.shape[0], -1), rowvar=False) for fm in feature_maps]
            layer_cov_invs = [np.linalg.inv(cov.numpy()) for cov in layer_covs]

            sample_scores = []
            for fm, mean, cov_inv in zip(feature_maps, layer_means, layer_cov_invs):
                sample_scores.append(mahalanobis_distance(fm.numpy().flatten(), mean.numpy().flatten(), cov_inv))

            scores.append(np.sum(sample_scores))
        return scores

    print('Computing Mahalanobis scores...')
    layers = act_layers
    mahalanobis_scores = get_mahalanobis_score(images, model, layers)
    mahalanobis_scores_adv = get_mahalanobis_score(images_advs, model, layers)

    characteristics = np.asarray(mahalanobis_scores, dtype=np.float32)
    characteristics_adv = np.asarray(mahalanobis_scores_adv, dtype=np.float32)

# Save the extracted characteristics
np.save('./data/characteristics/'+net+'_'+attack_method+'_'+detector,characteristics)
np.save('./data/characteristics/'+net+'_'+attack_method+'_'+detector+'_adv',characteristics_adv)
print('Characteristics extracted and saved successfully.')
